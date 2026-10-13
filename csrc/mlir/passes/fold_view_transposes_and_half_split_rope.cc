/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "csrc/mlir/passes/fold_view_transposes_and_half_split_rope.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

bool IsIdentityPerm(llvm::ArrayRef<int64_t> perm) {
  for (int64_t i = 0; i < static_cast<int64_t>(perm.size()); ++i) {
    if (perm[i] != i) {
      return false;
    }
  }
  return true;
}

llvm::SmallVector<int64_t> GetNonUnitDimSizes(llvm::ArrayRef<int64_t> shape) {
  llvm::SmallVector<int64_t> sizes;
  for (int64_t d : shape) {
    if (d > 1) {
      sizes.push_back(d);
    }
  }
  return sizes;
}

bool ComputeSqueezedPermutation(llvm::ArrayRef<int64_t> in_shape,
                                llvm::ArrayRef<int64_t> perm,
                                llvm::SmallVectorImpl<int64_t>& squeezed_perm) {
  if (in_shape.size() != perm.size()) {
    return false;
  }
  llvm::SmallVector<int64_t> axis_to_squeezed(in_shape.size(), -1);
  int64_t next_idx = 0;
  for (size_t i = 0; i < in_shape.size(); ++i) {
    if (in_shape[i] > 1) {
      axis_to_squeezed[i] = next_idx++;
    }
  }
  squeezed_perm.clear();
  squeezed_perm.reserve(next_idx);
  for (int64_t d : perm) {
    if (d < 0 || d >= static_cast<int64_t>(in_shape.size())) {
      return false;
    }
    if (axis_to_squeezed[d] != -1) {
      squeezed_perm.push_back(axis_to_squeezed[d]);
    }
  }
  return static_cast<int64_t>(squeezed_perm.size()) == next_idx;
}

mlir::Value PeelUnitDimPreservingReshapes(
    mlir::Value val, llvm::ArrayRef<int64_t> non_unit_sizes) {
  while (auto resh = val.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    auto in_type =
        mlir::dyn_cast<mlir::RankedTensorType>(resh.getOperand().getType());
    if (!in_type || !in_type.hasStaticShape() ||
        GetNonUnitDimSizes(in_type.getShape()) != non_unit_sizes) {
      break;
    }
    val = resh.getOperand();
  }
  return val;
}

mlir::Value EmitReshapeIfNeeded(mlir::OpBuilder& builder, mlir::Location loc,
                                mlir::Value val,
                                mlir::RankedTensorType dst_type) {
  if (auto resh = val.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    auto op_type =
        mlir::dyn_cast<mlir::RankedTensorType>(resh.getOperand().getType());
    if (op_type && op_type.hasStaticShape() &&
        op_type.getNumElements() == dst_type.getNumElements()) {
      val = resh.getOperand();
    }
  }
  if (val.getType() == dst_type) {
    return val;
  }
  return builder.create<mlir::stablehlo::ReshapeOp>(loc, dst_type, val);
}

void EraseTriviallyDeadOps(mlir::func::FuncOp func) {
  bool changed = true;
  while (changed) {
    changed = false;
    llvm::SmallVector<mlir::Operation*> dead_ops;
    func.walk([&](mlir::Operation* op) {
      if (mlir::isOpTriviallyDead(op)) {
        dead_ops.push_back(op);
      }
    });
    for (auto it = dead_ops.rbegin(); it != dead_ops.rend(); ++it) {
      if (mlir::isOpTriviallyDead(*it)) {
        (*it)->erase();
        changed = true;
      }
    }
  }
}

// Cancels inverse transposes separated by unit-dimension reshapes or a
// single-dimension slice:
// 1. `TransposeOp(SliceOp(TransposeOp(x, perm1), starts, limits), perm2)` ->
//    `SliceOp(x, orig_starts, orig_limits)` when `perm2` is the inverse of
//    `perm1` and `SliceOp` slices a single dimension.
// 2. `TransposeOp([unit-dim ReshapeOp](TransposeOp(x, perm1)), perm2)` ->
//    `ReshapeOp(x)` when the composed non-unit permutation is the identity.
void CancelInverseViewTransposes(mlir::func::FuncOp func) {
  bool changed = true;
  while (changed) {
    changed = false;
    llvm::SmallVector<mlir::stablehlo::TransposeOp> tr_ops;
    func.walk([&](mlir::stablehlo::TransposeOp op) { tr_ops.push_back(op); });

    for (mlir::stablehlo::TransposeOp tr2 : tr_ops) {
      if (!tr2 || tr2->use_empty()) {
        continue;
      }
      auto tr2_type = mlir::dyn_cast<mlir::RankedTensorType>(tr2.getType());
      auto tr2_in_type =
          mlir::dyn_cast<mlir::RankedTensorType>(tr2.getOperand().getType());
      if (!tr2_type || !tr2_type.hasStaticShape() || !tr2_in_type ||
          !tr2_in_type.hasStaticShape()) {
        continue;
      }

      if (auto slice_op =
              tr2.getOperand().getDefiningOp<mlir::stablehlo::SliceOp>()) {
        if (auto tr1 = slice_op.getOperand()
                           .getDefiningOp<mlir::stablehlo::TransposeOp>()) {
          auto x_type = mlir::dyn_cast<mlir::RankedTensorType>(
              tr1.getOperand().getType());
          auto tr1_type = mlir::dyn_cast<mlir::RankedTensorType>(tr1.getType());
          llvm::ArrayRef<int64_t> perm1 = tr1.getPermutation();
          llvm::ArrayRef<int64_t> perm2 = tr2.getPermutation();
          if (x_type && x_type.hasStaticShape() && tr1_type &&
              tr1_type.hasStaticShape() && x_type.getNumElements() >= 8192 &&
              perm1.size() == perm2.size()) {
            const int64_t rank = perm1.size();
            bool is_inverse = true;
            for (int64_t j = 0; j < rank; ++j) {
              if (perm2[j] < 0 || perm2[j] >= rank || perm1[perm2[j]] != j) {
                is_inverse = false;
                break;
              }
              if (perm1[j] != j && tr1_type.getDimSize(j) > 1 &&
                  tr1_type.getDimSize(j) < 4) {
                is_inverse = false;
                break;
              }
            }
            llvm::ArrayRef<int64_t> starts = slice_op.getStartIndices();
            llvm::ArrayRef<int64_t> limits = slice_op.getLimitIndices();
            llvm::ArrayRef<int64_t> strides = slice_op.getStrides();
            int64_t num_sliced_dims = 0;
            for (int64_t i = 0; i < rank; ++i) {
              if (starts[i] != 0 || limits[i] != tr1_type.getDimSize(i) ||
                  strides[i] != 1) {
                ++num_sliced_dims;
              }
            }
            if (is_inverse && num_sliced_dims == 1) {
              llvm::SmallVector<int64_t> orig_starts(rank);
              llvm::SmallVector<int64_t> orig_limits(rank);
              llvm::SmallVector<int64_t> orig_strides(rank);
              for (int64_t i = 0; i < rank; ++i) {
                orig_starts[perm1[i]] = starts[i];
                orig_limits[perm1[i]] = limits[i];
                orig_strides[perm1[i]] = strides[i];
              }
              mlir::OpBuilder builder(tr2);
              auto new_slice = builder.create<mlir::stablehlo::SliceOp>(
                  tr2.getLoc(), tr2_type, tr1.getOperand(),
                  builder.getDenseI64ArrayAttr(orig_starts),
                  builder.getDenseI64ArrayAttr(orig_limits),
                  builder.getDenseI64ArrayAttr(orig_strides));
              tr2.replaceAllUsesWith(new_slice.getResult());
              changed = true;
              continue;
            }
          }
        }
      }

      llvm::SmallVector<int64_t> non_unit_before_tr2 =
          GetNonUnitDimSizes(tr2_in_type.getShape());
      llvm::SmallVector<int64_t> squeezed_perm2;
      if (!ComputeSqueezedPermutation(tr2_in_type.getShape(),
                                      tr2.getPermutation(), squeezed_perm2) ||
          IsIdentityPerm(squeezed_perm2)) {
        continue;
      }

      mlir::Value peeled =
          PeelUnitDimPreservingReshapes(tr2.getOperand(), non_unit_before_tr2);
      auto tr1 = peeled.getDefiningOp<mlir::stablehlo::TransposeOp>();
      if (!tr1) {
        continue;
      }
      auto tr1_in_type =
          mlir::dyn_cast<mlir::RankedTensorType>(tr1.getOperand().getType());
      llvm::SmallVector<int64_t> squeezed_perm1;
      if (!tr1_in_type || !tr1_in_type.hasStaticShape() ||
          !ComputeSqueezedPermutation(tr1_in_type.getShape(),
                                      tr1.getPermutation(), squeezed_perm1) ||
          squeezed_perm1.size() != squeezed_perm2.size() ||
          IsIdentityPerm(squeezed_perm1)) {
        continue;
      }
      llvm::SmallVector<int64_t> composed_perm(squeezed_perm2.size());
      for (size_t i = 0; i < squeezed_perm2.size(); ++i) {
        composed_perm[i] = squeezed_perm1[squeezed_perm2[i]];
      }
      if (IsIdentityPerm(composed_perm)) {
        llvm::SmallVector<int64_t> orig_non_unit =
            GetNonUnitDimSizes(tr1_in_type.getShape());
        mlir::Value root_val =
            PeelUnitDimPreservingReshapes(tr1.getOperand(), orig_non_unit);
        mlir::OpBuilder builder(tr2);
        mlir::Value rep =
            EmitReshapeIfNeeded(builder, tr2.getLoc(), root_val, tr2_type);
        tr2.replaceAllUsesWith(rep);
        changed = true;
      }
    }
    if (changed) {
      EraseTriviallyDeadOps(func);
    }
  }
}

struct HeadSplitDesc {
  mlir::Value src;
  int64_t b;
  int64_t h;
  int64_t s;
  int64_t d_full;
  int64_t slice_start;
  int64_t slice_limit;
  bool is_sliced;
};

// Matches a 4D head-split transpose producing `[B, H, S, d_out]` from a flat
// projection `src` of total elements `B * S * H * D_full`:
// - Full view: `Transpose(Reshape(src, [S, H, D, 1]), {3, 1, 0, 2})` or
//   `Transpose(Reshape(src, [B, S, H, D]), {0, 2, 1, 3})` or
//   `Reshape(Transpose(Reshape(src, [S, H, D]), {1, 0, 2}), [1, H, S, D])`.
// - Sliced view: `Transpose([Reshape(..., 1)](Slice(Reshape(src, [..., H, D]),
//   [..., start:limit])), perm)`.
bool MatchHeadSplitTransposeOp(mlir::Operation* op, HeadSplitDesc& desc) {
  auto out_type =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!out_type || !out_type.hasStaticShape() || out_type.getRank() != 4 ||
      !mlir::isa<mlir::FloatType>(out_type.getElementType())) {
    return false;
  }
  const int64_t b = out_type.getDimSize(0);
  const int64_t h = out_type.getDimSize(1);
  const int64_t s = out_type.getDimSize(2);
  const int64_t d_out = out_type.getDimSize(3);
  if (b < 1 || h < 2 || s < 16 || d_out < 16) {
    return false;
  }

  if (auto outer_resh = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(op)) {
    if (b != 1) {
      return false;
    }
    auto tr3 =
        outer_resh.getOperand().getDefiningOp<mlir::stablehlo::TransposeOp>();
    if (!tr3 || tr3.getPermutation() != llvm::ArrayRef<int64_t>({1, 0, 2})) {
      return false;
    }
    auto r3 = tr3.getOperand().getDefiningOp<mlir::stablehlo::ReshapeOp>();
    if (!r3) {
      return false;
    }
    auto r3_type = mlir::dyn_cast<mlir::RankedTensorType>(r3.getType());
    if (!r3_type ||
        r3_type.getShape() != llvm::ArrayRef<int64_t>({s, h, d_out})) {
      return false;
    }
    desc = {r3.getOperand(), b, h, s, d_out, 0, d_out, false};
    return true;
  }

  auto tr_op = mlir::dyn_cast<mlir::stablehlo::TransposeOp>(op);
  if (!tr_op) {
    return false;
  }
  llvm::ArrayRef<int64_t> perm = tr_op.getPermutation();
  mlir::Value tr_in = tr_op.getOperand();

  if (b == 1 && perm == llvm::ArrayRef<int64_t>({3, 1, 0, 2})) {
    auto r4 = tr_in.getDefiningOp<mlir::stablehlo::ReshapeOp>();
    if (!r4) {
      return false;
    }
    auto r4_type = mlir::dyn_cast<mlir::RankedTensorType>(r4.getType());
    if (!r4_type ||
        r4_type.getShape() != llvm::ArrayRef<int64_t>({s, h, d_out, 1})) {
      return false;
    }
    mlir::Value r4_in = r4.getOperand();
    if (auto slice_op = r4_in.getDefiningOp<mlir::stablehlo::SliceOp>()) {
      auto r3 =
          slice_op.getOperand().getDefiningOp<mlir::stablehlo::ReshapeOp>();
      if (!r3) {
        return false;
      }
      auto r3_type = mlir::dyn_cast<mlir::RankedTensorType>(r3.getType());
      if (!r3_type || !r3_type.hasStaticShape() || r3_type.getRank() != 3 ||
          r3_type.getDimSize(0) != s || r3_type.getDimSize(1) != h) {
        return false;
      }
      llvm::ArrayRef<int64_t> starts = slice_op.getStartIndices();
      llvm::ArrayRef<int64_t> limits = slice_op.getLimitIndices();
      llvm::ArrayRef<int64_t> strides = slice_op.getStrides();
      if (starts[0] != 0 || limits[0] != s || strides[0] != 1 ||
          starts[1] != 0 || limits[1] != h || strides[1] != 1 ||
          strides[2] != 1 || limits[2] - starts[2] != d_out) {
        return false;
      }
      desc = {r3.getOperand(),       b,         h,         s,
              r3_type.getDimSize(2), starts[2], limits[2], true};
      return true;
    }
    desc = {r4_in, b, h, s, d_out, 0, d_out, false};
    return true;
  }

  if (perm == llvm::ArrayRef<int64_t>({0, 2, 1, 3})) {
    if (auto slice_op = tr_in.getDefiningOp<mlir::stablehlo::SliceOp>()) {
      auto r4 =
          slice_op.getOperand().getDefiningOp<mlir::stablehlo::ReshapeOp>();
      if (!r4) {
        return false;
      }
      auto r4_type = mlir::dyn_cast<mlir::RankedTensorType>(r4.getType());
      if (!r4_type || !r4_type.hasStaticShape() || r4_type.getRank() != 4 ||
          r4_type.getDimSize(0) != b || r4_type.getDimSize(1) != s ||
          r4_type.getDimSize(2) != h) {
        return false;
      }
      llvm::ArrayRef<int64_t> starts = slice_op.getStartIndices();
      llvm::ArrayRef<int64_t> limits = slice_op.getLimitIndices();
      llvm::ArrayRef<int64_t> strides = slice_op.getStrides();
      if (starts[0] != 0 || limits[0] != b || strides[0] != 1 ||
          starts[1] != 0 || limits[1] != s || strides[1] != 1 ||
          starts[2] != 0 || limits[2] != h || strides[2] != 1 ||
          strides[3] != 1 || limits[3] - starts[3] != d_out) {
        return false;
      }
      desc = {r4.getOperand(),       b,         h,         s,
              r4_type.getDimSize(3), starts[3], limits[3], true};
      return true;
    }
    if (auto r4 = tr_in.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
      auto r4_type = mlir::dyn_cast<mlir::RankedTensorType>(r4.getType());
      if (r4_type &&
          r4_type.getShape() == llvm::ArrayRef<int64_t>({b, s, h, d_out})) {
        desc = {r4.getOperand(), b, h, s, d_out, 0, d_out, false};
        return true;
      }
    }
  }
  return false;
}

// Commutes minor-dimension slices past 4D attention head-split transposes and
// unifies duplicate `[B, H, S, D]` head-split transposes on the same `src`
// projection tensor when `src` has sliced half-split rotary views.
void UnifyHeadSplitTransposesAndCommuteSlices(mlir::func::FuncOp func) {
  struct Key {
    mlir::Value src;
    int64_t b;
    int64_t h;
    int64_t s;
    int64_t d_full;
    mlir::Value unified_val;
  };
  llvm::SmallVector<Key> cache;
  llvm::SmallVector<Key> sliced_keys;

  llvm::SmallVector<mlir::Operation*> candidates;
  func.walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::TransposeOp, mlir::stablehlo::ReshapeOp>(
            op)) {
      candidates.push_back(op);
    }
  });

  // First pass: record which (src, b, h, s, d_full) keys have sliced
  // half-split views so projections without sliced views (such as V
  // projections or non-RoPE models) are never rewritten.
  for (mlir::Operation* op : candidates) {
    if (!op || op->use_empty()) {
      continue;
    }
    HeadSplitDesc desc;
    if (!MatchHeadSplitTransposeOp(op, desc) || !desc.is_sliced) {
      continue;
    }
    if (desc.d_full < 64 || desc.d_full % 64 != 0) {
      continue;
    }
    // When `h` is not sublane-tile-aligned (`h > 16 && h % 8 != 0`) and `s` is
    // a large multiple of the 128-lane tile width (`s >= 2048 && s % 128 ==
    // 0`), XLA lays out `[B, H, S, D]` with `S` minor and `H` untiled major for
    // zero tile padding; keep the sliced views so XLA does not insert layout
    // transposes before and after a `DotGeneralOp` on `D`.
    if (desc.h > 16 && desc.h % 8 != 0 && desc.s >= 2048 && desc.s % 128 == 0) {
      continue;
    }
    bool found = false;
    for (const Key& k : sliced_keys) {
      if (k.src == desc.src && k.b == desc.b && k.h == desc.h &&
          k.s == desc.s && k.d_full == desc.d_full) {
        found = true;
        break;
      }
    }
    if (!found) {
      sliced_keys.push_back(
          {desc.src, desc.b, desc.h, desc.s, desc.d_full, nullptr});
    }
  }
  if (sliced_keys.empty()) {
    return;
  }

  for (mlir::Operation* op : candidates) {
    if (!op || op->use_empty()) {
      continue;
    }
    HeadSplitDesc desc;
    if (!MatchHeadSplitTransposeOp(op, desc)) {
      continue;
    }
    bool has_sliced = false;
    for (const Key& k : sliced_keys) {
      if (k.src == desc.src && k.b == desc.b && k.h == desc.h &&
          k.s == desc.s && k.d_full == desc.d_full) {
        has_sliced = true;
        break;
      }
    }
    if (!has_sliced) {
      continue;
    }

    mlir::Value unified;
    for (const Key& entry : cache) {
      if (entry.src == desc.src && entry.b == desc.b && entry.h == desc.h &&
          entry.s == desc.s && entry.d_full == desc.d_full) {
        unified = entry.unified_val;
        break;
      }
    }
    if (!unified) {
      mlir::OpBuilder builder(op);
      mlir::Location loc = op->getLoc();
      mlir::Type elem_type =
          mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType())
              .getElementType();
      auto full_4d_type = mlir::RankedTensorType::get(
          {desc.b, desc.h, desc.s, desc.d_full}, elem_type);
      if (desc.b == 1) {
        auto r3_type = mlir::RankedTensorType::get(
            {desc.s, desc.h, desc.d_full}, elem_type);
        mlir::Value r3 = EmitReshapeIfNeeded(builder, loc, desc.src, r3_type);
        mlir::Value t3 = builder.create<mlir::stablehlo::TransposeOp>(
            loc, r3, builder.getDenseI64ArrayAttr({1, 0, 2}));
        unified =
            builder.create<mlir::stablehlo::ReshapeOp>(loc, full_4d_type, t3);
      } else {
        auto r4_type = mlir::RankedTensorType::get(
            {desc.b, desc.s, desc.h, desc.d_full}, elem_type);
        mlir::Value r4 = EmitReshapeIfNeeded(builder, loc, desc.src, r4_type);
        unified = builder.create<mlir::stablehlo::TransposeOp>(
            loc, full_4d_type, r4, builder.getDenseI64ArrayAttr({0, 2, 1, 3}));
      }
      cache.push_back({desc.src, desc.b, desc.h, desc.s, desc.d_full, unified});
    }

    if (!desc.is_sliced) {
      if (op->getResult(0) != unified) {
        op->getResult(0).replaceAllUsesWith(unified);
      }
    } else {
      mlir::OpBuilder builder(op);
      auto sliced = builder.create<mlir::stablehlo::SliceOp>(
          op->getLoc(), op->getResult(0).getType(), unified,
          builder.getDenseI64ArrayAttr({0, 0, 0, desc.slice_start}),
          builder.getDenseI64ArrayAttr(
              {desc.b, desc.h, desc.s, desc.slice_limit}),
          builder.getDenseI64ArrayAttr({1, 1, 1, 1}));
      op->getResult(0).replaceAllUsesWith(sliced.getResult());
    }
  }
  EraseTriviallyDeadOps(func);
}

bool MatchMinorHalfSlice(mlir::Value val, int64_t half_d, bool is_upper,
                         mlir::Value& base_val) {
  auto val_type = mlir::dyn_cast<mlir::RankedTensorType>(val.getType());
  if (!val_type || !val_type.hasStaticShape() || val_type.getRank() < 2 ||
      val_type.getDimSize(val_type.getRank() - 1) != half_d) {
    return false;
  }
  llvm::SmallVector<int64_t> leading_non_unit =
      GetNonUnitDimSizes(val_type.getShape().drop_back(1));

  mlir::Value cur = val;
  if (auto resh = cur.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    auto r_in_type =
        mlir::dyn_cast<mlir::RankedTensorType>(resh.getOperand().getType());
    if (r_in_type && r_in_type.hasStaticShape() && r_in_type.getRank() >= 2 &&
        r_in_type.getDimSize(r_in_type.getRank() - 1) == half_d &&
        GetNonUnitDimSizes(r_in_type.getShape().drop_back(1)) ==
            leading_non_unit) {
      cur = resh.getOperand();
    }
  }
  auto slice_op = cur.getDefiningOp<mlir::stablehlo::SliceOp>();
  if (!slice_op) {
    return false;
  }
  auto src_type =
      mlir::dyn_cast<mlir::RankedTensorType>(slice_op.getOperand().getType());
  if (!src_type || !src_type.hasStaticShape() || src_type.getRank() < 2 ||
      src_type.getDimSize(src_type.getRank() - 1) != 2 * half_d ||
      GetNonUnitDimSizes(src_type.getShape().drop_back(1)) !=
          leading_non_unit) {
    return false;
  }
  const int64_t s_rank = src_type.getRank();
  llvm::ArrayRef<int64_t> starts = slice_op.getStartIndices();
  llvm::ArrayRef<int64_t> limits = slice_op.getLimitIndices();
  llvm::ArrayRef<int64_t> strides = slice_op.getStrides();
  for (int64_t i = 0; i < s_rank - 1; ++i) {
    if (starts[i] != 0 || limits[i] != src_type.getDimSize(i) ||
        strides[i] != 1) {
      return false;
    }
  }
  const int64_t expected_start = is_upper ? half_d : 0;
  const int64_t expected_limit = is_upper ? 2 * half_d : half_d;
  if (starts[s_rank - 1] != expected_start ||
      limits[s_rank - 1] != expected_limit || strides[s_rank - 1] != 1) {
    return false;
  }

  mlir::Value base = slice_op.getOperand();
  if (auto base_resh = base.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
    auto b_in_type = mlir::dyn_cast<mlir::RankedTensorType>(
        base_resh.getOperand().getType());
    if (b_in_type && b_in_type.hasStaticShape() && b_in_type.getRank() >= 2 &&
        b_in_type.getDimSize(b_in_type.getRank() - 1) == 2 * half_d &&
        GetNonUnitDimSizes(b_in_type.getShape().drop_back(1)) ==
            leading_non_unit) {
      base = base_resh.getOperand();
    }
  }
  base_val = base;
  return true;
}

// Builds the `D x D` skew-symmetric half-rotation matrix `J`:
// - If `negate_upper_in_first_half` is true (forward `rotate_half(x) =
//   [-x_hi, x_lo]`):
//   `J[half_d + k, k] = -1` and `J[k, half_d + k] = +1` for `k in [0, half_d)`.
// - If `negate_upper_in_first_half` is false (backward `[x_hi, -x_lo]`):
//   `J[half_d + k, k] = +1` and `J[k, half_d + k] = -1` for `k in [0, half_d)`.
mlir::Value BuildHalfSplitRotationMatrix(mlir::OpBuilder& builder,
                                         mlir::Location loc,
                                         mlir::Type elem_type, int64_t half_d,
                                         bool negate_upper_in_first_half) {
  const int64_t d = 2 * half_d;
  auto float_type = mlir::cast<mlir::FloatType>(elem_type);
  llvm::APFloat zero(float_type.getFloatSemantics(), 0);
  llvm::APFloat pos_one(float_type.getFloatSemantics(), 1);
  llvm::APFloat neg_one = pos_one;
  neg_one.changeSign();

  llvm::SmallVector<llvm::APFloat> vals(d * d, zero);
  for (int64_t k = 0; k < half_d; ++k) {
    vals[(half_d + k) * d + k] = negate_upper_in_first_half ? neg_one : pos_one;
    vals[k * d + (half_d + k)] = negate_upper_in_first_half ? pos_one : neg_one;
  }
  auto mat_type = mlir::RankedTensorType::get({d, d}, float_type);
  return builder.create<mlir::stablehlo::ConstantOp>(
      loc, mlir::DenseElementsAttr::get(mat_type, vals));
}

mlir::Value EmitMinorDimContraction(mlir::OpBuilder& builder,
                                    mlir::Location loc,
                                    mlir::RankedTensorType out_type,
                                    mlir::Value x, mlir::Value mat) {
  const int64_t rank = out_type.getRank();
  auto dnums = mlir::stablehlo::DotDimensionNumbersAttr::get(
      builder.getContext(), /*lhsBatchingDimensions=*/{},
      /*rhsBatchingDimensions=*/{}, /*lhsContractingDimensions=*/{rank - 1},
      /*rhsContractingDimensions=*/{0});
  return builder.create<mlir::stablehlo::DotGeneralOp>(
      loc, out_type, x, mat, dnums, /*precision_config=*/nullptr,
      /*algorithm=*/nullptr);
}

bool TryFoldHalfSplitRotaryConcat(mlir::stablehlo::ConcatenateOp concat_op) {
  auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(concat_op.getType());
  if (!out_type || !out_type.hasStaticShape() ||
      concat_op.getInputs().size() != 2) {
    return false;
  }
  const int64_t rank = out_type.getRank();
  if (rank < 2 || static_cast<int64_t>(concat_op.getDimension()) != rank - 1) {
    return false;
  }
  mlir::Type elem_type = out_type.getElementType();
  if (!mlir::isa<mlir::FloatType>(elem_type)) {
    return false;
  }
  const int64_t d = out_type.getDimSize(rank - 1);
  if (d < 64 || d > 512 || d % 64 != 0) {
    return false;
  }
  const int64_t half_d = d / 2;
  const int64_t seq_dim = out_type.getDimSize(rank - 2);
  int64_t leading_elems = 1;
  for (int64_t i = 0; i < rank - 1; ++i) {
    leading_elems *= out_type.getDimSize(i);
  }
  if (seq_dim < 32 || leading_elems < 256) {
    return false;
  }
  // When `d == 64` (half of the 128-lane TPU vector register width), only
  // lower to `DotGeneralOp` on long sequences (`seq_dim >= 512` and
  // `leading_elems >= 8192`) where eliminating the sub-tile slices and
  // `ConcatBitcast` async DMAs outweighs the 64->128 lane padding.
  if (d < 128 && (seq_dim < 512 || leading_elems < 8192)) {
    return false;
  }
  if (rank >= 3) {
    const int64_t head_dim = out_type.getDimSize(rank - 3);
    if (head_dim > 16 && head_dim % 8 != 0 && seq_dim >= 2048 &&
        seq_dim % 128 == 0) {
      return false;
    }
  }

  mlir::Value lhs = concat_op.getInputs()[0];
  mlir::Value rhs = concat_op.getInputs()[1];
  auto lhs_type = mlir::dyn_cast<mlir::RankedTensorType>(lhs.getType());
  auto rhs_type = mlir::dyn_cast<mlir::RankedTensorType>(rhs.getType());
  if (!lhs_type || !rhs_type || lhs_type != rhs_type ||
      lhs_type.getDimSize(rank - 1) != half_d) {
    return false;
  }

  mlir::OpBuilder builder(concat_op);
  mlir::Location loc = concat_op.getLoc();

  // Form 1a: `Concatenate(Neg(x_hi), x_lo, dim = rank - 1)` -> `x @ J_fwd`
  if (auto neg_lhs = lhs.getDefiningOp<mlir::stablehlo::NegOp>()) {
    mlir::Value base_hi;
    mlir::Value base_lo;
    if (MatchMinorHalfSlice(neg_lhs.getOperand(), half_d, /*is_upper=*/true,
                            base_hi) &&
        MatchMinorHalfSlice(rhs, half_d, /*is_upper=*/false, base_lo) &&
        base_hi == base_lo) {
      mlir::Value x = EmitReshapeIfNeeded(builder, loc, base_lo, out_type);
      mlir::Value j_fwd =
          BuildHalfSplitRotationMatrix(builder, loc, elem_type, half_d,
                                       /*negate_upper_in_first_half=*/true);
      mlir::Value replacement =
          EmitMinorDimContraction(builder, loc, out_type, x, j_fwd);
      concat_op.replaceAllUsesWith(replacement);
      return true;
    }
  }

  // Form 1b: `Concatenate(x_hi, Neg(x_lo), dim = rank - 1)` -> `x @ J_bwd`
  if (auto neg_rhs = rhs.getDefiningOp<mlir::stablehlo::NegOp>()) {
    mlir::Value base_hi;
    mlir::Value base_lo;
    if (MatchMinorHalfSlice(lhs, half_d, /*is_upper=*/true, base_hi) &&
        MatchMinorHalfSlice(neg_rhs.getOperand(), half_d, /*is_upper=*/false,
                            base_lo) &&
        base_hi == base_lo) {
      mlir::Value x = EmitReshapeIfNeeded(builder, loc, base_lo, out_type);
      mlir::Value j_bwd =
          BuildHalfSplitRotationMatrix(builder, loc, elem_type, half_d,
                                       /*negate_upper_in_first_half=*/false);
      mlir::Value replacement =
          EmitMinorDimContraction(builder, loc, out_type, x, j_bwd);
      concat_op.replaceAllUsesWith(replacement);
      return true;
    }
  }

  return false;
}

void FoldHalfSplitRotaryEmbedding(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::stablehlo::ConcatenateOp> concat_ops;
  func.walk(
      [&](mlir::stablehlo::ConcatenateOp op) { concat_ops.push_back(op); });
  for (mlir::stablehlo::ConcatenateOp concat_op : concat_ops) {
    if (concat_op && !concat_op->use_empty()) {
      TryFoldHalfSplitRotaryConcat(concat_op);
    }
  }
  EraseTriviallyDeadOps(func);
}

struct FoldViewTransposesAndHalfSplitRopePass
    : public mlir::PassWrapper<FoldViewTransposesAndHalfSplitRopePass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      FoldViewTransposesAndHalfSplitRopePass)

  llvm::StringRef getArgument() const override {
    return "fold-view-transposes-and-half-split-rope";
  }
  llvm::StringRef getDescription() const override {
    return "Cancels inverse view transposes across unit-dim reshapes and "
           "slices, unifies rotary head-split transposes, and lowers "
           "half-split rotary embeddings via skew-symmetric contractions.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    CancelInverseViewTransposes(func);
    UnifyHeadSplitTransposesAndCommuteSlices(func);
    FoldHalfSplitRotaryEmbedding(func);
    CancelInverseViewTransposes(func);
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createFoldViewTransposesAndHalfSplitRopePass() {
  return std::make_unique<FoldViewTransposesAndHalfSplitRopePass>();
}

}  // namespace torch_tpu
