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

#include "csrc/mlir/passes/optimize_sdpa_softmax.h"

#include <cstdint>
#include <memory>
#include <optional>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
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

bool IsDropLastDimBroadcast(llvm::ArrayRef<int64_t> dims, int64_t rank) {
  if (static_cast<int64_t>(dims.size()) != rank - 1) {
    return false;
  }
  for (int64_t i = 0; i < rank - 1; ++i) {
    if (dims[i] != i) {
      return false;
    }
  }
  return true;
}

template <typename BinaryOpTy>
bool IsSingleDimReduceWithOp(mlir::Value val, int64_t expected_dim,
                             mlir::stablehlo::ReduceOp* out_reduce = nullptr) {
  auto reduce_op = val.getDefiningOp<mlir::stablehlo::ReduceOp>();
  if (!reduce_op || reduce_op.getInputs().size() != 1 ||
      reduce_op.getResults().size() != 1) {
    return false;
  }
  llvm::ArrayRef<int64_t> dims = reduce_op.getDimensions();
  if (dims.size() != 1 || dims[0] != expected_dim) {
    return false;
  }
  mlir::Region& body = reduce_op.getBody();
  if (!body.hasOneBlock()) {
    return false;
  }
  mlir::Block& block = body.front();
  if (block.getOperations().size() != 2) {
    return false;
  }
  if (!mlir::isa<BinaryOpTy>(block.front())) {
    return false;
  }
  if (out_reduce) {
    *out_reduce = reduce_op;
  }
  return true;
}

mlir::Value ConvertIfNeeded(mlir::OpBuilder& builder, mlir::Location loc,
                            mlir::Value val, mlir::Type target_elem_type) {
  auto type = mlir::cast<mlir::RankedTensorType>(val.getType());
  if (type.getElementType() == target_elem_type) {
    return val;
  }
  return builder.create<mlir::stablehlo::ConvertOp>(
      loc, mlir::RankedTensorType::get(type.getShape(), target_elem_type), val);
}

mlir::Value CreateSplatFloatConst(mlir::OpBuilder& builder, mlir::Location loc,
                                  mlir::RankedTensorType type, float value) {
  auto float_type = mlir::cast<mlir::FloatType>(type.getElementType());
  llvm::APFloat apf(value);
  bool loses_info = false;
  apf.convert(float_type.getFloatSemantics(),
              llvm::APFloat::rmNearestTiesToEven, &loses_info);
  auto attr = mlir::DenseElementsAttr::get(type, apf);
  return builder.create<mlir::stablehlo::ConstantOp>(loc, type, attr);
}

template <typename BinaryOpTy>
mlir::stablehlo::ReduceOp CreateLastDimFloatReduce(mlir::OpBuilder& builder,
                                                   mlir::Location loc,
                                                   mlir::Value input,
                                                   float init_val) {
  auto in_type = mlir::cast<mlir::RankedTensorType>(input.getType());
  const int64_t rank = in_type.getRank();
  auto elem_type = mlir::cast<mlir::FloatType>(in_type.getElementType());
  auto scalar_type = mlir::RankedTensorType::get({}, elem_type);
  mlir::Value init_const =
      CreateSplatFloatConst(builder, loc, scalar_type, init_val);

  llvm::SmallVector<int64_t, 4> out_shape(in_type.getShape().begin(),
                                          in_type.getShape().end() - 1);
  auto out_type = mlir::RankedTensorType::get(out_shape, elem_type);

  auto reduce_op = builder.create<mlir::stablehlo::ReduceOp>(
      loc, mlir::TypeRange{out_type}, mlir::ValueRange{input},
      mlir::ValueRange{init_const}, builder.getDenseI64ArrayAttr({rank - 1}));

  mlir::Region& region = reduce_op.getBody();
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Block* block = builder.createBlock(&region);
    block->addArgument(scalar_type, loc);
    block->addArgument(scalar_type, loc);
    builder.setInsertionPointToStart(block);
    mlir::Value res = builder.create<BinaryOpTy>(loc, block->getArgument(0),
                                                 block->getArgument(1));
    builder.create<mlir::stablehlo::ReturnOp>(loc, mlir::ValueRange{res});
  }
  return reduce_op;
}

bool IsZeroSplatConstant(mlir::Value val) {
  if (auto bcast = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
    val = bcast.getOperand();
  }
  auto const_op = val.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!const_op) {
    return false;
  }
  auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
  if (!attr || attr.empty() || !attr.isSplat() ||
      !mlir::isa<mlir::FloatType>(attr.getElementType())) {
    return false;
  }
  return attr.getSplatValue<llvm::APFloat>().isZero();
}

// Checks whether a 4D `SelectOp(BroadcastInDim(pred_3d, {0,1,2}), 0.0, x)`
// masks out rows of `x`, returning `pred_3d` and the `SelectOp` when matched.
bool MatchRowAllMaskedSelect(mlir::Value val, mlir::Value& out_pred_3d,
                             mlir::stablehlo::SelectOp& out_select) {
  auto select_op = val.getDefiningOp<mlir::stablehlo::SelectOp>();
  if (!select_op) {
    return false;
  }
  auto sel_type = mlir::dyn_cast<mlir::RankedTensorType>(select_op.getType());
  if (!sel_type || !sel_type.hasStaticShape() || sel_type.getRank() < 3) {
    return false;
  }
  if (!IsZeroSplatConstant(select_op.getOnTrue())) {
    return false;
  }
  auto pred_bcast =
      select_op.getPred().getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  if (!pred_bcast ||
      !IsDropLastDimBroadcast(pred_bcast.getBroadcastDimensions(),
                              sel_type.getRank())) {
    return false;
  }
  out_pred_3d = pred_bcast.getOperand();
  out_select = select_op;
  return true;
}

// Checks whether XLA's FlashAttention `MatchSoftmax` (online-softmax) is both
// eligible and profitable to unblock for a forward SDPA with the given
// `q_len`, `kv_len`, and `head_dim`.
bool CanMatchXlaOnlineSoftmax(int64_t q_len, int64_t kv_len, int64_t head_dim,
                              bool has_mask_or_causal) {
  if (q_len < 2048 || kv_len < 2048 || (q_len % 128) != 0 ||
      (kv_len % 128) != 0 || q_len > 8192 || kv_len > 8192) {
    return false;
  }
  if (head_dim != 64 && head_dim != 128) {
    return false;
  }
  return head_dim == 128 || has_mask_or_causal;
}

// Checks whether an attention matrix has multi-tile unaligned self-attention
// sequence lengths (`128 < q_len == kv_len < 256`, `q_len % 128 != 0`), where
// XLA's unpadded `reduce-window(size=2*L-1)` spans multiple 128-element tiles.
bool IsUnalignedMultiTileSelfAttention(int64_t q_len, int64_t kv_len) {
  return q_len == kv_len && q_len > 128 && q_len < 256 && (q_len % 128) != 0;
}

// 1. Rewrites unaligned SDPA backward `ComputeGradP`:
//   rowsum = Reduce(Mul(softmax, ds), 0.0, AddOp, dims={R-1})
//   shifted_ds = Subtract(ds, BroadcastInDim(rowsum, dims={0..R-2}))
// to:
//   neg_rowsum = Negate(rowsum)
//   shifted_ds = Add(ds, BroadcastInDim(neg_rowsum, dims={0..R-2}))
// preventing XLA `PrefuseReduceBroadcastReuse` from converting `rowsum` into a
// standalone O(L^2) `reduce-window` fusion and allowing `rowsum` to fuse into
// the `dS` contraction.
void RewriteBackwardGradPRowsum(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::stablehlo::SubtractOp> sub_ops;
  func.walk([&](mlir::stablehlo::SubtractOp op) { sub_ops.push_back(op); });

  for (mlir::stablehlo::SubtractOp sub_op : sub_ops) {
    auto sub_type = mlir::dyn_cast<mlir::RankedTensorType>(sub_op.getType());
    if (!sub_type || !sub_type.hasStaticShape() || sub_type.getRank() < 3 ||
        !mlir::isa<mlir::FloatType>(sub_type.getElementType())) {
      continue;
    }
    const int64_t rank = sub_type.getRank();
    const int64_t q_len = sub_type.getDimSize(rank - 2);
    const int64_t kv_len = sub_type.getDimSize(rank - 1);
    if (!IsUnalignedMultiTileSelfAttention(q_len, kv_len)) {
      continue;
    }
    auto bcast_op =
        sub_op.getRhs().getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
    if (!bcast_op ||
        !IsDropLastDimBroadcast(bcast_op.getBroadcastDimensions(), rank)) {
      continue;
    }
    mlir::stablehlo::ReduceOp reduce_op;
    if (!IsSingleDimReduceWithOp<mlir::stablehlo::AddOp>(
            bcast_op.getOperand(), rank - 1, &reduce_op)) {
      continue;
    }
    auto mul_op =
        reduce_op.getInputs()[0].getDefiningOp<mlir::stablehlo::MulOp>();
    if (!mul_op || (mul_op.getLhs() != sub_op.getLhs() &&
                    mul_op.getRhs() != sub_op.getLhs())) {
      continue;
    }

    mlir::OpBuilder builder(sub_op);
    mlir::Location loc = sub_op.getLoc();
    mlir::Value neg_rowsum =
        builder.create<mlir::stablehlo::NegOp>(loc, bcast_op.getOperand());
    mlir::Value neg_bcast = builder.create<mlir::stablehlo::BroadcastInDimOp>(
        loc, sub_type, neg_rowsum, bcast_op.getBroadcastDimensionsAttr());
    mlir::Value add_val =
        builder.create<mlir::stablehlo::AddOp>(loc, sub_op.getLhs(), neg_bcast);
    sub_op.replaceAllUsesWith(add_val);
  }
}

struct ShiftedLogitsMatch {
  mlir::Value masked_logits;
  mlir::stablehlo::ReduceOp max_reduce;
  mlir::stablehlo::SubtractOp shifted_sub;
};

// Matches `exp_val = Exp(OptConvert(Subtract(masked,
// Broadcast(Reduce(MAX)))))`.
std::optional<ShiftedLogitsMatch> MatchExpOfShiftedLogits(mlir::Value exp_val,
                                                          int64_t rank) {
  auto exp_op = exp_val.getDefiningOp<mlir::stablehlo::ExpOp>();
  if (!exp_op) {
    return std::nullopt;
  }
  mlir::Value logits = exp_op.getOperand();
  if (auto conv = logits.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    logits = conv.getOperand();
  }
  auto sub_op = logits.getDefiningOp<mlir::stablehlo::SubtractOp>();
  if (!sub_op) {
    return std::nullopt;
  }
  auto bcast_op =
      sub_op.getRhs().getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  if (!bcast_op ||
      !IsDropLastDimBroadcast(bcast_op.getBroadcastDimensions(), rank)) {
    return std::nullopt;
  }
  mlir::stablehlo::ReduceOp max_reduce;
  if (!IsSingleDimReduceWithOp<mlir::stablehlo::MaxOp>(bcast_op.getOperand(),
                                                       rank - 1, &max_reduce)) {
    return std::nullopt;
  }
  if (max_reduce.getInputs()[0] != sub_op.getLhs()) {
    return std::nullopt;
  }
  return ShiftedLogitsMatch{sub_op.getLhs(), max_reduce, sub_op};
}

// 2. Optimizes unaligned row-masked Backward SDPA softmax
// (`BuildScaledDotProductAttentionBackward`), identified by `safe_sum_exp =
// Select(Compare(sum_exp, 0, EQ), 1, sum_exp)`.
void RewriteBackwardSdpaSoftmax(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::Operation*> candidates;
  func.walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::DivOp, mlir::stablehlo::MulOp>(op)) {
      candidates.push_back(op);
    }
  });

  for (mlir::Operation* op : candidates) {
    if (!op || op->getParentOp() == nullptr) {
      continue;
    }
    auto type =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!type || !type.hasStaticShape() || type.getRank() != 4 ||
        !mlir::isa<mlir::FloatType>(type.getElementType())) {
      continue;
    }
    const int64_t rank = 4;
    const int64_t q_len = type.getDimSize(2);
    const int64_t kv_len = type.getDimSize(3);
    if (!IsUnalignedMultiTileSelfAttention(q_len, kv_len)) {
      continue;
    }

    mlir::Value exp_val = op->getOperand(0);
    auto bcast_op =
        op->getOperand(1).getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
    if (!bcast_op ||
        !IsDropLastDimBroadcast(bcast_op.getBroadcastDimensions(), rank)) {
      continue;
    }

    mlir::Value safe_sum_val = bcast_op.getOperand();
    if (mlir::isa<mlir::stablehlo::MulOp>(op)) {
      auto inv_div = safe_sum_val.getDefiningOp<mlir::stablehlo::DivOp>();
      if (!inv_div) {
        continue;
      }
      safe_sum_val = inv_div.getRhs();
    }

    auto safe_select = safe_sum_val.getDefiningOp<mlir::stablehlo::SelectOp>();
    if (!safe_select) {
      continue;
    }
    auto cmp_op =
        safe_select.getPred().getDefiningOp<mlir::stablehlo::CompareOp>();
    if (!cmp_op || cmp_op.getComparisonDirection() !=
                       mlir::stablehlo::ComparisonDirection::EQ) {
      continue;
    }

    auto shifted_match = MatchExpOfShiftedLogits(exp_val, rank);
    if (!shifted_match) {
      continue;
    }

    // Require an outer 4D row-masked `SelectOp` wrapping `op` (where the 4D
    // `SelectOp` prevents XLA from fusing the division into downstream users).
    if (!op->hasOneUse()) {
      continue;
    }
    mlir::stablehlo::SelectOp masked_sel;
    mlir::Value row_mask_pred_3d;
    if (!MatchRowAllMaskedSelect((*op->getUsers().begin())->getResult(0),
                                 row_mask_pred_3d, masked_sel)) {
      continue;
    }
    mlir::Operation* root_softmax_op = masked_sel.getOperation();

    mlir::OpBuilder builder(root_softmax_op);
    mlir::Location loc = root_softmax_op->getLoc();
    mlir::Type calc_elem_type = type.getElementType();
    auto reduced_3d_type = mlir::RankedTensorType::get(
        {type.getDimSize(0), type.getDimSize(1), type.getDimSize(2)},
        calc_elem_type);

    mlir::Value masked_calc = ConvertIfNeeded(
        builder, loc, shifted_match->masked_logits, calc_elem_type);
    auto max_red = CreateLastDimFloatReduce<mlir::stablehlo::MaxOp>(
        builder, loc, masked_calc, -std::numeric_limits<float>::infinity());
    mlir::Value neg_max =
        builder.create<mlir::stablehlo::NegOp>(loc, max_red.getResult(0));
    mlir::Value neg_max_bcast =
        builder.create<mlir::stablehlo::BroadcastInDimOp>(
            loc, type, neg_max, builder.getDenseI64ArrayAttr({0, 1, 2}));
    mlir::Value shifted_bwd =
        builder.create<mlir::stablehlo::AddOp>(loc, masked_calc, neg_max_bcast);
    mlir::Value exp_bwd =
        builder.create<mlir::stablehlo::ExpOp>(loc, shifted_bwd);

    mlir::Value sum_exp_3d = safe_select.getOnFalse();
    mlir::Value zero_3d =
        CreateSplatFloatConst(builder, loc, reduced_3d_type, 0.0f);
    mlir::Value one_3d =
        CreateSplatFloatConst(builder, loc, reduced_3d_type, 1.0f);
    mlir::Value is_zero = builder.create<mlir::stablehlo::CompareOp>(
        loc, sum_exp_3d, zero_3d, mlir::stablehlo::ComparisonDirection::EQ);
    mlir::Value new_safe_sum = builder.create<mlir::stablehlo::SelectOp>(
        loc, is_zero, one_3d, sum_exp_3d);
    mlir::Value inv_sum =
        builder.create<mlir::stablehlo::DivOp>(loc, one_3d, new_safe_sum);
    inv_sum = builder.create<mlir::stablehlo::SelectOp>(loc, row_mask_pred_3d,
                                                        zero_3d, inv_sum);
    mlir::Value inv_sum_bcast =
        builder.create<mlir::stablehlo::BroadcastInDimOp>(
            loc, type, inv_sum, builder.getDenseI64ArrayAttr({0, 1, 2}));
    mlir::Value new_softmax =
        builder.create<mlir::stablehlo::MulOp>(loc, exp_bwd, inv_sum_bcast);
    root_softmax_op->replaceAllUsesWith(mlir::ValueRange{new_softmax});
  }
}

struct DownstreamOutputDotMatch {
  mlir::stablehlo::DotGeneralOp dot_op;
  mlir::stablehlo::ReshapeOp gqa_probs_reshape;
  mlir::stablehlo::ReshapeOp gqa_out_reshape;
  mlir::stablehlo::ConvertOp probs_convert;
};

// Finds the forward `ComputeAttentionOutput` `DotGeneralOp` consuming
// `softmax_val` (either directly, through a `ConvertOp`, or through a 5D GQA
// `ReshapeOp`).
std::optional<DownstreamOutputDotMatch> FindDownstreamAttentionOutputDot(
    mlir::Value softmax_val) {
  for (mlir::Operation* user : softmax_val.getUsers()) {
    mlir::stablehlo::ConvertOp probs_convert;
    mlir::Value curr = softmax_val;
    if (auto conv = mlir::dyn_cast<mlir::stablehlo::ConvertOp>(user)) {
      probs_convert = conv;
      curr = conv.getResult();
    }
    for (mlir::Operation* u2 : curr.getUsers()) {
      if (auto dg = mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(u2)) {
        if (dg.getLhs() == curr) {
          return DownstreamOutputDotMatch{dg, nullptr, nullptr, probs_convert};
        }
      }
      if (auto resh = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(u2)) {
        auto r_type = mlir::dyn_cast<mlir::RankedTensorType>(resh.getType());
        if (r_type && r_type.hasStaticShape() && r_type.getRank() == 5 &&
            resh->hasOneUse()) {
          if (auto dg = mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(
                  *resh->getUsers().begin())) {
            if (dg.getLhs() == resh.getResult() && dg->hasOneUse()) {
              if (auto out_resh = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(
                      *dg->getUsers().begin())) {
                return DownstreamOutputDotMatch{dg, resh, out_resh,
                                                probs_convert};
              }
            }
          }
        }
      }
    }
  }
  return std::nullopt;
}

// 3. Optimizes Forward SDPA softmax
// (`BuildScaledDotProductFusedAttentionShloOp`).
void RewriteForwardSdpaSoftmax(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::Operation*> candidates;
  func.walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::DivOp, mlir::stablehlo::MulOp>(op)) {
      candidates.push_back(op);
    }
  });

  for (mlir::Operation* op : candidates) {
    if (!op || op->getParentOp() == nullptr) {
      continue;
    }
    auto type =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!type || !type.hasStaticShape() || type.getRank() != 4 ||
        !mlir::isa<mlir::FloatType>(type.getElementType())) {
      continue;
    }
    const int64_t rank = 4;
    const int64_t b_size = type.getDimSize(0);
    const int64_t h_q = type.getDimSize(1);
    const int64_t q_len = type.getDimSize(2);
    const int64_t kv_len = type.getDimSize(3);

    mlir::Value exp_val = op->getOperand(0);
    auto bcast_op =
        op->getOperand(1).getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
    if (!bcast_op ||
        !IsDropLastDimBroadcast(bcast_op.getBroadcastDimensions(), rank)) {
      continue;
    }

    mlir::Value sum_exp_val = bcast_op.getOperand();
    const bool is_mul_reciprocal = mlir::isa<mlir::stablehlo::MulOp>(op);
    if (is_mul_reciprocal) {
      auto inv_div = sum_exp_val.getDefiningOp<mlir::stablehlo::DivOp>();
      if (!inv_div) {
        continue;
      }
      sum_exp_val = inv_div.getRhs();
    }

    mlir::stablehlo::ReduceOp sum_reduce;
    if (!IsSingleDimReduceWithOp<mlir::stablehlo::AddOp>(sum_exp_val, rank - 1,
                                                         &sum_reduce) ||
        sum_reduce.getInputs()[0] != exp_val) {
      continue;
    }

    auto shifted_match = MatchExpOfShiftedLogits(exp_val, rank);
    if (!shifted_match) {
      continue;
    }

    mlir::Operation* root_softmax_op = op;
    mlir::stablehlo::SelectOp row_mask_select;
    mlir::Value row_mask_pred_3d;
    if (op->hasOneUse()) {
      mlir::stablehlo::SelectOp masked_sel;
      mlir::Value pred_3d;
      if (MatchRowAllMaskedSelect((*op->getUsers().begin())->getResult(0),
                                  pred_3d, masked_sel)) {
        root_softmax_op = masked_sel.getOperation();
        row_mask_select = masked_sel;
        row_mask_pred_3d = pred_3d;
      }
    }

    auto dot_match =
        FindDownstreamAttentionOutputDot(root_softmax_op->getResult(0));
    if (!dot_match) {
      continue;
    }

    mlir::Value v_4d = dot_match->dot_op.getRhs();
    auto v_type = mlir::dyn_cast<mlir::RankedTensorType>(v_4d.getType());
    if (!v_type || !v_type.hasStaticShape() || v_type.getRank() != 4) {
      continue;
    }
    const int64_t head_dim = v_type.getDimSize(3);
    const bool has_mask_or_causal =
        row_mask_select != nullptr ||
        mlir::isa_and_nonnull<mlir::stablehlo::SelectOp,
                              mlir::stablehlo::AddOp>(
            shifted_match->masked_logits.getDefiningOp());
    int exp_uses = 0;
    for (mlir::Operation* u : exp_val.getUsers()) {
      ++exp_uses;
      if (u != sum_reduce.getOperation() && u != op) {
        exp_uses = 99;
      }
    }
    const bool is_standalone_forward =
        sum_reduce->hasOneUse() && bcast_op->hasOneUse() && exp_uses == 2;
    const bool can_match_online =
        !is_mul_reciprocal && is_standalone_forward &&
        CanMatchXlaOnlineSoftmax(q_len, kv_len, head_dim, has_mask_or_causal);

    if (can_match_online) {
      // Subcase 3A: XLA `MatchSoftmax` (online-softmax) applies.
      // If `row_mask_select` or 5D GQA `DotGeneral` blocks `MatchSoftmax`,
      // normalize `probs @ V` to 4D and sink `row_mask_select` after
      // `DotGeneral`.
      if (!row_mask_select && !dot_match->gqa_probs_reshape) {
        continue;
      }
      mlir::Operation* old_out_op =
          dot_match->gqa_out_reshape ? dot_match->gqa_out_reshape.getOperation()
                                     : dot_match->dot_op.getOperation();
      auto out_4d_type = mlir::cast<mlir::RankedTensorType>(
          old_out_op->getResult(0).getType());

      mlir::OpBuilder builder(old_out_op);
      mlir::Location loc = old_out_op->getLoc();

      mlir::Value unmasked_probs = op->getResult(0);
      mlir::Value probs_for_dot = ConvertIfNeeded(builder, loc, unmasked_probs,
                                                  v_type.getElementType());

      mlir::Value v_for_dot = v_4d;
      if (dot_match->gqa_probs_reshape) {
        const int64_t h_kv = v_type.getDimSize(1);
        const int64_t g = h_q / h_kv;
        auto v_5d_type = mlir::RankedTensorType::get(
            {b_size, h_kv, g, kv_len, head_dim}, v_type.getElementType());
        mlir::Value v_5d = builder.create<mlir::stablehlo::BroadcastInDimOp>(
            loc, v_5d_type, v_4d, builder.getDenseI64ArrayAttr({0, 1, 3, 4}));
        auto v_rep_type = mlir::RankedTensorType::get(
            {b_size, h_q, kv_len, head_dim}, v_type.getElementType());
        v_for_dot =
            builder.create<mlir::stablehlo::ReshapeOp>(loc, v_rep_type, v_5d);
      }

      auto dnums = mlir::stablehlo::DotDimensionNumbersAttr::get(
          builder.getContext(), /*lhsBatchingDimensions=*/{0, 1},
          /*rhsBatchingDimensions=*/{0, 1}, /*lhsContractingDimensions=*/{3},
          /*rhsContractingDimensions=*/{2});
      mlir::Value new_out = builder.create<mlir::stablehlo::DotGeneralOp>(
          loc, out_4d_type, probs_for_dot, v_for_dot, dnums,
          dot_match->dot_op.getPrecisionConfigAttr(),
          dot_match->dot_op.getAlgorithmAttr());

      if (row_mask_pred_3d) {
        auto pred_4d_type = mlir::RankedTensorType::get(out_4d_type.getShape(),
                                                        builder.getI1Type());
        mlir::Value pred_4d = builder.create<mlir::stablehlo::BroadcastInDimOp>(
            loc, pred_4d_type, row_mask_pred_3d,
            builder.getDenseI64ArrayAttr({0, 1, 2}));
        mlir::Value zero_out =
            CreateSplatFloatConst(builder, loc, out_4d_type, 0.0f);
        new_out = builder.create<mlir::stablehlo::SelectOp>(loc, pred_4d,
                                                            zero_out, new_out);
      }
      old_out_op->replaceAllUsesWith(mlir::ValueRange{new_out});
    } else if (IsUnalignedMultiTileSelfAttention(q_len, kv_len) &&
               row_mask_select) {
      // Subcase 3B: Unaligned row-masked forward SDPA (`128 < q_len == kv_len <
      // 256` with `row_mask_select`), where the 4D row-masked `SelectOp` blocks
      // `Div(exp, Broadcast(sum_exp))` from fusing into `DotGeneral(probs, V)`
      // and strands both `max_logits` and `sum_exp` as standalone
      // `reduce-window` output fusions.
      mlir::OpBuilder builder(root_softmax_op);
      mlir::Location loc = root_softmax_op->getLoc();
      mlir::Type calc_elem_type = type.getElementType();
      auto reduced_3d_type =
          mlir::RankedTensorType::get({b_size, h_q, q_len}, calc_elem_type);

      mlir::Value masked_calc = ConvertIfNeeded(
          builder, loc, shifted_match->masked_logits, calc_elem_type);
      auto max_red = CreateLastDimFloatReduce<mlir::stablehlo::MaxOp>(
          builder, loc, masked_calc, -std::numeric_limits<float>::infinity());
      mlir::Value neg_max =
          builder.create<mlir::stablehlo::NegOp>(loc, max_red.getResult(0));
      mlir::Value neg_max_bcast =
          builder.create<mlir::stablehlo::BroadcastInDimOp>(
              loc, type, neg_max, builder.getDenseI64ArrayAttr({0, 1, 2}));
      mlir::Value new_shifted = builder.create<mlir::stablehlo::AddOp>(
          loc, masked_calc, neg_max_bcast);
      mlir::Value new_exp =
          builder.create<mlir::stablehlo::ExpOp>(loc, new_shifted);
      auto new_sum_red = CreateLastDimFloatReduce<mlir::stablehlo::AddOp>(
          builder, loc, new_exp, 0.0f);
      mlir::Value new_sum = new_sum_red.getResult(0);

      mlir::Value one_3d =
          CreateSplatFloatConst(builder, loc, reduced_3d_type, 1.0f);
      mlir::Value inv_sum =
          builder.create<mlir::stablehlo::DivOp>(loc, one_3d, new_sum);
      mlir::Value zero_3d =
          CreateSplatFloatConst(builder, loc, reduced_3d_type, 0.0f);
      inv_sum = builder.create<mlir::stablehlo::SelectOp>(loc, row_mask_pred_3d,
                                                          zero_3d, inv_sum);
      mlir::Value inv_sum_bcast =
          builder.create<mlir::stablehlo::BroadcastInDimOp>(
              loc, type, inv_sum, builder.getDenseI64ArrayAttr({0, 1, 2}));
      mlir::Value new_softmax =
          builder.create<mlir::stablehlo::MulOp>(loc, new_exp, inv_sum_bcast);

      sum_reduce.getResult(0).replaceAllUsesWith(new_sum);
      root_softmax_op->replaceAllUsesWith(mlir::ValueRange{new_softmax});
    }
  }
}

void EraseTriviallyDeadOps(mlir::func::FuncOp func) {
  bool changed = true;
  while (changed) {
    changed = false;
    llvm::SmallVector<mlir::Operation*> dead_ops;
    func.walk([&](mlir::Operation* op) {
      if (op != func.getOperation() && op->use_empty() &&
          mlir::isOpTriviallyDead(op)) {
        dead_ops.push_back(op);
      }
    });
    for (mlir::Operation* op : dead_ops) {
      op->erase();
      changed = true;
    }
  }
}

struct OptimizeSdpaSoftmaxPass
    : public mlir::PassWrapper<OptimizeSdpaSoftmaxPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizeSdpaSoftmaxPass)

  llvm::StringRef getArgument() const override {
    return "optimize-sdpa-softmax";
  }
  llvm::StringRef getDescription() const override {
    return "Eliminates standalone SDPA reduce-window output fusions, hoists "
           "3D reciprocals, and unblocks masked/GQA online-softmax.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    RewriteBackwardGradPRowsum(func);
    RewriteBackwardSdpaSoftmax(func);
    EraseTriviallyDeadOps(func);
    RewriteForwardSdpaSoftmax(func);
    EraseTriviallyDeadOps(func);
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createOptimizeSdpaSoftmaxPass() {
  return std::make_unique<OptimizeSdpaSoftmaxPass>();
}

}  // namespace torch_tpu
