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

#include "csrc/mlir/passes/sink_projection_reshapes.h"

#include <cstdint>
#include <memory>
#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

constexpr int64_t kMinMultiHeadSeqLen = 128;
constexpr int64_t kMinMultiHeadDim = 16;
constexpr int64_t kMin3DProjectionSeqLen = 256;
constexpr int64_t kMin3DProjectionFeatures = 64;

bool IsDotOrDotGeneral(mlir::Operation* op) {
  return op && (mlir::isa<mlir::stablehlo::DotOp>(op) ||
                mlir::isa<mlir::stablehlo::DotGeneralOp>(op));
}

bool Is0213Permutation(llvm::ArrayRef<int64_t> perm) {
  return perm.size() == 4 && perm[0] == 0 && perm[1] == 2 && perm[2] == 1 &&
         perm[3] == 3;
}

enum class PostDotStepKind {
  kBiasAdd,
  kScalarMul,
};

struct PostDotStep {
  PostDotStepKind kind;
  // For kBiasAdd: a 1D tensor<Nxdtype> value (or reshapeable to [N]).
  // For kScalarMul: a 0D/1-element value or splat constant.
  mlir::Value operand;
  bool lhs_was_main = true;
};

// Matches a scalar broadcast or splat constant `val` of float/integer type.
bool IsScalarBroadcastOrSplat(mlir::Value val) {
  if (auto const_op = val.getDefiningOp<mlir::stablehlo::ConstantOp>()) {
    auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
    return attr && !attr.empty() && attr.isSplat();
  }
  if (auto bcast_op = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
    auto in_type =
        mlir::dyn_cast<mlir::RankedTensorType>(bcast_op.getOperand().getType());
    return in_type && in_type.hasStaticShape() && in_type.getNumElements() == 1;
  }
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(val.getType());
  return type && type.hasStaticShape() && type.getNumElements() == 1;
}

// Re-materializes a scalar broadcast or splat constant `val` at `target_type`.
mlir::Value RebuildScalarBroadcastOrSplat(mlir::OpBuilder& builder,
                                          mlir::Location loc, mlir::Value val,
                                          mlir::RankedTensorType target_type) {
  if (auto const_op = val.getDefiningOp<mlir::stablehlo::ConstantOp>()) {
    auto attr = mlir::cast<mlir::DenseElementsAttr>(const_op.getValue());
    auto splat_attr = mlir::DenseElementsAttr::get(
        target_type, attr.getSplatValue<mlir::Attribute>());
    return builder.create<mlir::stablehlo::ConstantOp>(loc, target_type,
                                                       splat_attr);
  }
  mlir::Value src = val;
  if (auto bcast_op = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
    src = bcast_op.getOperand();
  }
  auto src_type = mlir::cast<mlir::RankedTensorType>(src.getType());
  if (src_type.getRank() != 0) {
    auto scalar_type =
        mlir::RankedTensorType::get({}, target_type.getElementType());
    src = builder.create<mlir::stablehlo::ReshapeOp>(loc, scalar_type, src);
  }
  return builder.create<mlir::stablehlo::BroadcastInDimOp>(
      loc, target_type, src, builder.getDenseI64ArrayAttr({}));
}

// Checks whether `val` is a broadcast of a 1D feature bias of size `feature_n`
// along the last dimension of a tensor of rank `tensor_rank`. Returns the
// underlying 1-D bias value (or a tensor with `numElements == feature_n` whose
// non-unit dimension is mapped to `tensor_rank - 1`).
std::optional<mlir::Value> MatchLastDimFeatureBroadcast(mlir::Value val,
                                                        int64_t tensor_rank,
                                                        int64_t feature_n) {
  auto bcast_op = val.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  if (!bcast_op || tensor_rank < 2) {
    return std::nullopt;
  }
  mlir::Value src = bcast_op.getOperand();
  auto src_type = mlir::dyn_cast<mlir::RankedTensorType>(src.getType());
  if (!src_type || !src_type.hasStaticShape() ||
      src_type.getNumElements() != feature_n) {
    return std::nullopt;
  }
  llvm::ArrayRef<int64_t> dims = bcast_op.getBroadcastDimensions();
  if (static_cast<int64_t>(dims.size()) != src_type.getRank()) {
    return std::nullopt;
  }
  if (src_type.getRank() == 1) {
    if (dims[0] == tensor_rank - 1 && src_type.getDimSize(0) == feature_n) {
      return src;
    }
    return std::nullopt;
  }
  // Allow [1, ..., 1, feature_n] broadcast where only the last dim has size
  // `feature_n` and maps to `tensor_rank - 1`.
  for (int64_t i = 0; i < src_type.getRank() - 1; ++i) {
    if (src_type.getDimSize(i) != 1) {
      return std::nullopt;
    }
  }
  if (src_type.getDimSize(src_type.getRank() - 1) == feature_n &&
      dims.back() == tensor_rank - 1) {
    return src;
  }
  return std::nullopt;
}

struct MatchedPostDotChain {
  mlir::Operation* dot_op = nullptr;
  mlir::RankedTensorType dot_type;
  llvm::SmallVector<PostDotStep, 4> steps;
};

// Walks backwards from `start_val` (the input to an outer ReshapeOp) through
// single-use intermediate ReshapeOps, 1D-feature-broadcast AddOps, and
// scalar-broadcast MulOps to find a single-use 2D+ DotOp or DotGeneralOp whose
// last dimension equals `expected_feature_n`.
std::optional<MatchedPostDotChain> MatchPostDotChain(
    mlir::Value start_val, int64_t expected_feature_n) {
  llvm::SmallVector<PostDotStep, 4> reverse_steps;
  mlir::Value curr = start_val;

  while (curr) {
    mlir::Operation* def_op = curr.getDefiningOp();
    if (!def_op || !def_op->hasOneUse()) {
      return std::nullopt;
    }
    auto curr_type = mlir::dyn_cast<mlir::RankedTensorType>(curr.getType());
    if (!curr_type || !curr_type.hasStaticShape() || curr_type.getRank() < 2 ||
        curr_type.getShape().back() != expected_feature_n) {
      return std::nullopt;
    }

    if (IsDotOrDotGeneral(def_op)) {
      if (reverse_steps.empty()) {
        return std::nullopt;
      }
      MatchedPostDotChain result;
      result.dot_op = def_op;
      result.dot_type = curr_type;
      result.steps.assign(reverse_steps.rbegin(), reverse_steps.rend());
      return result;
    }

    if (auto reshape_op = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(def_op)) {
      curr = reshape_op.getOperand();
      continue;
    }

    if (auto add_op = mlir::dyn_cast<mlir::stablehlo::AddOp>(def_op)) {
      const int64_t rank = curr_type.getRank();
      if (auto bias = MatchLastDimFeatureBroadcast(add_op.getRhs(), rank,
                                                   expected_feature_n)) {
        reverse_steps.push_back(PostDotStep{PostDotStepKind::kBiasAdd, *bias,
                                            /*lhs_was_main=*/true});
        curr = add_op.getLhs();
        continue;
      }
      if (auto bias = MatchLastDimFeatureBroadcast(add_op.getLhs(), rank,
                                                   expected_feature_n)) {
        reverse_steps.push_back(PostDotStep{PostDotStepKind::kBiasAdd, *bias,
                                            /*lhs_was_main=*/false});
        curr = add_op.getRhs();
        continue;
      }
      return std::nullopt;
    }

    if (auto mul_op = mlir::dyn_cast<mlir::stablehlo::MulOp>(def_op)) {
      if (IsScalarBroadcastOrSplat(mul_op.getRhs())) {
        reverse_steps.push_back(PostDotStep{PostDotStepKind::kScalarMul,
                                            mul_op.getRhs(),
                                            /*lhs_was_main=*/true});
        curr = mul_op.getLhs();
        continue;
      }
      if (IsScalarBroadcastOrSplat(mul_op.getLhs())) {
        reverse_steps.push_back(PostDotStep{PostDotStepKind::kScalarMul,
                                            mul_op.getLhs(),
                                            /*lhs_was_main=*/false});
        curr = mul_op.getRhs();
        continue;
      }
      return std::nullopt;
    }

    return std::nullopt;
  }

  return std::nullopt;
}

// Checks whether `dot_op` is a 2D matrix multiplication whose LHS operand is a
// ReshapeOp from a 3D+ tensor and has NO backward `grad_weight` user
// contracting along dimension 0 (the flattened batch*seq dimension).
bool IsForwardFlattened2DDotWithoutGradWeightUser(mlir::Operation* dot_op,
                                                  int64_t expected_b,
                                                  int64_t expected_s) {
  if (dot_op->getNumOperands() < 2) {
    return false;
  }
  mlir::Value lhs = dot_op->getOperand(0);
  mlir::Value rhs = dot_op->getOperand(1);
  auto lhs_type = mlir::dyn_cast<mlir::RankedTensorType>(lhs.getType());
  auto rhs_type = mlir::dyn_cast<mlir::RankedTensorType>(rhs.getType());
  if (!lhs_type || !rhs_type || !lhs_type.hasStaticShape() ||
      !rhs_type.hasStaticShape() || lhs_type.getRank() != 2 ||
      rhs_type.getRank() != 2) {
    return false;
  }
  if (auto dg = mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(dot_op)) {
    auto dnums = dg.getDotDimensionNumbers();
    if (!dnums.getLhsBatchingDimensions().empty() ||
        !dnums.getRhsBatchingDimensions().empty() ||
        dnums.getLhsContractingDimensions().size() != 1 ||
        dnums.getLhsContractingDimensions()[0] != 1 ||
        dnums.getRhsContractingDimensions().size() != 1) {
      return false;
    }
  }

  auto lhs_reshape = lhs.getDefiningOp<mlir::stablehlo::ReshapeOp>();
  if (!lhs_reshape) {
    return false;
  }
  auto orig_lhs_type = mlir::dyn_cast<mlir::RankedTensorType>(
      lhs_reshape.getOperand().getType());
  if (!orig_lhs_type || !orig_lhs_type.hasStaticShape() ||
      orig_lhs_type.getRank() < 3 ||
      orig_lhs_type.getDimSize(0) != expected_b ||
      orig_lhs_type.getDimSize(1) != expected_s) {
    return false;
  }

  // Verify that no user of `lhs` is a backward `grad_weight` contraction
  // contracting along dimension 0 (the flattened B*S dimension).
  for (mlir::Operation* user : lhs.getUsers()) {
    if (auto dg = mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(user)) {
      auto dnums = dg.getDotDimensionNumbers();
      if (dg.getLhs() == lhs) {
        for (int64_t c : dnums.getLhsContractingDimensions()) {
          if (c == 0) {
            return false;
          }
        }
      }
      if (dg.getRhs() == lhs) {
        for (int64_t c : dnums.getRhsContractingDimensions()) {
          if (c == 0) {
            return false;
          }
        }
      }
    } else if (auto d = mlir::dyn_cast<mlir::stablehlo::DotOp>(user)) {
      if (d.getRhs() == lhs) {
        return false;
      }
    }
  }

  return true;
}

mlir::Value ReplayPostDotSteps4D(mlir::OpBuilder& builder, mlir::Location loc,
                                 mlir::Value acc,
                                 llvm::ArrayRef<PostDotStep> steps,
                                 int64_t num_heads, int64_t head_dim,
                                 llvm::ArrayRef<int64_t> bias_bcast_dims) {
  auto acc_type = mlir::cast<mlir::RankedTensorType>(acc.getType());
  mlir::Type elem_type = acc_type.getElementType();
  auto bias_2d_type =
      mlir::RankedTensorType::get({num_heads, head_dim}, elem_type);

  for (const PostDotStep& step : steps) {
    if (step.kind == PostDotStepKind::kBiasAdd) {
      mlir::Value bias_2d = builder.create<mlir::stablehlo::ReshapeOp>(
          loc, bias_2d_type, step.operand);
      mlir::Value bias_bcast =
          builder.create<mlir::stablehlo::BroadcastInDimOp>(
              loc, acc_type, bias_2d,
              builder.getDenseI64ArrayAttr(bias_bcast_dims));
      acc = step.lhs_was_main
                ? builder.create<mlir::stablehlo::AddOp>(loc, acc, bias_bcast)
                : builder.create<mlir::stablehlo::AddOp>(loc, bias_bcast, acc);
    } else {
      mlir::Value scale_bcast =
          RebuildScalarBroadcastOrSplat(builder, loc, step.operand, acc_type);
      acc = step.lhs_was_main
                ? builder.create<mlir::stablehlo::MulOp>(loc, acc, scale_bcast)
                : builder.create<mlir::stablehlo::MulOp>(loc, scale_bcast, acc);
    }
  }
  return acc;
}

mlir::Value ReplayPostDotSteps3D(mlir::OpBuilder& builder, mlir::Location loc,
                                 mlir::Value acc,
                                 llvm::ArrayRef<PostDotStep> steps,
                                 int64_t feature_n) {
  auto acc_type = mlir::cast<mlir::RankedTensorType>(acc.getType());
  mlir::Type elem_type = acc_type.getElementType();
  auto bias_1d_type = mlir::RankedTensorType::get({feature_n}, elem_type);

  for (const PostDotStep& step : steps) {
    if (step.kind == PostDotStepKind::kBiasAdd) {
      mlir::Value bias_1d = step.operand;
      if (mlir::cast<mlir::RankedTensorType>(bias_1d.getType()).getRank() !=
          1) {
        bias_1d = builder.create<mlir::stablehlo::ReshapeOp>(loc, bias_1d_type,
                                                             bias_1d);
      }
      mlir::Value bias_bcast =
          builder.create<mlir::stablehlo::BroadcastInDimOp>(
              loc, acc_type, bias_1d, builder.getDenseI64ArrayAttr({2}));
      acc = step.lhs_was_main
                ? builder.create<mlir::stablehlo::AddOp>(loc, acc, bias_bcast)
                : builder.create<mlir::stablehlo::AddOp>(loc, bias_bcast, acc);
    } else {
      mlir::Value scale_bcast =
          RebuildScalarBroadcastOrSplat(builder, loc, step.operand, acc_type);
      acc = step.lhs_was_main
                ? builder.create<mlir::stablehlo::MulOp>(loc, acc, scale_bcast)
                : builder.create<mlir::stablehlo::MulOp>(loc, scale_bcast, acc);
    }
  }
  return acc;
}

// Case 1: Sinks 4D multi-head attention Q/K/V reshapes `[B, S, H, D]` (and an
// immediately following `[0, 2, 1, 3]` transpose to `[B, H, S, D]` when
// present) upward past bias additions and scalar multiplications onto the
// producing `dot` / `dot_general`.
void TrySink4DMultiHeadReshape(mlir::stablehlo::ReshapeOp reshape_op) {
  auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(reshape_op.getType());
  if (!out_type || !out_type.hasStaticShape() || out_type.getRank() != 4) {
    return;
  }
  const int64_t b = out_type.getDimSize(0);
  const int64_t s = out_type.getDimSize(1);
  const int64_t h = out_type.getDimSize(2);
  const int64_t d = out_type.getDimSize(3);
  if (b < 1 || s < kMinMultiHeadSeqLen || h <= 1 || d < kMinMultiHeadDim) {
    return;
  }
  const int64_t feature_n = h * d;

  auto matched = MatchPostDotChain(reshape_op.getOperand(), feature_n);
  if (!matched) {
    return;
  }
  if (!IsForwardFlattened2DDotWithoutGradWeightUser(matched->dot_op, b, s)) {
    return;
  }

  mlir::stablehlo::TransposeOp trans_user;
  if (reshape_op->hasOneUse()) {
    if (auto t = mlir::dyn_cast<mlir::stablehlo::TransposeOp>(
            *reshape_op->getUsers().begin())) {
      if (Is0213Permutation(t.getPermutation())) {
        trans_user = t;
      }
    }
  }

  if (trans_user) {
    mlir::OpBuilder builder(trans_user);
    mlir::Location loc = trans_user.getLoc();
    mlir::Value dot_4d = builder.create<mlir::stablehlo::ReshapeOp>(
        loc, out_type, matched->dot_op->getResult(0));
    auto trans_type = mlir::cast<mlir::RankedTensorType>(trans_user.getType());
    mlir::Value trans_dot = builder.create<mlir::stablehlo::TransposeOp>(
        loc, trans_type, dot_4d, builder.getDenseI64ArrayAttr({0, 2, 1, 3}));
    mlir::Value replaced =
        ReplayPostDotSteps4D(builder, loc, trans_dot, matched->steps, h, d,
                             /*bias_bcast_dims=*/{1, 3});
    trans_user.replaceAllUsesWith(replaced);
  } else {
    mlir::OpBuilder builder(reshape_op);
    mlir::Location loc = reshape_op.getLoc();
    mlir::Value dot_4d = builder.create<mlir::stablehlo::ReshapeOp>(
        loc, out_type, matched->dot_op->getResult(0));
    mlir::Value replaced = ReplayPostDotSteps4D(
        builder, loc, dot_4d, matched->steps, h, d, /*bias_bcast_dims=*/{2, 3});
    reshape_op.replaceAllUsesWith(replaced);
  }
}

// Case 2: Sinks long-sequence (`S >= 256`) forward 3D projection reshapes
// `[B*S, N] -> [B, S, N]` upward past bias additions and scalar multiplications
// onto the producing 2D `dot` / `dot_general`.
void TrySink3DProjectionReshape(mlir::stablehlo::ReshapeOp reshape_op) {
  auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(reshape_op.getType());
  if (!out_type || !out_type.hasStaticShape() || out_type.getRank() != 3) {
    return;
  }
  // Skip if this 3D reshape is immediately consumed only by another reshape.
  if (reshape_op->hasOneUse() &&
      mlir::isa<mlir::stablehlo::ReshapeOp>(*reshape_op->getUsers().begin())) {
    return;
  }
  const int64_t b = out_type.getDimSize(0);
  const int64_t s = out_type.getDimSize(1);
  const int64_t feature_n = out_type.getDimSize(2);
  if (b < 1 || s < kMin3DProjectionSeqLen ||
      feature_n < kMin3DProjectionFeatures) {
    return;
  }

  auto matched = MatchPostDotChain(reshape_op.getOperand(), feature_n);
  if (!matched || matched->dot_type.getRank() != 2) {
    return;
  }
  if (!IsForwardFlattened2DDotWithoutGradWeightUser(matched->dot_op, b, s)) {
    return;
  }

  mlir::OpBuilder builder(reshape_op);
  mlir::Location loc = reshape_op.getLoc();
  mlir::Value dot_3d = builder.create<mlir::stablehlo::ReshapeOp>(
      loc, out_type, matched->dot_op->getResult(0));
  mlir::Value replaced =
      ReplayPostDotSteps3D(builder, loc, dot_3d, matched->steps, feature_n);
  reshape_op.replaceAllUsesWith(replaced);
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

struct SinkProjectionReshapesPass
    : public mlir::PassWrapper<SinkProjectionReshapesPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SinkProjectionReshapesPass)

  llvm::StringRef getArgument() const override {
    return "sink-projection-reshapes";
  }
  llvm::StringRef getDescription() const override {
    return "Sinks multi-head attention Q/K/V 4D reshape-transposes and "
           "long-sequence forward 3D projection reshapes past bias and scale "
           "onto stablehlo dot contractions.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();

    llvm::SmallVector<mlir::stablehlo::ReshapeOp> reshapes_4d;
    func.walk(
        [&](mlir::stablehlo::ReshapeOp op) { reshapes_4d.push_back(op); });
    for (mlir::stablehlo::ReshapeOp op : reshapes_4d) {
      if (op->getParentOp() != nullptr) {
        TrySink4DMultiHeadReshape(op);
      }
    }
    EraseTriviallyDeadOps(func);

    llvm::SmallVector<mlir::stablehlo::ReshapeOp> reshapes_3d;
    func.walk(
        [&](mlir::stablehlo::ReshapeOp op) { reshapes_3d.push_back(op); });
    for (mlir::stablehlo::ReshapeOp op : reshapes_3d) {
      if (op->getParentOp() != nullptr) {
        TrySink3DProjectionReshape(op);
      }
    }
    EraseTriviallyDeadOps(func);
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createSinkProjectionReshapesPass() {
  return std::make_unique<SinkProjectionReshapesPass>();
}

}  // namespace torch_tpu
