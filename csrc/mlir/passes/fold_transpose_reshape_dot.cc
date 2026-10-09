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

#include "csrc/mlir/passes/fold_transpose_reshape_dot.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

struct SplitTransposeInfo {
  mlir::Value orig_input;
  llvm::SmallVector<int64_t, 4> non_contract_dims;
  llvm::SmallVector<int64_t, 4> non_contract_sizes;
  llvm::SmallVector<int64_t, 4> contract_dims;
  llvm::SmallVector<int64_t, 4> contract_sizes;
};

bool IsStrictlyAscending(llvm::ArrayRef<int64_t> dims) {
  for (size_t i = 1; i < dims.size(); ++i) {
    if (dims[i] <= dims[i - 1]) {
      return false;
    }
  }
  return true;
}

int64_t PermutedDimProduct(llvm::ArrayRef<int64_t> shape,
                           llvm::ArrayRef<int64_t> perm_slice) {
  int64_t prod = 1;
  for (int64_t dim : perm_slice) {
    prod *= shape[dim];
  }
  return prod;
}

llvm::SmallVector<int64_t, 4> GatherDimSizes(llvm::ArrayRef<int64_t> shape,
                                             llvm::ArrayRef<int64_t> dims) {
  llvm::SmallVector<int64_t, 4> sizes;
  sizes.reserve(dims.size());
  for (int64_t idx : dims) {
    sizes.push_back(shape[idx]);
  }
  return sizes;
}

std::optional<SplitTransposeInfo> TryBuildSplitTransposeInfo(
    mlir::Value orig_input, llvm::ArrayRef<int64_t> x_shape,
    llvm::ArrayRef<int64_t> perm, int64_t split, int64_t d0, int64_t d1) {
  llvm::ArrayRef<int64_t> nc_perm = perm.take_front(split);
  llvm::ArrayRef<int64_t> c_perm = perm.drop_front(split);
  if (PermutedDimProduct(x_shape, nc_perm) != d0 ||
      PermutedDimProduct(x_shape, c_perm) != d1) {
    return std::nullopt;
  }
  if (!IsStrictlyAscending(nc_perm) || !IsStrictlyAscending(c_perm)) {
    return std::nullopt;
  }
  // Skip trivial identity order where no cross-group transpose occurred.
  if (nc_perm.back() < c_perm.front()) {
    return std::nullopt;
  }

  SplitTransposeInfo info;
  info.orig_input = orig_input;
  info.non_contract_dims.assign(nc_perm.begin(), nc_perm.end());
  info.non_contract_sizes = GatherDimSizes(x_shape, nc_perm);
  info.contract_dims.assign(c_perm.begin(), c_perm.end());
  info.contract_sizes = GatherDimSizes(x_shape, c_perm);
  return info;
}

// Matches `reshape(transpose(x, perm), [D0, D1])` where `x` has static rank >=
// 3, `perm[0..m-1]` has product D0 and strictly ascending original indices,
// `perm[m..r-1]` has product D1 and strictly ascending original indices, and
// `perm[m - 1] > perm[m]` (so the transpose genuinely crosses non-contracting
// and contracting dimensions, e.g. `[B, H, S, D] -> [B, S, H, D] -> [B*S,
// H*D]`).
std::optional<SplitTransposeInfo> MatchTransposeReshape2D(mlir::Value val) {
  auto reshape_op = val.getDefiningOp<mlir::stablehlo::ReshapeOp>();
  if (!reshape_op) {
    return std::nullopt;
  }
  auto r_type = mlir::dyn_cast<mlir::RankedTensorType>(reshape_op.getType());
  if (!r_type || !r_type.hasStaticShape() || r_type.getRank() != 2) {
    return std::nullopt;
  }

  auto transpose_op =
      reshape_op.getOperand().getDefiningOp<mlir::stablehlo::TransposeOp>();
  if (!transpose_op) {
    return std::nullopt;
  }
  auto x_type = mlir::dyn_cast<mlir::RankedTensorType>(
      transpose_op.getOperand().getType());
  if (!x_type || !x_type.hasStaticShape() || x_type.getRank() < 3) {
    return std::nullopt;
  }

  const int64_t d0 = r_type.getDimSize(0);
  const int64_t d1 = r_type.getDimSize(1);
  if (d0 <= 0 || d1 <= 0) {
    return std::nullopt;
  }

  llvm::ArrayRef<int64_t> x_shape = x_type.getShape();
  llvm::ArrayRef<int64_t> perm = transpose_op.getPermutation();
  const int64_t rank = x_type.getRank();
  for (int64_t split = 1; split < rank; ++split) {
    if (auto info = TryBuildSplitTransposeInfo(transpose_op.getOperand(),
                                               x_shape, perm, split, d0, d1)) {
      return info;
    }
  }
  return std::nullopt;
}

bool IsValid2DDotForSplitRewrite(mlir::stablehlo::DotGeneralOp dot_op,
                                 int64_t& rhs_c_dim) {
  auto out_2d_type = mlir::dyn_cast<mlir::RankedTensorType>(dot_op.getType());
  if (!out_2d_type || !out_2d_type.hasStaticShape() ||
      out_2d_type.getRank() != 2) {
    return false;
  }

  auto dot_dims = dot_op.getDotDimensionNumbersAttr();
  if (!dot_dims.getLhsBatchingDimensions().empty() ||
      !dot_dims.getRhsBatchingDimensions().empty() ||
      dot_dims.getLhsContractingDimensions().size() != 1 ||
      dot_dims.getRhsContractingDimensions().size() != 1) {
    return false;
  }

  // Only rewrite when the contracting dimension of `lhs` is axis 1 (so the
  // multi-axis split group `[H, D]` is contracted on TPU MXU without
  // non-contracting output padding).
  if (dot_dims.getLhsContractingDimensions()[0] != 1) {
    return false;
  }
  rhs_c_dim = dot_dims.getRhsContractingDimensions()[0];

  auto rhs_type =
      mlir::dyn_cast<mlir::RankedTensorType>(dot_op.getRhs().getType());
  return rhs_type && rhs_type.hasStaticShape() && rhs_type.getRank() == 2;
}

mlir::Value BuildReshapedRhs(mlir::OpBuilder& builder, mlir::Location loc,
                             mlir::Value rhs, int64_t rhs_c_dim,
                             int64_t out_features,
                             llvm::ArrayRef<int64_t> contract_sizes,
                             llvm::SmallVectorImpl<int64_t>& new_rhs_c_dims) {
  // Unwrap `transpose(w_orig, {1, 0})` when present so `w_orig : [M, K]` is
  // reshaped along its contiguous minor dimension directly into `[M, H, D]`.
  mlir::Value rhs_val = rhs;
  int64_t effective_rhs_c_dim = rhs_c_dim;
  if (auto wt = rhs_val.getDefiningOp<mlir::stablehlo::TransposeOp>()) {
    if (wt.getPermutation() == llvm::ArrayRef<int64_t>({1, 0})) {
      rhs_val = wt.getOperand();
      effective_rhs_c_dim = 1 - rhs_c_dim;
    }
  }

  const int64_t num_c = contract_sizes.size();
  const int64_t c_offset = (effective_rhs_c_dim == 1) ? 1 : 0;
  llvm::SmallVector<int64_t, 4> new_rhs_shape;
  if (effective_rhs_c_dim == 1) {
    new_rhs_shape.push_back(out_features);
  }
  for (int64_t i = 0; i < num_c; ++i) {
    new_rhs_shape.push_back(contract_sizes[i]);
    new_rhs_c_dims.push_back(c_offset + i);
  }
  if (effective_rhs_c_dim != 1) {
    new_rhs_shape.push_back(out_features);
  }

  auto elem_type =
      mlir::cast<mlir::RankedTensorType>(rhs.getType()).getElementType();
  auto new_rhs_type = mlir::RankedTensorType::get(new_rhs_shape, elem_type);
  return mlir::stablehlo::ReshapeOp::create(builder, loc, new_rhs_type,
                                            rhs_val);
}

void TryMatchSingleUseBiasAdd(mlir::Value dot_result, mlir::Operation* user,
                              mlir::stablehlo::AddOp& bias_add,
                              mlir::stablehlo::BroadcastInDimOp& bias_bcast) {
  auto add_user = mlir::dyn_cast<mlir::stablehlo::AddOp>(user);
  if (!add_user || !dot_result.hasOneUse()) {
    return;
  }
  mlir::Value other =
      add_user.getLhs() == dot_result ? add_user.getRhs() : add_user.getLhs();
  auto bcast = other.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  if (!bcast) {
    return;
  }
  llvm::ArrayRef<int64_t> bdims = bcast.getBroadcastDimensions();
  if (bdims.empty() || bdims == llvm::ArrayRef<int64_t>({1})) {
    bias_add = add_user;
    bias_bcast = bcast;
  }
}

void RewriteBiasedDotUsers(mlir::stablehlo::DotGeneralOp dot_op,
                           mlir::stablehlo::DotGeneralOp new_dot,
                           mlir::RankedTensorType out_2d_type,
                           mlir::RankedTensorType new_out_type,
                           mlir::stablehlo::AddOp bias_add,
                           mlir::stablehlo::BroadcastInDimOp bias_bcast) {
  mlir::OpBuilder add_builder(bias_add);
  llvm::SmallVector<int64_t, 1> new_bdims;
  if (!bias_bcast.getBroadcastDimensions().empty()) {
    new_bdims.push_back(new_out_type.getRank() - 1);
  }
  mlir::Value new_bcast = mlir::stablehlo::BroadcastInDimOp::create(
      add_builder, bias_add.getLoc(), new_out_type, bias_bcast.getOperand(),
      add_builder.getDenseI64ArrayAttr(new_bdims));
  mlir::Value lhs =
      bias_add.getLhs() == dot_op.getResult() ? new_dot.getResult() : new_bcast;
  mlir::Value rhs =
      bias_add.getLhs() == dot_op.getResult() ? new_bcast : new_dot.getResult();
  mlir::Value new_add =
      mlir::stablehlo::AddOp::create(add_builder, bias_add.getLoc(), lhs, rhs)
          .getResult();

  llvm::SmallVector<mlir::Operation*, 4> add_reshapes;
  bool has_other_add_users = false;
  for (mlir::Operation* u : bias_add.getResult().getUsers()) {
    auto ru = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(u);
    if (ru && ru.getType() == new_out_type) {
      add_reshapes.push_back(ru);
    } else {
      has_other_add_users = true;
    }
  }
  for (mlir::Operation* ru : add_reshapes) {
    ru->replaceAllUsesWith(mlir::ValueRange{new_add});
    ru->erase();
  }
  if (has_other_add_users) {
    mlir::Value fallback_2d = mlir::stablehlo::ReshapeOp::create(
        add_builder, bias_add.getLoc(), out_2d_type, new_add);
    bias_add.replaceAllUsesWith(fallback_2d);
  }
  bias_add->erase();
  dot_op->erase();
}

void ReplaceUnbiasedDotUsers(mlir::OpBuilder& builder, mlir::Location loc,
                             mlir::stablehlo::DotGeneralOp dot_op,
                             mlir::stablehlo::DotGeneralOp new_dot,
                             mlir::RankedTensorType out_2d_type,
                             llvm::ArrayRef<mlir::Operation*> direct_reshapes) {
  for (mlir::Operation* ru : direct_reshapes) {
    ru->replaceAllUsesWith(mlir::ValueRange{new_dot.getResult()});
    ru->erase();
  }
  if (!dot_op.getResult().use_empty()) {
    mlir::Value fallback_2d = mlir::stablehlo::ReshapeOp::create(
        builder, loc, out_2d_type, new_dot.getResult());
    dot_op.replaceAllUsesWith(fallback_2d);
  } else {
    dot_op->erase();
  }
}

bool TryRewriteDotGeneral(mlir::stablehlo::DotGeneralOp dot_op) {
  int64_t rhs_c_dim = 0;
  if (!IsValid2DDotForSplitRewrite(dot_op, rhs_c_dim)) {
    return false;
  }

  auto split_opt = MatchTransposeReshape2D(dot_op.getLhs());
  constexpr int64_t kTpuLaneWidth = 128;
  if (!split_opt.has_value() ||
      split_opt->contract_sizes.back() != kTpuLaneWidth) {
    return false;
  }
  const SplitTransposeInfo& split = *split_opt;

  auto out_2d_type = mlir::cast<mlir::RankedTensorType>(dot_op.getType());
  auto rhs_type = mlir::cast<mlir::RankedTensorType>(dot_op.getRhs().getType());
  const int64_t out_features = rhs_type.getDimSize(1 - rhs_c_dim);

  mlir::OpBuilder builder(dot_op);
  mlir::Location loc = dot_op.getLoc();
  llvm::SmallVector<int64_t, 4> new_rhs_c_dims;
  mlir::Value new_rhs =
      BuildReshapedRhs(builder, loc, dot_op.getRhs(), rhs_c_dim, out_features,
                       split.contract_sizes, new_rhs_c_dims);

  llvm::SmallVector<int64_t, 4> new_out_shape(split.non_contract_sizes.begin(),
                                              split.non_contract_sizes.end());
  new_out_shape.push_back(out_features);
  auto new_out_type =
      mlir::RankedTensorType::get(new_out_shape, out_2d_type.getElementType());

  auto new_dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
      builder.getContext(),
      /*lhsBatchingDimensions=*/{},
      /*rhsBatchingDimensions=*/{},
      /*lhsContractingDimensions=*/split.contract_dims,
      /*rhsContractingDimensions=*/new_rhs_c_dims);

  auto new_dot = mlir::stablehlo::DotGeneralOp::create(
      builder, loc, new_out_type, split.orig_input, new_rhs, new_dot_dims,
      dot_op.getPrecisionConfigAttr(), mlir::stablehlo::DotAlgorithmAttr{});

  // Propagate ND shape `[B, S, M]` directly through any `add(bcast(bias), dot)`
  // and `reshape(..., [B, S, M])` users so no intermediate 2D `[B*S, M]`
  // collapse breaks XLA's direct 4D contraction fusion.
  llvm::SmallVector<mlir::Operation*, 4> direct_reshapes;
  mlir::stablehlo::AddOp bias_add = nullptr;
  mlir::stablehlo::BroadcastInDimOp bias_bcast = nullptr;

  for (mlir::Operation* user : dot_op.getResult().getUsers()) {
    auto res_user = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(user);
    if (res_user && res_user.getType() == new_out_type) {
      direct_reshapes.push_back(res_user);
    } else {
      TryMatchSingleUseBiasAdd(dot_op.getResult(), user, bias_add, bias_bcast);
    }
  }

  if (bias_add && bias_bcast) {
    RewriteBiasedDotUsers(dot_op, new_dot, out_2d_type, new_out_type, bias_add,
                          bias_bcast);
    return true;
  }

  ReplaceUnbiasedDotUsers(builder, loc, dot_op, new_dot, out_2d_type,
                          direct_reshapes);
  return true;
}

class FoldTransposeReshapeDotPass
    : public mlir::PassWrapper<FoldTransposeReshapeDotPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldTransposeReshapeDotPass)

  mlir::StringRef getArgument() const final {
    return "fold-transpose-reshape-dot";
  }

  mlir::StringRef getDescription() const final {
    return "Fold activation Transpose -> Reshape -> DotGeneral into direct "
           "multi-axis DotGeneral contractions.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    llvm::SmallVector<mlir::stablehlo::DotGeneralOp, 16> candidates;
    func.walk(
        [&](mlir::stablehlo::DotGeneralOp op) { candidates.push_back(op); });
    for (mlir::stablehlo::DotGeneralOp dot_op : candidates) {
      TryRewriteDotGeneral(dot_op);
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createFoldTransposeReshapeDotPass() {
  return std::make_unique<FoldTransposeReshapeDotPass>();
}

}  // namespace torch_tpu
