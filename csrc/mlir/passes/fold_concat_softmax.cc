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

#include "csrc/mlir/passes/fold_concat_softmax.h"

#include <cstdint>
#include <memory>

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
#include "mlir/IR/TypeRange.h"
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

// Maximum tail dimension size to partition; attention sinks typically
// append a small number of sink tokens (e.g. <= 16).
constexpr int64_t kMaxSmallTailConcatDimSize = 16;

bool IsIdentityBroadcastDims(llvm::ArrayRef<int64_t> dims, int64_t rank) {
  if (static_cast<int64_t>(dims.size()) != rank) {
    return false;
  }
  for (int64_t i = 0; i < rank; ++i) {
    if (dims[i] != i) {
      return false;
    }
  }
  return true;
}

bool IsDropDimBroadcastDims(llvm::ArrayRef<int64_t> dims, int64_t rank,
                            int64_t dropped_dim) {
  if (static_cast<int64_t>(dims.size()) != rank - 1) {
    return false;
  }
  int64_t idx = 0;
  for (int64_t i = 0; i < rank; ++i) {
    if (i == dropped_dim) {
      continue;
    }
    if (dims[idx++] != i) {
      return false;
    }
  }
  return true;
}

mlir::Value StripOptionalConvert(mlir::Value val) {
  if (auto conv = val.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    return conv.getOperand();
  }
  return val;
}

mlir::Value StripOptionalFloatUpcast(mlir::Value val,
                                     mlir::Type outer_elem_type) {
  auto upcast = val.getDefiningOp<mlir::stablehlo::ConvertOp>();
  if (!upcast) {
    return val;
  }
  auto src_type =
      mlir::dyn_cast<mlir::RankedTensorType>(upcast.getOperand().getType());
  if (src_type && src_type.hasStaticShape() &&
      mlir::isa<mlir::FloatType>(src_type.getElementType()) &&
      src_type.getElementType().getIntOrFloatBitWidth() <=
          outer_elem_type.getIntOrFloatBitWidth()) {
    return upcast.getOperand();
  }
  return val;
}

bool IsBinaryReduceBody(mlir::Region& region, bool is_max) {
  if (region.empty()) {
    return false;
  }
  mlir::Block& block = region.front();
  if (block.getNumArguments() != 2) {
    return false;
  }
  auto ret_op =
      mlir::dyn_cast<mlir::stablehlo::ReturnOp>(block.getTerminator());
  if (!ret_op || ret_op.getOperands().size() != 1) {
    return false;
  }
  mlir::Operation* body_op = ret_op.getOperands()[0].getDefiningOp();
  if (!body_op || body_op->getNumOperands() != 2) {
    return false;
  }
  const bool op_matches = is_max ? mlir::isa<mlir::stablehlo::MaxOp>(body_op)
                                 : mlir::isa<mlir::stablehlo::AddOp>(body_op);
  if (!op_matches) {
    return false;
  }
  mlir::Value arg0 = block.getArgument(0);
  mlir::Value arg1 = block.getArgument(1);
  mlir::Value lhs = body_op->getOperand(0);
  mlir::Value rhs = body_op->getOperand(1);
  return (lhs == arg0 && rhs == arg1) || (lhs == arg1 && rhs == arg0);
}

bool MatchSingleDimReduceOp(mlir::Value val, int64_t dim,
                            mlir::Value& reduce_input, bool is_max,
                            mlir::stablehlo::ReduceOp* out_reduce_op) {
  auto reduce_op = val.getDefiningOp<mlir::stablehlo::ReduceOp>();
  if (!reduce_op || reduce_op.getNumResults() != 1 ||
      reduce_op.getInputs().size() != 1 ||
      reduce_op.getInitValues().size() != 1) {
    return false;
  }
  llvm::ArrayRef<int64_t> dims = reduce_op.getDimensions();
  if (dims.size() != 1 || dims[0] != dim ||
      !IsBinaryReduceBody(reduce_op.getBody(), is_max)) {
    return false;
  }
  reduce_input = reduce_op.getInputs()[0];
  if (out_reduce_op != nullptr) {
    *out_reduce_op = reduce_op;
  }
  return true;
}

mlir::Value UnwrapIdentityBroadcastOperand(mlir::Value cur, int64_t rank,
                                           int64_t dim) {
  auto cur_type = mlir::dyn_cast<mlir::RankedTensorType>(cur.getType());
  if (!cur_type || cur_type.getRank() != rank ||
      cur_type.getDimSize(dim) != 1) {
    return nullptr;
  }
  cur = StripOptionalConvert(cur);
  auto reshape = cur.getDefiningOp<mlir::stablehlo::ReshapeOp>();
  if (!reshape) {
    return nullptr;
  }
  cur = reshape.getOperand();
  auto pre_reshape_type = mlir::dyn_cast<mlir::RankedTensorType>(cur.getType());
  if (!pre_reshape_type || pre_reshape_type.getRank() != rank - 1) {
    return nullptr;
  }
  return StripOptionalConvert(cur);
}

bool MatchBroadcastedSingleDimReduce(
    mlir::Value val, int64_t dim, mlir::Value& reduce_input, bool is_max,
    mlir::stablehlo::ReduceOp* out_reduce_op = nullptr) {
  auto val_type = mlir::dyn_cast<mlir::RankedTensorType>(val.getType());
  if (!val_type || dim < 0 || dim >= val_type.getRank()) {
    return false;
  }
  const int64_t rank = val_type.getRank();
  auto bcast = StripOptionalConvert(val)
                   .getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  if (!bcast) {
    return false;
  }

  mlir::Value cur = bcast.getOperand();
  llvm::ArrayRef<int64_t> bcast_dims = bcast.getBroadcastDimensions();
  if (IsDropDimBroadcastDims(bcast_dims, rank, dim)) {
    return MatchSingleDimReduceOp(StripOptionalConvert(cur), dim, reduce_input,
                                  is_max, out_reduce_op);
  }
  if (IsIdentityBroadcastDims(bcast_dims, rank)) {
    mlir::Value unwrapped = UnwrapIdentityBroadcastOperand(cur, rank, dim);
    return unwrapped && MatchSingleDimReduceOp(unwrapped, dim, reduce_input,
                                               is_max, out_reduce_op);
  }
  return false;
}

void EraseTriviallyDeadOps(mlir::func::FuncOp func) {
  bool changed = true;
  while (changed) {
    changed = false;
    llvm::SmallVector<mlir::Operation*> dead_ops;
    func.walk([&](mlir::Operation* op) {  // REF_CAPTURE_OK=called synchronously
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

mlir::Value ConvertIfNeeded(mlir::OpBuilder& builder, mlir::Location loc,
                            mlir::Value val, mlir::RankedTensorType dst_type) {
  if (val.getType() == dst_type) {
    return val;
  }
  return builder.create<mlir::stablehlo::ConvertOp>(loc, dst_type, val);
}

void BuildReducedShapeAndAxes(mlir::RankedTensorType type, int64_t dim,
                              llvm::SmallVectorImpl<int64_t>& reduced_shape,
                              llvm::SmallVectorImpl<int64_t>& non_dim_axes) {
  const int64_t rank = type.getRank();
  reduced_shape.reserve(rank - 1);
  non_dim_axes.reserve(rank - 1);
  for (int64_t i = 0; i < rank; ++i) {
    if (i != dim) {
      reduced_shape.push_back(type.getDimSize(i));
      non_dim_axes.push_back(i);
    }
  }
}

template <typename BinaryOp>
mlir::Value CreateSingleDimReduce(mlir::OpBuilder& builder, mlir::Location loc,
                                  mlir::RankedTensorType reduced_type,
                                  mlir::Value input, mlir::Value init,
                                  int64_t dim) {
  auto reduce_op = builder.create<mlir::stablehlo::ReduceOp>(
      loc, mlir::TypeRange{reduced_type}, mlir::ValueRange{input},
      mlir::ValueRange{init}, builder.getDenseI64ArrayAttr({dim}));
  mlir::stablehlo::buildReduceBody<BinaryOp>(reduced_type.getElementType(),
                                             reduce_op.getBody(), builder);
  return reduce_op.getResult(0);
}

template <typename BinaryOp>
mlir::Value CreateSingleDimReduceOrReshape(mlir::OpBuilder& builder,
                                           mlir::Location loc,
                                           mlir::RankedTensorType reduced_type,
                                           mlir::Value input,
                                           mlir::RankedTensorType input_type,
                                           mlir::Value init, int64_t dim) {
  if (input_type.getDimSize(dim) == 1) {
    return builder.create<mlir::stablehlo::ReshapeOp>(loc, reduced_type, input);
  }
  return CreateSingleDimReduce<BinaryOp>(builder, loc, reduced_type, input,
                                         init, dim);
}

bool TryFoldShiftInvariantPreSubtractionAlongDim(
    mlir::stablehlo::SubtractOp shifted_outer,
    mlir::RankedTensorType outer_type, int64_t dim) {
  mlir::Value x = shifted_outer.getLhs();
  mlir::Value reduce2_in;
  mlir::stablehlo::ReduceOp reduce2_op;
  if (!MatchBroadcastedSingleDimReduce(shifted_outer.getRhs(), dim, reduce2_in,
                                       /*is_max=*/true, &reduce2_op) ||
      reduce2_in != x) {
    return false;
  }

  mlir::Type outer_elem_type = outer_type.getElementType();
  mlir::Value inner_val = StripOptionalFloatUpcast(x, outer_elem_type);
  auto inner_sub = inner_val.getDefiningOp<mlir::stablehlo::SubtractOp>();
  if (!inner_sub) {
    return false;
  }
  mlir::Value base = inner_sub.getLhs();
  auto base_type = mlir::dyn_cast<mlir::RankedTensorType>(base.getType());
  if (!base_type || !mlir::isa<mlir::FloatType>(base_type.getElementType())) {
    return false;
  }

  mlir::Value reduce1_in;
  if (!MatchBroadcastedSingleDimReduce(inner_sub.getRhs(), dim, reduce1_in,
                                       /*is_max=*/true) ||
      reduce1_in != base) {
    return false;
  }

  mlir::OpBuilder builder(shifted_outer);
  mlir::Location loc = shifted_outer.getLoc();
  mlir::Value base_in_outer = ConvertIfNeeded(builder, loc, base, outer_type);

  llvm::SmallVector<int64_t> reduced_shape;
  llvm::SmallVector<int64_t> non_dim_axes;
  BuildReducedShapeAndAxes(outer_type, dim, reduced_shape, non_dim_axes);
  auto reduced_type =
      mlir::RankedTensorType::get(reduced_shape, outer_elem_type);
  mlir::Value max_init =
      ConvertIfNeeded(builder, loc, reduce2_op.getInitValues()[0],
                      mlir::RankedTensorType::get({}, outer_elem_type));

  mlir::Value new_max = CreateSingleDimReduce<mlir::stablehlo::MaxOp>(
      builder, loc, reduced_type, base_in_outer, max_init, dim);
  mlir::Value new_bcast_max = builder.create<mlir::stablehlo::BroadcastInDimOp>(
      loc, outer_type, new_max, builder.getDenseI64ArrayAttr(non_dim_axes));
  mlir::Value new_shifted = builder.create<mlir::stablehlo::SubtractOp>(
      loc, outer_type, base_in_outer, new_bcast_max);
  shifted_outer.replaceAllUsesWith(new_shifted);
  return true;
}

// Eliminates redundant shift-invariant pre-subtractions before softmax.
// When an attention pattern computes `x - max(x)` before calling softmax,
// the pre-subtraction is mathematically redundant with softmax's own max
// reduction (`softmax(x - c) == softmax(x)`), so we fold the shift into
// the downstream softmax and drop the redundant reduction.
void FoldShiftInvariantPreSubtraction(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::stablehlo::SubtractOp> sub_ops;
  func.walk([&]  // REF_CAPTURE_OK=called synchronously
            (mlir::stablehlo::SubtractOp op) { sub_ops.push_back(op); });

  for (mlir::stablehlo::SubtractOp shifted_outer : sub_ops) {
    auto outer_type =
        mlir::dyn_cast<mlir::RankedTensorType>(shifted_outer.getType());
    if (!outer_type || !outer_type.hasStaticShape() ||
        !mlir::isa<mlir::FloatType>(outer_type.getElementType())) {
      continue;
    }
    for (int64_t dim = 0; dim < outer_type.getRank(); ++dim) {
      if (TryFoldShiftInvariantPreSubtractionAlongDim(shifted_outer, outer_type,
                                                      dim)) {
        break;
      }
    }
  }
  EraseTriviallyDeadOps(func);
}

bool IsFullUnitStrideSlice(llvm::ArrayRef<int64_t> start,
                           llvm::ArrayRef<int64_t> limit,
                           llvm::ArrayRef<int64_t> strides,
                           mlir::RankedTensorType part_type) {
  for (int64_t i = 0; i < part_type.getRank(); ++i) {
    if (start[i] != 0 || limit[i] != part_type.getDimSize(i) ||
        strides[i] != 1) {
      return false;
    }
  }
  return true;
}

void ReplaceSliceWithPart(mlir::stablehlo::SliceOp slice, mlir::Value part_val,
                          llvm::ArrayRef<int64_t> start,
                          llvm::ArrayRef<int64_t> limit) {
  auto part_type = mlir::cast<mlir::RankedTensorType>(part_val.getType());
  if (IsFullUnitStrideSlice(start, limit, slice.getStrides(), part_type)) {
    slice.replaceAllUsesWith(part_val);
    return;
  }
  mlir::OpBuilder builder(slice);
  auto new_slice = builder.create<mlir::stablehlo::SliceOp>(
      slice.getLoc(), part_val, builder.getDenseI64ArrayAttr(start),
      builder.getDenseI64ArrayAttr(limit), slice.getStridesAttr());
  slice.replaceAllUsesWith(new_slice.getResult());
}

void SimplifyConcatSliceUsers(mlir::stablehlo::ConcatenateOp concat_op,
                              mlir::Value lhs_val, mlir::Value rhs_val,
                              int64_t dim) {
  auto lhs_type = mlir::cast<mlir::RankedTensorType>(lhs_val.getType());
  const int64_t lhs_dim_size = lhs_type.getDimSize(dim);

  llvm::SmallVector<mlir::Operation*> users(concat_op->user_begin(),
                                            concat_op->user_end());
  for (mlir::Operation* user : users) {
    auto slice = mlir::dyn_cast<mlir::stablehlo::SliceOp>(user);
    if (!slice) {
      continue;
    }
    llvm::ArrayRef<int64_t> start = slice.getStartIndices();
    llvm::ArrayRef<int64_t> limit = slice.getLimitIndices();
    if (limit[dim] <= lhs_dim_size) {
      ReplaceSliceWithPart(slice, lhs_val, start, limit);
    } else if (start[dim] >= lhs_dim_size) {
      llvm::SmallVector<int64_t> new_start(start.begin(), start.end());
      llvm::SmallVector<int64_t> new_limit(limit.begin(), limit.end());
      new_start[dim] -= lhs_dim_size;
      new_limit[dim] -= lhs_dim_size;
      ReplaceSliceWithPart(slice, rhs_val, new_start, new_limit);
    }
  }
}

bool MatchSmallTailConcatInput(mlir::Value raw_in, int64_t dim,
                               mlir::Value& lhs, mlir::Value& rhs,
                               mlir::RankedTensorType& lhs_type,
                               mlir::RankedTensorType& rhs_type) {
  auto concat_op = raw_in.getDefiningOp<mlir::stablehlo::ConcatenateOp>();
  if (!concat_op || concat_op.getInputs().size() != 2 ||
      static_cast<int64_t>(concat_op.getDimension()) != dim) {
    return false;
  }
  lhs = concat_op.getInputs()[0];
  rhs = concat_op.getInputs()[1];
  lhs_type = mlir::dyn_cast<mlir::RankedTensorType>(lhs.getType());
  rhs_type = mlir::dyn_cast<mlir::RankedTensorType>(rhs.getType());
  if (!lhs_type || !rhs_type || !lhs_type.hasStaticShape() ||
      !rhs_type.hasStaticShape()) {
    return false;
  }
  return lhs_type.getDimSize(dim) >= 1 && rhs_type.getDimSize(dim) >= 1 &&
         rhs_type.getDimSize(dim) <= kMaxSmallTailConcatDimSize;
}

mlir::stablehlo::ConvertOp GetSingleRankedConvertUser(mlir::Operation* op) {
  if (!op->hasOneUse()) {
    return nullptr;
  }
  auto convert_out =
      mlir::dyn_cast<mlir::stablehlo::ConvertOp>(*op->user_begin());
  if (convert_out && mlir::isa<mlir::RankedTensorType>(convert_out.getType())) {
    return convert_out;
  }
  return nullptr;
}

void EmitPartitionedConcatSoftmax(mlir::stablehlo::DivOp div_op,
                                  mlir::Type comp_elem_type, int64_t dim,
                                  mlir::Value lhs, mlir::Value rhs,
                                  mlir::RankedTensorType lhs_type,
                                  mlir::RankedTensorType rhs_type,
                                  mlir::stablehlo::ReduceOp max_reduce_op,
                                  mlir::stablehlo::ReduceOp sum_reduce_op) {
  mlir::stablehlo::ConvertOp convert_out =
      GetSingleRankedConvertUser(div_op.getOperation());
  mlir::Operation* target_op =
      convert_out ? convert_out.getOperation() : div_op.getOperation();
  mlir::OpBuilder builder(target_op);
  mlir::Location loc = target_op->getLoc();

  auto lhs_comp_type =
      mlir::RankedTensorType::get(lhs_type.getShape(), comp_elem_type);
  auto rhs_comp_type =
      mlir::RankedTensorType::get(rhs_type.getShape(), comp_elem_type);
  mlir::Value lhs_comp = ConvertIfNeeded(builder, loc, lhs, lhs_comp_type);
  mlir::Value rhs_comp = ConvertIfNeeded(builder, loc, rhs, rhs_comp_type);

  llvm::SmallVector<int64_t> reduced_shape;
  llvm::SmallVector<int64_t> non_dim_axes;
  BuildReducedShapeAndAxes(lhs_type, dim, reduced_shape, non_dim_axes);
  auto reduced_type =
      mlir::RankedTensorType::get(reduced_shape, comp_elem_type);
  auto scalar_comp_type = mlir::RankedTensorType::get({}, comp_elem_type);

  mlir::Value max_init = ConvertIfNeeded(
      builder, loc, max_reduce_op.getInitValues()[0], scalar_comp_type);
  mlir::Value sum_init = ConvertIfNeeded(
      builder, loc, sum_reduce_op.getInitValues()[0], scalar_comp_type);

  mlir::Value max_lhs = CreateSingleDimReduce<mlir::stablehlo::MaxOp>(
      builder, loc, reduced_type, lhs_comp, max_init, dim);
  mlir::Value max_rhs = CreateSingleDimReduceOrReshape<mlir::stablehlo::MaxOp>(
      builder, loc, reduced_type, rhs_comp, rhs_type, max_init, dim);
  mlir::Value max_total =
      builder.create<mlir::stablehlo::MaxOp>(loc, max_lhs, max_rhs);
  mlir::DenseI64ArrayAttr non_dim_attr =
      builder.getDenseI64ArrayAttr(non_dim_axes);

  mlir::Value bcast_max_lhs = builder.create<mlir::stablehlo::BroadcastInDimOp>(
      loc, lhs_comp_type, max_total, non_dim_attr);
  mlir::Value bcast_max_rhs = builder.create<mlir::stablehlo::BroadcastInDimOp>(
      loc, rhs_comp_type, max_total, non_dim_attr);

  mlir::Value exp_lhs = builder.create<mlir::stablehlo::ExpOp>(
      loc, builder.create<mlir::stablehlo::SubtractOp>(loc, lhs_comp,
                                                       bcast_max_lhs));
  mlir::Value exp_rhs = builder.create<mlir::stablehlo::ExpOp>(
      loc, builder.create<mlir::stablehlo::SubtractOp>(loc, rhs_comp,
                                                       bcast_max_rhs));

  mlir::Value sum_lhs = CreateSingleDimReduce<mlir::stablehlo::AddOp>(
      builder, loc, reduced_type, exp_lhs, sum_init, dim);
  mlir::Value sum_rhs = CreateSingleDimReduceOrReshape<mlir::stablehlo::AddOp>(
      builder, loc, reduced_type, exp_rhs, rhs_type, sum_init, dim);
  mlir::Value sum_total =
      builder.create<mlir::stablehlo::AddOp>(loc, sum_lhs, sum_rhs);

  mlir::Value bcast_sum_lhs = builder.create<mlir::stablehlo::BroadcastInDimOp>(
      loc, lhs_comp_type, sum_total, non_dim_attr);
  mlir::Value bcast_sum_rhs = builder.create<mlir::stablehlo::BroadcastInDimOp>(
      loc, rhs_comp_type, sum_total, non_dim_attr);

  mlir::Value final_lhs =
      builder.create<mlir::stablehlo::DivOp>(loc, exp_lhs, bcast_sum_lhs);
  mlir::Value final_rhs =
      builder.create<mlir::stablehlo::DivOp>(loc, exp_rhs, bcast_sum_rhs);
  if (convert_out) {
    mlir::Type res_elem_type =
        mlir::cast<mlir::RankedTensorType>(convert_out.getType())
            .getElementType();
    final_lhs = builder.create<mlir::stablehlo::ConvertOp>(
        loc, mlir::RankedTensorType::get(lhs_type.getShape(), res_elem_type),
        final_lhs);
    final_rhs = builder.create<mlir::stablehlo::ConvertOp>(
        loc, mlir::RankedTensorType::get(rhs_type.getShape(), res_elem_type),
        final_rhs);
  }

  auto new_concat = builder.create<mlir::stablehlo::ConcatenateOp>(
      loc, mlir::ValueRange{final_lhs, final_rhs}, dim);
  target_op->replaceAllUsesWith(new_concat.getOperation());
  SimplifyConcatSliceUsers(new_concat, final_lhs, final_rhs, dim);
}

bool TryPartitionConcatSoftmaxAlongDim(mlir::stablehlo::DivOp div_op,
                                       mlir::stablehlo::ExpOp exp_op,
                                       mlir::stablehlo::SubtractOp shifted_op,
                                       mlir::Type comp_elem_type, int64_t dim) {
  mlir::Value sum_in;
  mlir::stablehlo::ReduceOp sum_reduce_op;
  if (!MatchBroadcastedSingleDimReduce(div_op.getRhs(), dim, sum_in,
                                       /*is_max=*/false, &sum_reduce_op) ||
      sum_in != exp_op.getResult()) {
    return false;
  }

  mlir::Value in_val = shifted_op.getLhs();
  mlir::Value max_in;
  mlir::stablehlo::ReduceOp max_reduce_op;
  if (!MatchBroadcastedSingleDimReduce(shifted_op.getRhs(), dim, max_in,
                                       /*is_max=*/true, &max_reduce_op) ||
      max_in != in_val) {
    return false;
  }

  mlir::Value raw_in = StripOptionalFloatUpcast(in_val, comp_elem_type);
  mlir::Value lhs;
  mlir::Value rhs;
  mlir::RankedTensorType lhs_type;
  mlir::RankedTensorType rhs_type;
  if (!MatchSmallTailConcatInput(raw_in, dim, lhs, rhs, lhs_type, rhs_type)) {
    return false;
  }

  EmitPartitionedConcatSoftmax(div_op, comp_elem_type, dim, lhs, rhs, lhs_type,
                               rhs_type, max_reduce_op, sum_reduce_op);
  return true;
}

void TryPartitionSmallTailConcatSoftmax(mlir::stablehlo::DivOp div_op) {
  auto div_type = mlir::dyn_cast<mlir::RankedTensorType>(div_op.getType());
  if (!div_type || !div_type.hasStaticShape() || div_type.getRank() < 1 ||
      !mlir::isa<mlir::FloatType>(div_type.getElementType())) {
    return;
  }
  auto exp_op = div_op.getLhs().getDefiningOp<mlir::stablehlo::ExpOp>();
  if (!exp_op) {
    return;
  }
  auto shifted_op =
      exp_op.getOperand().getDefiningOp<mlir::stablehlo::SubtractOp>();
  if (!shifted_op) {
    return;
  }
  for (int64_t dim = 0; dim < div_type.getRank(); ++dim) {
    if (TryPartitionConcatSoftmaxAlongDim(div_op, exp_op, shifted_op,
                                          div_type.getElementType(), dim)) {
      break;
    }
  }
}

// Partitions softmax computed over `concat(lhs, rhs)` where `rhs` is a small
// tail (e.g. attention sinks) and only the `lhs` slice of probabilities is
// used. Reductions are partitioned across `lhs` and `rhs` directly:
//   max(concat(lhs, rhs)) = max(max(lhs), max(rhs))
//   sum(exp(...)) = sum(exp(lhs - m)) + sum(exp(rhs - m))
// allowing the `lhs` softmax to be evaluated without materializing the
// concatenated intermediate tensor or intermediate activations.
void PartitionSmallTailConcatSoftmax(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::stablehlo::DivOp> div_ops;
  func.walk(
      [&](mlir::stablehlo::DivOp op) {  // REF_CAPTURE_OK=called synchronously
        div_ops.push_back(op);
      });
  for (mlir::stablehlo::DivOp div_op : div_ops) {
    TryPartitionSmallTailConcatSoftmax(div_op);
  }
  EraseTriviallyDeadOps(func);
}

struct FoldConcatSoftmaxPass
    : public mlir::PassWrapper<FoldConcatSoftmaxPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldConcatSoftmaxPass)

  llvm::StringRef getArgument() const override { return "fold-concat-softmax"; }
  llvm::StringRef getDescription() const override {
    return "Folds shift-invariant softmax pre-subtractions and partitions "
           "softmax across small tail concatenations.";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    FoldShiftInvariantPreSubtraction(func);
    PartitionSmallTailConcatSoftmax(func);
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createFoldConcatSoftmaxPass() {
  return std::make_unique<FoldConcatSoftmaxPass>();
}

}  // namespace torch_tpu
