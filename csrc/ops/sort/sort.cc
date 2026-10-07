/*
 * Copyright 2025 Google LLC
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

#include "csrc/ops/sort/sort.h"

#include <cstdint>
#include <limits>
#include <optional>

#include "csrc/ops/op_builder_utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

namespace {

bool IsInt32BoundedConvert(stablehlo::ConvertOp conv) {
  mlir::Type src_type = mlir::getElementTypeOrSelf(conv.getOperand().getType());
  if (auto int_type = mlir::dyn_cast<mlir::IntegerType>(src_type)) {
    return int_type.getWidth() <= 32 &&
           (int_type.getWidth() < 32 || !int_type.isUnsigned());
  }
  return false;
}

bool IsInt32BoundedDim(mlir::Type type, uint64_t dim) {
  auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!out_type || !out_type.hasStaticShape() ||
      dim >= static_cast<uint64_t>(out_type.getRank())) {
    return false;
  }
  return out_type.getDimSize(dim) <=
         static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
}

bool IsInt32BoundedConstant(stablehlo::ConstantOp cst) {
  auto dense = mlir::dyn_cast<mlir::DenseIntElementsAttr>(cst.getValue());
  if (!dense) {
    return false;
  }
  for (const llvm::APInt& v : dense.getValues<llvm::APInt>()) {
    const int64_t s = v.getSExtValue();
    if (s < std::numeric_limits<int32_t>::min() ||
        s > std::numeric_limits<int32_t>::max()) {
      return false;
    }
  }
  return true;
}

bool IsInt32BoundedSortResult(mlir::Value val, stablehlo::SortOp sort_op) {
  auto res = mlir::dyn_cast<mlir::OpResult>(val);
  return res && res.getResultNumber() == 1 &&
         IsInt32BoundedDim(val.getType(), sort_op.getDimension());
}

mlir::Value UnwrapUnaryViewOperand(mlir::Operation* def) {
  if (auto reshape = mlir::dyn_cast<stablehlo::ReshapeOp>(def)) {
    return reshape.getOperand();
  }
  if (auto transpose = mlir::dyn_cast<stablehlo::TransposeOp>(def)) {
    return transpose.getOperand();
  }
  if (auto bcast = mlir::dyn_cast<stablehlo::BroadcastInDimOp>(def)) {
    return bcast.getOperand();
  }
  if (auto slice = mlir::dyn_cast<stablehlo::SliceOp>(def)) {
    return slice.getOperand();
  }
  return nullptr;
}

bool IsInt32BoundedSortKey(mlir::Value val) {
  for (int depth = 0; val && depth <= 8; ++depth) {
    auto* def = val.getDefiningOp();
    if (!def) {
      return false;
    }
    if (auto conv = mlir::dyn_cast<stablehlo::ConvertOp>(def)) {
      return IsInt32BoundedConvert(conv);
    }
    if (auto iota = mlir::dyn_cast<stablehlo::IotaOp>(def)) {
      return IsInt32BoundedDim(iota.getType(), iota.getIotaDimension());
    }
    if (auto cst = mlir::dyn_cast<stablehlo::ConstantOp>(def)) {
      return IsInt32BoundedConstant(cst);
    }
    if (auto sort_op = mlir::dyn_cast<stablehlo::SortOp>(def)) {
      return IsInt32BoundedSortResult(val, sort_op);
    }
    val = UnwrapUnaryViewOperand(def);
  }
  return false;
}

}  // namespace

SortShloOutputs BuildSortShlo(mlir::MlirOp input_op, bool stable, int64_t dim,
                              bool descending) {
  const mlir::RankedTensorType inputType = GetTensorTypeOrDie(input_op);
  mlir::MlirBuilder& builder = input_op.getBuilder();
  const bool use_i32_indices =
      dim >= 0 && dim < inputType.getRank() &&
      inputType.getShape()[dim] != mlir::ShapedType::kDynamic &&
      inputType.getShape()[dim] <=
          static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
  const bool use_i32_key =
      GetElementTypeOrDie(input_op) == mlir::ElementType::I64 &&
      IsInt32BoundedSortKey(input_op.getValue());

  mlir::MlirOp sort_key = use_i32_key ? stablehlo::ConvertElementType(
                                            input_op, mlir::ElementType::I32)
                                      : input_op;
  const mlir::Type sort_key_elem_type =
      GetTensorTypeOrDie(sort_key).getElementType();
  const mlir::ElementType index_elem_type =
      use_i32_indices ? mlir::ElementType::I32 : mlir::ElementType::I64;

  mlir::MlirOp indices =
      stablehlo::Iota(builder,
                      makeTensorType(builder.getContext(), inputType.getShape(),
                                     index_elem_type),
                      dim);
  auto comparator = [sort_key_elem_type, use_i32_indices,
                     descending](mlir::RegionBuilder& rb) {
    mlir::OpBuilder& op_builder = rb.getOpBuilder();
    std::optional<llvm::StringRef> compare_type = std::nullopt;
    if (mlir::isa<mlir::FloatType>(sort_key_elem_type)) {
      compare_type = "TOTALORDER";
    }
    const mlir::Type idx_mlir_type =
        use_i32_indices ? op_builder.getI32Type() : op_builder.getI64Type();
    stablehlo::buildSortComparisonBody(
        {sort_key_elem_type, idx_mlir_type},
        descending ? stablehlo::ComparisonDirection::GT
                   : stablehlo::ComparisonDirection::LT,
        compare_type, &rb.getRegion(), &op_builder);
  };
  auto outputs =
      stablehlo::Sort(builder, {sort_key, indices}, comparator, dim, stable);
  mlir::MlirOp sorted_values =
      use_i32_key
          ? stablehlo::ConvertElementType(outputs[0], mlir::ElementType::I64)
          : outputs[0];
  mlir::MlirOp sorted_indices =
      use_i32_indices
          ? stablehlo::ConvertElementType(outputs[1], mlir::ElementType::I64)
          : outputs[1];
  return SortShloOutputs{.values = sorted_values, .indices = sorted_indices};
}

}  // namespace torch_tpu
