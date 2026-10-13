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

#include "absl/status/status_matchers.h"
#include "csrc/mlir/pipeline.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/Api.h"
#include "stablehlo/reference/Configuration.h"

namespace torch_tpu {
namespace {

using ::absl_testing::IsOk;

void InitContext(mlir::MLIRContext& context) {
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::func::FuncDialect>();
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
}

mlir::DenseElementsAttr MakeBF16Tensor(mlir::MLIRContext& ctx,
                                       llvm::ArrayRef<int64_t> shape) {
  auto type = mlir::RankedTensorType::get(shape, mlir::BFloat16Type::get(&ctx));
  const int64_t n = type.getNumElements();
  llvm::SmallVector<llvm::APFloat> vals;
  vals.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    float f = static_cast<float>((i * 13 + 3) % 64) * 0.125f - 2.0f;
    llvm::APFloat apf(f);
    bool loses_info = false;
    apf.convert(llvm::APFloat::BFloat(), llvm::APFloat::rmNearestTiesToEven,
                &loses_info);
    vals.push_back(apf);
  }
  return mlir::DenseElementsAttr::get(type, vals);
}

void ExpectBF16Close(mlir::DenseElementsAttr a, mlir::DenseElementsAttr b,
                     float tol) {
  ASSERT_EQ(a.getType(), b.getType());
  auto it_a = a.value_begin<llvm::APFloat>();
  auto it_b = b.value_begin<llvm::APFloat>();
  for (int64_t i = 0; i < a.getNumElements(); ++i, ++it_a, ++it_b) {
    EXPECT_NEAR((*it_a).convertToFloat(), (*it_b).convertToFloat(), tol)
        << "Mismatch at index " << i;
  }
}

TEST(FoldViewTransposesAndHalfSplitRopeTest,
     UnifiesHeadSplitTransposesAndFoldsRotateHalf) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%mm: tensor<32x1024xbf16>, %cos: tensor<1x8x32x128xbf16>, %sin: tensor<1x8x32x128xbf16>) -> tensor<1x8x32x128xbf16> {
    %r_full = stablehlo.reshape %mm : (tensor<32x1024xbf16>) -> tensor<32x8x128x1xbf16>
    %q = stablehlo.transpose %r_full, dims = [3, 1, 0, 2] : (tensor<32x8x128x1xbf16>) -> tensor<1x8x32x128xbf16>
    %q_mul = stablehlo.multiply %q, %cos : tensor<1x8x32x128xbf16>

    %r_3d = stablehlo.reshape %mm : (tensor<32x1024xbf16>) -> tensor<32x8x128xbf16>
    %s_hi = stablehlo.slice %r_3d [0:32, 0:8, 64:128] : (tensor<32x8x128xbf16>) -> tensor<32x8x64xbf16>
    %r_hi = stablehlo.reshape %s_hi : (tensor<32x8x64xbf16>) -> tensor<32x8x64x1xbf16>
    %t_hi = stablehlo.transpose %r_hi, dims = [3, 1, 0, 2] : (tensor<32x8x64x1xbf16>) -> tensor<1x8x32x64xbf16>
    %neg_hi = stablehlo.negate %t_hi : tensor<1x8x32x64xbf16>

    %s_lo = stablehlo.slice %r_3d [0:32, 0:8, 0:64] : (tensor<32x8x128xbf16>) -> tensor<32x8x64xbf16>
    %r_lo = stablehlo.reshape %s_lo : (tensor<32x8x64xbf16>) -> tensor<32x8x64x1xbf16>
    %t_lo = stablehlo.transpose %r_lo, dims = [3, 1, 0, 2] : (tensor<32x8x64x1xbf16>) -> tensor<1x8x32x64xbf16>

    %rot = stablehlo.concatenate %neg_hi, %t_lo, dim = 3 : (tensor<1x8x32x64xbf16>, tensor<1x8x32x64xbf16>) -> tensor<1x8x32x128xbf16>
    %rot_mul = stablehlo.multiply %rot, %sin : tensor<1x8x32x128xbf16>
    %add = stablehlo.add %q_mul, %rot_mul : tensor<1x8x32x128xbf16>
    return %add : tensor<1x8x32x128xbf16>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module_ref);
  mlir::OwningOpRef<mlir::ModuleOp> module_opt =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module_opt);

  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module_opt, &context), IsOk());

  int transpose_count = 0;
  int slice_count = 0;
  int concat_count = 0;
  int dot_count = 0;
  module_opt->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::TransposeOp>(op)) {
      ++transpose_count;
    }
    if (mlir::isa<mlir::stablehlo::SliceOp>(op)) {
      ++slice_count;
    }
    if (mlir::isa<mlir::stablehlo::ConcatenateOp>(op)) {
      ++concat_count;
    }
    if (mlir::isa<mlir::stablehlo::DotGeneralOp>(op)) {
      ++dot_count;
    }
  });

  EXPECT_EQ(transpose_count, 1);
  EXPECT_EQ(slice_count, 0);
  EXPECT_EQ(concat_count, 0);
  EXPECT_EQ(dot_count, 1);

  llvm::SmallVector<mlir::DenseElementsAttr> args = {
      MakeBF16Tensor(context, {32, 1024}),
      MakeBF16Tensor(context, {1, 8, 32, 128}),
      MakeBF16Tensor(context, {1, 8, 32, 128}),
  };
  mlir::stablehlo::InterpreterConfiguration config;
  auto ref_out = mlir::stablehlo::evalModule(*module_ref, args, config);
  auto opt_out = mlir::stablehlo::evalModule(*module_opt, args, config);
  ASSERT_TRUE(mlir::succeeded(ref_out));
  ASSERT_TRUE(mlir::succeeded(opt_out));
  ExpectBF16Close((*ref_out)[0], (*opt_out)[0], 0.0f);
}

TEST(FoldViewTransposesAndHalfSplitRopeTest,
     CancelsInverseTransposeAcrossSingleDimSliceAndFoldsBackwardRotateHalf) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<1x64x8x128xbf16>, %dy: tensor<1x8x64x128xbf16>) -> (tensor<1x64x8x64xbf16>, tensor<1x8x64x128xbf16>) {
    %t1 = stablehlo.transpose %x, dims = [0, 2, 1, 3] : (tensor<1x64x8x128xbf16>) -> tensor<1x8x64x128xbf16>
    %sl = stablehlo.slice %t1 [0:1, 0:8, 0:64, 0:64] : (tensor<1x8x64x128xbf16>) -> tensor<1x8x64x64xbf16>
    %t2 = stablehlo.transpose %sl, dims = [0, 2, 1, 3] : (tensor<1x8x64x64xbf16>) -> tensor<1x64x8x64xbf16>

    %dy_hi = stablehlo.slice %dy [0:1, 0:8, 0:64, 64:128] : (tensor<1x8x64x128xbf16>) -> tensor<1x8x64x64xbf16>
    %dy_lo = stablehlo.slice %dy [0:1, 0:8, 0:64, 0:64] : (tensor<1x8x64x128xbf16>) -> tensor<1x8x64x64xbf16>
    %neg_dy_lo = stablehlo.negate %dy_lo : tensor<1x8x64x64xbf16>
    %bwd = stablehlo.concatenate %dy_hi, %neg_dy_lo, dim = 3 : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) -> tensor<1x8x64x128xbf16>
    return %t2, %bwd : tensor<1x64x8x64xbf16>, tensor<1x8x64x128xbf16>
  }
}
)mlir";

  mlir::MLIRContext context;
  InitContext(context);
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module_ref);
  mlir::OwningOpRef<mlir::ModuleOp> module_opt =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module_opt);

  EXPECT_THAT(RunTorchTpuOptimizationPasses(*module_opt, &context), IsOk());

  int transpose_count = 0;
  int slice_count = 0;
  int dot_count = 0;
  module_opt->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::TransposeOp>(op)) {
      ++transpose_count;
    }
    if (mlir::isa<mlir::stablehlo::SliceOp>(op)) {
      ++slice_count;
    }
    if (mlir::isa<mlir::stablehlo::DotGeneralOp>(op)) {
      ++dot_count;
    }
  });

  EXPECT_EQ(transpose_count, 0);
  EXPECT_EQ(slice_count, 1);
  EXPECT_EQ(dot_count, 1);

  llvm::SmallVector<mlir::DenseElementsAttr> args = {
      MakeBF16Tensor(context, {1, 64, 8, 128}),
      MakeBF16Tensor(context, {1, 8, 64, 128}),
  };
  mlir::stablehlo::InterpreterConfiguration config;
  auto ref_out = mlir::stablehlo::evalModule(*module_ref, args, config);
  auto opt_out = mlir::stablehlo::evalModule(*module_opt, args, config);
  ASSERT_TRUE(mlir::succeeded(ref_out));
  ASSERT_TRUE(mlir::succeeded(opt_out));
  ExpectBF16Close((*ref_out)[0], (*opt_out)[0], 0.0f);
  ExpectBF16Close((*ref_out)[1], (*opt_out)[1], 0.0f);
}

}  // namespace
}  // namespace torch_tpu
