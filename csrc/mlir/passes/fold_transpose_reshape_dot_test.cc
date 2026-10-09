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

#include "absl/status/status_matchers.h"
#include "csrc/mlir/pipeline.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

class FoldTransposeReshapeDotTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context_
        .loadDialect<mlir::func::FuncDialect, mlir::stablehlo::StablehloDialect,
                     mlir::chlo::ChloDialect>();
  }

  mlir::OwningOpRef<mlir::ModuleOp> ParseModule(const char* ir) {
    return mlir::parseSourceString<mlir::ModuleOp>(ir, &context_);
  }

  mlir::MLIRContext context_;
};

TEST_F(FoldTransposeReshapeDotTest, FoldsForwardAttentionOProjAndResidualAdd) {
  const char* kIr = R"mlir(
    module {
      func.func @main(
          %attn: tensor<8x16x2048x128xbf16>,
          %w: tensor<1024x2048xbf16>,
          %res: tensor<8x2048x1024xbf16>) -> tensor<8x2048x1024xbf16> {
        %t = stablehlo.transpose %attn, dims = [0, 2, 1, 3] : (tensor<8x16x2048x128xbf16>) -> tensor<8x2048x16x128xbf16>
        %r = stablehlo.reshape %t : (tensor<8x2048x16x128xbf16>) -> tensor<16384x2048xbf16>
        %wt = stablehlo.transpose %w, dims = [1, 0] : (tensor<1024x2048xbf16>) -> tensor<2048x1024xbf16>
        %dot = stablehlo.dot_general %r, %wt, contracting_dims = [1] x [0] : (tensor<16384x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<16384x1024xbf16>
        %proj3d = stablehlo.reshape %dot : (tensor<16384x1024xbf16>) -> tensor<8x2048x1024xbf16>
        %out = stablehlo.add %res, %proj3d : tensor<8x2048x1024xbf16>
        return %out : tensor<8x2048x1024xbf16>
      }
    }
  )mlir";

  auto module = ParseModule(kIr);
  ASSERT_TRUE(module);
  ASSERT_OK(RunTorchTpuOptimizationPasses(*module, &context_));

  int transpose_count = 0;
  int dot_count = 0;
  module->walk([&](mlir::stablehlo::TransposeOp) { ++transpose_count; });
  module->walk([&](mlir::stablehlo::DotGeneralOp dot) {
    ++dot_count;
    auto dims = dot.getDotDimensionNumbersAttr();
    EXPECT_EQ(dims.getLhsContractingDimensions().size(), 2);
    EXPECT_EQ(dims.getLhsContractingDimensions()[0], 1);
    EXPECT_EQ(dims.getLhsContractingDimensions()[1], 3);
    EXPECT_EQ(dims.getRhsContractingDimensions().size(), 2);
    EXPECT_EQ(dims.getRhsContractingDimensions()[0], 1);
    EXPECT_EQ(dims.getRhsContractingDimensions()[1], 2);
  });
  EXPECT_EQ(transpose_count, 0);
  EXPECT_EQ(dot_count, 1);
}

TEST_F(FoldTransposeReshapeDotTest, FoldsBiasedAddmmAndBackwardDxProjection) {
  const char* kIr = R"mlir(
    module {
      func.func @main(
          %dq: tensor<8x8x2048x128xbf16>,
          %wk: tensor<1024x1024xbf16>,
          %bias: tensor<1024xbf16>) -> tensor<8x2048x1024xbf16> {
        %t = stablehlo.transpose %dq, dims = [0, 2, 1, 3] : (tensor<8x8x2048x128xbf16>) -> tensor<8x2048x8x128xbf16>
        %r = stablehlo.reshape %t : (tensor<8x2048x8x128xbf16>) -> tensor<16384x1024xbf16>
        %dot = stablehlo.dot_general %r, %wk, contracting_dims = [1] x [0] : (tensor<16384x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<16384x1024xbf16>
        %bcast = stablehlo.broadcast_in_dim %bias, dims = [1] : (tensor<1024xbf16>) -> tensor<16384x1024xbf16>
        %add = stablehlo.add %bcast, %dot : tensor<16384x1024xbf16>
        %out = stablehlo.reshape %add : (tensor<16384x1024xbf16>) -> tensor<8x2048x1024xbf16>
        return %out : tensor<8x2048x1024xbf16>
      }
    }
  )mlir";

  auto module = ParseModule(kIr);
  ASSERT_TRUE(module);
  ASSERT_OK(RunTorchTpuOptimizationPasses(*module, &context_));

  int transpose_count = 0;
  module->walk([&](mlir::stablehlo::TransposeOp) { ++transpose_count; });
  EXPECT_EQ(transpose_count, 0);
}

TEST_F(FoldTransposeReshapeDotTest, SkipsNonLaneWidthMinorContractingDim) {
  const char* kIr = R"mlir(
    module {
      func.func @main(
          %x64: tensor<8x16x2048x64xbf16>,
          %w64: tensor<1024x1024xbf16>,
          %x256: tensor<8x8x2048x256xbf16>,
          %w256: tensor<2048x1024xbf16>)
          -> (tensor<16384x1024xbf16>, tensor<16384x1024xbf16>) {
        %t64 = stablehlo.transpose %x64, dims = [0, 2, 1, 3] : (tensor<8x16x2048x64xbf16>) -> tensor<8x2048x16x64xbf16>
        %r64 = stablehlo.reshape %t64 : (tensor<8x2048x16x64xbf16>) -> tensor<16384x1024xbf16>
        %d64 = stablehlo.dot_general %r64, %w64, contracting_dims = [1] x [0] : (tensor<16384x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<16384x1024xbf16>
        %t256 = stablehlo.transpose %x256, dims = [0, 2, 1, 3] : (tensor<8x8x2048x256xbf16>) -> tensor<8x2048x8x256xbf16>
        %r256 = stablehlo.reshape %t256 : (tensor<8x2048x8x256xbf16>) -> tensor<16384x2048xbf16>
        %d256 = stablehlo.dot_general %r256, %w256, contracting_dims = [1] x [0] : (tensor<16384x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<16384x1024xbf16>
        return %d64, %d256 : tensor<16384x1024xbf16>, tensor<16384x1024xbf16>
      }
    }
  )mlir";

  auto module = ParseModule(kIr);
  ASSERT_TRUE(module);
  ASSERT_OK(RunTorchTpuOptimizationPasses(*module, &context_));

  int transpose_count = 0;
  module->walk([&](mlir::stablehlo::TransposeOp) { ++transpose_count; });
  EXPECT_EQ(transpose_count, 2);
}

}  // namespace
}  // namespace torch_tpu
