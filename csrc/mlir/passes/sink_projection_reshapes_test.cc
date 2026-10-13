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

#include <cstdint>
#include <string>

#include "absl/status/status_matchers.h"
#include "csrc/mlir/pipeline.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

using ::testing::HasSubstr;

void InitContext(mlir::MLIRContext& context) {
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::func::FuncDialect>();
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
}

std::string PrintModule(mlir::ModuleOp module) {
  std::string out;
  llvm::raw_string_ostream os(out);
  module.print(os);
  return out;
}

template <typename OpTy>
int CountOps(mlir::ModuleOp module) {
  int count = 0;
  module.walk([&](OpTy) { ++count; });
  return count;
}

TEST(SinkProjectionReshapesTest, Sinks4DMultiHeadQkvReshapeAndTranspose) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @q_proj_4d(
    %x: tensor<16x1500x1280xbf16>,
    %w: tensor<1280x1280xbf16>,
    %bias: tensor<1280xbf16>) -> tensor<16x20x1500x64xbf16> {
  %x_flat = stablehlo.reshape %x : (tensor<16x1500x1280xbf16>) -> tensor<24000x1280xbf16>
  %dot = stablehlo.dot_general %x_flat, %w, contracting_dims = [1] x [0]
      : (tensor<24000x1280xbf16>, tensor<1280x1280xbf16>) -> tensor<24000x1280xbf16>
  %b_bcast = stablehlo.broadcast_in_dim %bias, dims = [1]
      : (tensor<1280xbf16>) -> tensor<24000x1280xbf16>
  %biased = stablehlo.add %b_bcast, %dot : tensor<24000x1280xbf16>
  %r3d = stablehlo.reshape %biased : (tensor<24000x1280xbf16>) -> tensor<16x1500x1280xbf16>
  %scale = stablehlo.constant dense<0.125> : tensor<16x1500x1280xbf16>
  %scaled = stablehlo.multiply %r3d, %scale : tensor<16x1500x1280xbf16>
  %r4d = stablehlo.reshape %scaled : (tensor<16x1500x1280xbf16>) -> tensor<16x1500x20x64xbf16>
  %out = stablehlo.transpose %r4d, dims = [0, 2, 1, 3]
      : (tensor<16x1500x20x64xbf16>) -> tensor<16x20x1500x64xbf16>
  return %out : tensor<16x20x1500x64xbf16>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);
  ASSERT_OK(RunTorchTpuOptimizationPasses(*module, &context));

  std::string printed = PrintModule(*module);
  // The 4D reshape and transpose now directly wrap %dot, and bias is reshaped
  // to tensor<20x64xbf16> and broadcast along dims = [1, 3].
  EXPECT_THAT(printed, HasSubstr("tensor<20x64xbf16>"));
  EXPECT_THAT(printed, HasSubstr("dims = [1, 3]"));
}

TEST(SinkProjectionReshapesTest, SinksLongSeq3DProjectionReshape) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @fc1_3d(
    %x: tensor<16x1500x1280xbf16>,
    %w: tensor<1280x5120xbf16>,
    %bias: tensor<5120xbf16>) -> tensor<16x1500x5120xbf16> {
  %x_flat = stablehlo.reshape %x : (tensor<16x1500x1280xbf16>) -> tensor<24000x1280xbf16>
  %dot = stablehlo.dot_general %x_flat, %w, contracting_dims = [1] x [0]
      : (tensor<24000x1280xbf16>, tensor<1280x5120xbf16>) -> tensor<24000x5120xbf16>
  %b_bcast = stablehlo.broadcast_in_dim %bias, dims = [1]
      : (tensor<5120xbf16>) -> tensor<24000x5120xbf16>
  %biased = stablehlo.add %b_bcast, %dot : tensor<24000x5120xbf16>
  %out = stablehlo.reshape %biased : (tensor<24000x5120xbf16>) -> tensor<16x1500x5120xbf16>
  return %out : tensor<16x1500x5120xbf16>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);
  ASSERT_OK(RunTorchTpuOptimizationPasses(*module, &context));

  std::string printed = PrintModule(*module);
  EXPECT_THAT(printed, HasSubstr("dims = [2]"));
}

TEST(SinkProjectionReshapesTest, PreservesTrainingGradWeightGraph) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @train_linear_with_dw(
    %x: tensor<16x1500x1280xbf16>,
    %w: tensor<1280x5120xbf16>,
    %bias: tensor<5120xbf16>,
    %grad_out: tensor<24000x5120xbf16>) -> (tensor<16x1500x5120xbf16>, tensor<5120x1280xbf16>) {
  %x_flat = stablehlo.reshape %x : (tensor<16x1500x1280xbf16>) -> tensor<24000x1280xbf16>
  %dot = stablehlo.dot_general %x_flat, %w, contracting_dims = [1] x [0]
      : (tensor<24000x1280xbf16>, tensor<1280x5120xbf16>) -> tensor<24000x5120xbf16>
  %b_bcast = stablehlo.broadcast_in_dim %bias, dims = [1]
      : (tensor<5120xbf16>) -> tensor<24000x5120xbf16>
  %biased = stablehlo.add %b_bcast, %dot : tensor<24000x5120xbf16>
  %out = stablehlo.reshape %biased : (tensor<24000x5120xbf16>) -> tensor<16x1500x5120xbf16>
  %dw = stablehlo.dot_general %grad_out, %x_flat, contracting_dims = [0] x [0]
      : (tensor<24000x5120xbf16>, tensor<24000x1280xbf16>) -> tensor<5120x1280xbf16>
  return %out, %dw : tensor<16x1500x5120xbf16>, tensor<5120x1280xbf16>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);
  ASSERT_OK(RunTorchTpuOptimizationPasses(*module, &context));

  std::string printed = PrintModule(*module);
  // Because %x_flat has a grad_weight user contracting on dim 0, the 2D bias
  // broadcast dims = [1] is preserved.
  EXPECT_THAT(printed, HasSubstr("dims = [1]"));
}

}  // namespace
}  // namespace torch_tpu
