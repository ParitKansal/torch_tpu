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
using ::testing::Not;

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

TEST(OptimizeSdpaSoftmaxTest, RewritesUnalignedForwardAndBackwardRowsum) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @conformer_sdpa_fwd_bwd(
    %logits: tensor<16x4x142x142xbf16>,
    %v: tensor<16x4x142x64xbf16>,
    %ds: tensor<16x4x142x142xf32>,
    %row_masked: tensor<16x4x142xi1>) -> (tensor<16x4x142x64xbf16>, tensor<16x4x142x142xf32>) {
  %neg_inf = stablehlo.constant dense<0xFF80> : tensor<bf16>
  %zero_f32 = stablehlo.constant dense<0.0> : tensor<f32>
  %zero_3d = stablehlo.constant dense<0.0> : tensor<16x4x142xf32>
  %one_3d = stablehlo.constant dense<1.0> : tensor<16x4x142xf32>
  %zero_4d = stablehlo.constant dense<0.0> : tensor<16x4x142x142xf32>
  %mask_bcast = stablehlo.broadcast_in_dim %row_masked, dims = [0, 1, 2]
      : (tensor<16x4x142xi1>) -> tensor<16x4x142x142xi1>

  %max_l = stablehlo.reduce(%logits init: %neg_inf) applies stablehlo.maximum across dimensions = [3]
      : (tensor<16x4x142x142xbf16>, tensor<bf16>) -> tensor<16x4x142xbf16>
  %max_bcast = stablehlo.broadcast_in_dim %max_l, dims = [0, 1, 2]
      : (tensor<16x4x142xbf16>) -> tensor<16x4x142x142xbf16>
  %shifted = stablehlo.subtract %logits, %max_bcast : tensor<16x4x142x142xbf16>
  %shifted_f32 = stablehlo.convert %shifted : (tensor<16x4x142x142xbf16>) -> tensor<16x4x142x142xf32>
  %exp_val = stablehlo.exponential %shifted_f32 : tensor<16x4x142x142xf32>
  %sum_exp = stablehlo.reduce(%exp_val init: %zero_f32) applies stablehlo.add across dimensions = [3]
      : (tensor<16x4x142x142xf32>, tensor<f32>) -> tensor<16x4x142xf32>
  %sum_bcast = stablehlo.broadcast_in_dim %sum_exp, dims = [0, 1, 2]
      : (tensor<16x4x142xf32>) -> tensor<16x4x142x142xf32>
  %probs_f32 = stablehlo.divide %exp_val, %sum_bcast : tensor<16x4x142x142xf32>
  %masked_probs_f32 = stablehlo.select %mask_bcast, %zero_4d, %probs_f32
      : tensor<16x4x142x142xi1>, tensor<16x4x142x142xf32>
  %probs = stablehlo.convert %masked_probs_f32 : (tensor<16x4x142x142xf32>) -> tensor<16x4x142x142xbf16>
  %out = stablehlo.dot_general %probs, %v, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2]
      : (tensor<16x4x142x142xbf16>, tensor<16x4x142x64xbf16>) -> tensor<16x4x142x64xbf16>

  // Backward softmax + ComputeGradP
  %is_zero = stablehlo.compare EQ, %sum_exp, %zero_3d : (tensor<16x4x142xf32>, tensor<16x4x142xf32>) -> tensor<16x4x142xi1>
  %safe_sum = stablehlo.select %is_zero, %one_3d, %sum_exp : tensor<16x4x142xi1>, tensor<16x4x142xf32>
  %safe_bcast = stablehlo.broadcast_in_dim %safe_sum, dims = [0, 1, 2]
      : (tensor<16x4x142xf32>) -> tensor<16x4x142x142xf32>
  %probs_bwd_raw = stablehlo.divide %exp_val, %safe_bcast : tensor<16x4x142x142xf32>
  %probs_bwd = stablehlo.select %mask_bcast, %zero_4d, %probs_bwd_raw
      : tensor<16x4x142x142xi1>, tensor<16x4x142x142xf32>
  %s_ds = stablehlo.multiply %probs_bwd, %ds : tensor<16x4x142x142xf32>
  %rowsum = stablehlo.reduce(%s_ds init: %zero_f32) applies stablehlo.add across dimensions = [3]
      : (tensor<16x4x142x142xf32>, tensor<f32>) -> tensor<16x4x142xf32>
  %rowsum_bcast = stablehlo.broadcast_in_dim %rowsum, dims = [0, 1, 2]
      : (tensor<16x4x142xf32>) -> tensor<16x4x142x142xf32>
  %shifted_ds = stablehlo.subtract %ds, %rowsum_bcast : tensor<16x4x142x142xf32>
  %dp = stablehlo.multiply %probs_bwd, %shifted_ds : tensor<16x4x142x142xf32>
  return %out, %dp : tensor<16x4x142x64xbf16>, tensor<16x4x142x142xf32>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);
  ASSERT_OK(RunTorchTpuOptimizationPasses(*module, &context));

  // All 4D SubtractOps and 4D DivOps are eliminated; only 3D NegOps and 3D
  // DivOps remain.
  EXPECT_EQ(CountOps<mlir::stablehlo::SubtractOp>(*module), 0);
  EXPECT_GE(CountOps<mlir::stablehlo::NegOp>(*module), 2);
  module->walk([&](mlir::stablehlo::DivOp div_op) {
    auto div_type =
        mlir::cast<mlir::RankedTensorType>(div_op.getResult().getType());
    EXPECT_EQ(div_type.getRank(), 3);
  });
  std::string printed = PrintModule(*module);
  EXPECT_THAT(printed, HasSubstr("stablehlo.negate"));
}

TEST(OptimizeSdpaSoftmaxTest,
     SinksMaskedSelectAndNormalizesGqaForOnlineSoftmax) {
  mlir::MLIRContext context;
  InitContext(context);

  const char* kIr = R"mlir(
func.func @gqa_masked_online_sdpa(
    %logits: tensor<2x16x2048x2048xbf16>,
    %v: tensor<2x2x2048x128xbf16>,
    %row_masked: tensor<2x16x2048xi1>) -> tensor<2x16x2048x128xbf16> {
  %neg_inf = stablehlo.constant dense<0xFF80> : tensor<bf16>
  %zero_f32 = stablehlo.constant dense<0.0> : tensor<f32>
  %zero_4d = stablehlo.constant dense<0.0> : tensor<2x16x2048x2048xf32>

  %max_l = stablehlo.reduce(%logits init: %neg_inf) applies stablehlo.maximum across dimensions = [3]
      : (tensor<2x16x2048x2048xbf16>, tensor<bf16>) -> tensor<2x16x2048xbf16>
  %max_bcast = stablehlo.broadcast_in_dim %max_l, dims = [0, 1, 2]
      : (tensor<2x16x2048xbf16>) -> tensor<2x16x2048x2048xbf16>
  %shifted = stablehlo.subtract %logits, %max_bcast : tensor<2x16x2048x2048xbf16>
  %shifted_f32 = stablehlo.convert %shifted : (tensor<2x16x2048x2048xbf16>) -> tensor<2x16x2048x2048xf32>
  %exp_val = stablehlo.exponential %shifted_f32 : tensor<2x16x2048x2048xf32>
  %sum_exp = stablehlo.reduce(%exp_val init: %zero_f32) applies stablehlo.add across dimensions = [3]
      : (tensor<2x16x2048x2048xf32>, tensor<f32>) -> tensor<2x16x2048xf32>
  %sum_bcast = stablehlo.broadcast_in_dim %sum_exp, dims = [0, 1, 2]
      : (tensor<2x16x2048xf32>) -> tensor<2x16x2048x2048xf32>
  %probs_f32 = stablehlo.divide %exp_val, %sum_bcast : tensor<2x16x2048x2048xf32>
  %mask_bcast = stablehlo.broadcast_in_dim %row_masked, dims = [0, 1, 2]
      : (tensor<2x16x2048xi1>) -> tensor<2x16x2048x2048xi1>
  %masked_probs_f32 = stablehlo.select %mask_bcast, %zero_4d, %probs_f32
      : tensor<2x16x2048x2048xi1>, tensor<2x16x2048x2048xf32>
  %probs = stablehlo.convert %masked_probs_f32 : (tensor<2x16x2048x2048xf32>) -> tensor<2x16x2048x2048xbf16>
  %probs_5d = stablehlo.reshape %probs : (tensor<2x16x2048x2048xbf16>) -> tensor<2x2x8x2048x2048xbf16>
  %out_5d = stablehlo.dot_general %probs_5d, %v, batching_dims = [0, 1] x [0, 1], contracting_dims = [4] x [2]
      : (tensor<2x2x8x2048x2048xbf16>, tensor<2x2x2048x128xbf16>) -> tensor<2x2x8x2048x128xbf16>
  %out = stablehlo.reshape %out_5d : (tensor<2x2x8x2048x128xbf16>) -> tensor<2x16x2048x128xbf16>
  return %out : tensor<2x16x2048x128xbf16>
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);
  ASSERT_OK(RunTorchTpuOptimizationPasses(*module, &context));

  std::string printed = PrintModule(*module);
  // The 5D probs reshape is eliminated, V is broadcast to 5D and reshaped to
  // 4D, and Select is now on the 4D [2, 16, 2048, 128] output after DotGeneral.
  EXPECT_THAT(printed, Not(HasSubstr("tensor<2x2x8x2048x2048xbf16>")));
  EXPECT_THAT(printed, HasSubstr("tensor<2x16x2048x128xi1>"));
}

}  // namespace
}  // namespace torch_tpu
