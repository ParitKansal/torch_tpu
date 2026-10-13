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

#include "csrc/mlir/passes/partition_concat_softmax_and_deinterleave.h"

#include <cmath>
#include <cstdint>
#include <vector>

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

TEST(PartitionConcatSoftmaxAndDeinterleaveTest,
     PartitionsReciprocalConcatSoftmaxAndTrainingBackward) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%lhs: tensor<1x2x16x16xbf16>, %rhs: tensor<1x2x16x1xbf16>, %grad_lhs: tensor<1x2x16x16xbf16>) -> (tensor<1x2x16x16xbf16>, tensor<1x2x16x16xbf16>, tensor<1x2x16x1xbf16>) {
    %cat = stablehlo.concatenate %lhs, %rhs, dim = 3 : (tensor<1x2x16x16xbf16>, tensor<1x2x16x1xbf16>) -> tensor<1x2x16x17xbf16>
    %iota = stablehlo.iota dim = 3 : tensor<1x2x16x17xi64>
    %neg_inf_bf16 = stablehlo.constant dense<-3.389530e+38> : tensor<bf16>
    %zero_i64 = stablehlo.constant dense<0> : tensor<i64>
    %max_out:2 = "stablehlo.reduce"(%cat, %iota, %neg_inf_bf16, %zero_i64) ({
    ^bb0(%v0: tensor<bf16>, %i0: tensor<i64>, %v1: tensor<bf16>, %i1: tensor<i64>):
      %cmp = stablehlo.compare GE, %v0, %v1 : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      %mv = stablehlo.maximum %v0, %v1 : tensor<bf16>
      %mi = stablehlo.select %cmp, %i0, %i1 : tensor<i1>, tensor<i64>
      stablehlo.return %mv, %mi : tensor<bf16>, tensor<i64>
    }) {dimensions = array<i64: 3>} : (tensor<1x2x16x17xbf16>, tensor<1x2x16x17xi64>, tensor<bf16>, tensor<i64>) -> (tensor<1x2x16xbf16>, tensor<1x2x16xi64>)
    %max_unsq = stablehlo.reshape %max_out#0 : (tensor<1x2x16xbf16>) -> tensor<1x2x16x1xbf16>
    %max_bcast = stablehlo.broadcast_in_dim %max_unsq, dims = [0, 1, 2, 3] : (tensor<1x2x16x1xbf16>) -> tensor<1x2x16x17xbf16>
    %pre_sub = stablehlo.subtract %cat, %max_bcast : tensor<1x2x16x17xbf16>
    %in_f32 = stablehlo.convert %pre_sub : (tensor<1x2x16x17xbf16>) -> tensor<1x2x16x17xf32>
    %neg_inf_f32 = stablehlo.constant dense<-3.40282347e+38> : tensor<f32>
    %s_max = stablehlo.reduce(%in_f32 init: %neg_inf_f32) applies stablehlo.maximum across dimensions = [3] : (tensor<1x2x16x17xf32>, tensor<f32>) -> tensor<1x2x16xf32>
    %s_max_bcast = stablehlo.broadcast_in_dim %s_max, dims = [0, 1, 2] : (tensor<1x2x16xf32>) -> tensor<1x2x16x17xf32>
    %shifted = stablehlo.subtract %in_f32, %s_max_bcast : tensor<1x2x16x17xf32>
    %exp = stablehlo.exponential %shifted : tensor<1x2x16x17xf32>
    %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %sum = stablehlo.reduce(%exp init: %zero_f32) applies stablehlo.add across dimensions = [3] : (tensor<1x2x16x17xf32>, tensor<f32>) -> tensor<1x2x16xf32>
    %sum_unsq = stablehlo.reshape %sum : (tensor<1x2x16xf32>) -> tensor<1x2x16x1xf32>
    %one_f32 = stablehlo.constant dense<1.000000e+00> : tensor<1x2x16x1xf32>
    %inv_sum = stablehlo.divide %one_f32, %sum_unsq : tensor<1x2x16x1xf32>
    %inv_bcast = stablehlo.broadcast_in_dim %inv_sum, dims = [0, 1, 2, 3] : (tensor<1x2x16x1xf32>) -> tensor<1x2x16x17xf32>
    %probs_f32 = stablehlo.multiply %exp, %inv_bcast : tensor<1x2x16x17xf32>
    %probs = stablehlo.convert %probs_f32 : (tensor<1x2x16x17xf32>) -> tensor<1x2x16x17xbf16>
    %scores = stablehlo.slice %probs [0:1, 0:2, 0:16, 0:16] : (tensor<1x2x16x17xbf16>) -> tensor<1x2x16x16xbf16>

    // Backward pass
    %zero_rhs = stablehlo.constant dense<0.000000e+00> : tensor<1x2x16x1xbf16>
    %grad_cat = stablehlo.concatenate %grad_lhs, %zero_rhs, dim = 3 : (tensor<1x2x16x16xbf16>, tensor<1x2x16x1xbf16>) -> tensor<1x2x16x17xbf16>
    %b_probs = stablehlo.convert %probs : (tensor<1x2x16x17xbf16>) -> tensor<1x2x16x17xf32>
    %b_grad = stablehlo.convert %grad_cat : (tensor<1x2x16x17xbf16>) -> tensor<1x2x16x17xf32>
    %dot = stablehlo.dot_general %b_probs, %b_grad, batching_dims = [0, 1, 2] x [0, 1, 2], contracting_dims = [3] x [3] : (tensor<1x2x16x17xf32>, tensor<1x2x16x17xf32>) -> tensor<1x2x16xf32>
    %dot_bcast = stablehlo.broadcast_in_dim %dot, dims = [0, 1, 2] : (tensor<1x2x16xf32>) -> tensor<1x2x16x17xf32>
    %b_sub = stablehlo.subtract %b_grad, %dot_bcast : tensor<1x2x16x17xf32>
    %b_mul = stablehlo.multiply %b_probs, %b_sub : tensor<1x2x16x17xf32>
    %dy = stablehlo.convert %b_mul : (tensor<1x2x16x17xf32>) -> tensor<1x2x16x17xbf16>
    %neg_dy = stablehlo.negate %dy : tensor<1x2x16x17xbf16>
    %zero_bf16 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %sum_neg_dy = stablehlo.reduce(%neg_dy init: %zero_bf16) applies stablehlo.add across dimensions = [3] : (tensor<1x2x16x17xbf16>, tensor<bf16>) -> tensor<1x2x16xbf16>
    %sum_neg_dy_unsq = stablehlo.reshape %sum_neg_dy : (tensor<1x2x16xbf16>) -> tensor<1x2x16x1xbf16>
    %idx_5d = stablehlo.reshape %max_out#1 : (tensor<1x2x16xi64>) -> tensor<1x2x16x1x1xi64>
    %zero_17 = stablehlo.constant dense<0.000000e+00> : tensor<1x2x16x17xbf16>
    %sc = "stablehlo.scatter"(%zero_17, %idx_5d, %sum_neg_dy_unsq) <{
      indices_are_sorted = false,
      scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [],
        inserted_window_dims = [3],
        input_batching_dims = [0, 1, 2],
        scatter_indices_batching_dims = [0, 1, 2],
        scatter_dims_to_operand_dims = [3],
        index_vector_dim = 4
      >,
      unique_indices = false
    }> ({
    ^bb0(%a: tensor<bf16>, %b: tensor<bf16>):
      %r = stablehlo.add %a, %b : tensor<bf16>
      stablehlo.return %r : tensor<bf16>
    }) : (tensor<1x2x16x17xbf16>, tensor<1x2x16x1x1xi64>, tensor<1x2x16x1xbf16>) -> tensor<1x2x16x17xbf16>
    %dx = stablehlo.add %dy, %sc : tensor<1x2x16x17xbf16>
    %dx_lhs = stablehlo.slice %dx [0:1, 0:2, 0:16, 0:16] : (tensor<1x2x16x17xbf16>) -> tensor<1x2x16x16xbf16>
    %dx_rhs = stablehlo.slice %dx [0:1, 0:2, 0:16, 16:17] : (tensor<1x2x16x17xbf16>) -> tensor<1x2x16x1xbf16>
    return %scores, %dx_lhs, %dx_rhs : tensor<1x2x16x16xbf16>, tensor<1x2x16x16xbf16>, tensor<1x2x16x1xbf16>
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

  int dim17_count = 0;
  int scatter_count = 0;
  int concat_count = 0;
  module_opt->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::ScatterOp>(op)) {
      ++scatter_count;
    }
    if (mlir::isa<mlir::stablehlo::ConcatenateOp>(op)) {
      ++concat_count;
    }
    for (mlir::Value res : op->getResults()) {
      if (auto rtt = mlir::dyn_cast<mlir::RankedTensorType>(res.getType())) {
        for (int64_t d : rtt.getShape()) {
          if (d == 17) {
            ++dim17_count;
          }
        }
      }
    }
  });

  EXPECT_EQ(scatter_count, 0);
  EXPECT_EQ(concat_count, 0);
  EXPECT_EQ(dim17_count, 0);

  llvm::SmallVector<mlir::DenseElementsAttr> args = {
      MakeBF16Tensor(context, {1, 2, 16, 16}),
      MakeBF16Tensor(context, {1, 2, 16, 1}),
      MakeBF16Tensor(context, {1, 2, 16, 16}),
  };
  mlir::stablehlo::InterpreterConfiguration config;
  auto ref_out = mlir::stablehlo::evalModule(*module_ref, args, config);
  auto opt_out = mlir::stablehlo::evalModule(*module_opt, args, config);
  ASSERT_TRUE(mlir::succeeded(ref_out));
  ASSERT_TRUE(mlir::succeeded(opt_out));
  ASSERT_EQ(ref_out->size(), 3u);
  ASSERT_EQ(opt_out->size(), 3u);
  for (size_t i = 0; i < 3; ++i) {
    ExpectBF16Close((*ref_out)[i], (*opt_out)[i], 2e-2f);
  }
}

TEST(PartitionConcatSoftmaxAndDeinterleaveTest,
     FoldsDisjointMinorDimTwoConcatAdd) {
  static constexpr const char* kIr = R"mlir(
module {
  func.func @main(%x: tensor<128x320xbf16>, %de: tensor<128x160xbf16>, %do: tensor<128x160xbf16>) -> (tensor<128x160xbf16>, tensor<128x320xbf16>) {
    %r = stablehlo.reshape %x : (tensor<128x320xbf16>) -> tensor<128x160x2xbf16>
    %s_e = stablehlo.slice %r [0:128, 0:160, 0:1] : (tensor<128x160x2xbf16>) -> tensor<128x160x1xbf16>
    %s_o = stablehlo.slice %r [0:128, 0:160, 1:2] : (tensor<128x160x2xbf16>) -> tensor<128x160x1xbf16>
    %e = stablehlo.reshape %s_e : (tensor<128x160x1xbf16>) -> tensor<128x160xbf16>
    %o = stablehlo.reshape %s_o : (tensor<128x160x1xbf16>) -> tensor<128x160xbf16>
    %prod = stablehlo.multiply %e, %o : tensor<128x160xbf16>

    %z = stablehlo.constant dense<0.000000e+00> : tensor<128x160x1xbf16>
    %de_3d = stablehlo.reshape %de : (tensor<128x160xbf16>) -> tensor<128x160x1xbf16>
    %do_3d = stablehlo.reshape %do : (tensor<128x160xbf16>) -> tensor<128x160x1xbf16>
    %c0 = stablehlo.concatenate %z, %do_3d, dim = 2 : (tensor<128x160x1xbf16>, tensor<128x160x1xbf16>) -> tensor<128x160x2xbf16>
    %c1 = stablehlo.concatenate %de_3d, %z, dim = 2 : (tensor<128x160x1xbf16>, tensor<128x160x1xbf16>) -> tensor<128x160x2xbf16>
    %r0 = stablehlo.reshape %c0 : (tensor<128x160x2xbf16>) -> tensor<128x320xbf16>
    %r1 = stablehlo.reshape %c1 : (tensor<128x160x2xbf16>) -> tensor<128x320xbf16>
    %dx = stablehlo.add %r0, %r1 : tensor<128x320xbf16>
    return %prod, %dx : tensor<128x160xbf16>, tensor<128x320xbf16>
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

  int concat_count = 0;
  int add_count = 0;
  module_opt->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::ConcatenateOp>(op)) {
      ++concat_count;
    }
    if (mlir::isa<mlir::stablehlo::AddOp>(op)) {
      ++add_count;
    }
  });

  EXPECT_EQ(concat_count, 1);
  EXPECT_EQ(add_count, 0);

  llvm::SmallVector<mlir::DenseElementsAttr> args = {
      MakeBF16Tensor(context, {128, 320}),
      MakeBF16Tensor(context, {128, 160}),
      MakeBF16Tensor(context, {128, 160}),
  };
  mlir::stablehlo::InterpreterConfiguration config;
  auto ref_out = mlir::stablehlo::evalModule(*module_ref, args, config);
  auto opt_out = mlir::stablehlo::evalModule(*module_opt, args, config);
  ASSERT_TRUE(mlir::succeeded(ref_out));
  ASSERT_TRUE(mlir::succeeded(opt_out));
  ASSERT_EQ(ref_out->size(), 2u);
  ASSERT_EQ(opt_out->size(), 2u);
  ExpectBF16Close((*ref_out)[0], (*opt_out)[0], 0.0f);
  ExpectBF16Close((*ref_out)[1], (*opt_out)[1], 0.0f);
}

}  // namespace
}  // namespace torch_tpu
