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

#ifndef TORCH_TPU_CSRC_COMMON_COMPILATION_TEST_HELPER_H_
#define TORCH_TPU_CSRC_COMMON_COMPILATION_TEST_HELPER_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/strings/str_cat.h"
#include "csrc/common/compilation.h"
#include "csrc/common/compilation_spec.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"

namespace torch_tpu {

// An RAII wrapper for managing compiler option overrides within a scope.
//
// It is particularly useful in tests to ensure that compiler option overrides
// are confined to a specific scope, so that the context state is reset when the
// scope is exited, maintaining clean states between tests.
class ScopedCompilerOptionOverrides {
 public:
  explicit ScopedCompilerOptionOverrides(CompilerOptionOverrides overrides) {
    ABSL_CHECK_OK(  // CRASH_OK
        PushCompilerOptionOverrides(std::move(overrides)));
  }

  ~ScopedCompilerOptionOverrides() { PopCompilerOptionOverrides(); }
};

// Returns a StableHLO module whose main function returns the sum of its
// `num_arguments` f32[128,128] arguments and donates its first argument.
inline xla::MaybeOwningMlirModule MakeSumOfArgumentsModule(
    const int num_arguments) {
  ABSL_CHECK_GE(num_arguments, 2);  // CRASH_OK
  constexpr char kType[] = "tensor<128x128xf32>";
  std::string arguments =
      absl::StrCat("%arg0: ", kType, " {jax.buffer_donor = true}");
  std::string body;
  std::string sum = "%arg0";
  for (int i = 1; i < num_arguments; ++i) {
    absl::StrAppend(&arguments, ", %arg", i, ": ", kType);
    absl::StrAppend(&body, "    %sum", i, " = stablehlo.add ", sum, ", %arg", i,
                    " : ", kType, "\n");
    sum = absl::StrCat("%sum", i);
  }
  const std::string text =
      absl::StrCat("func.func public @main(", arguments, ") -> ", kType, " {\n",
                   body, "    return ", sum, " : ", kType, "\n}\n");

  std::shared_ptr<mlir::MLIRContext> context = MakeMlirContext();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
  ABSL_CHECK(module) << "failed to parse:\n" << text;  // CRASH_OK
  return xla::MaybeOwningMlirModule(std::move(context), std::move(module));
}

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_COMPILATION_TEST_HELPER_H_
