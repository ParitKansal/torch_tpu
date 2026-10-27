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

#include "csrc/common/mlir_test_modules.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/strings/str_cat.h"
#include "csrc/common/compilation.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"

namespace torch_tpu {

xla::MaybeOwningMlirModule MakeSumOfArgumentsModule(const int num_arguments) {
  ABSL_CHECK_GE(num_arguments, 2);  // CRASH_OK
  constexpr std::string_view kType = "tensor<128x128xf32>";
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
