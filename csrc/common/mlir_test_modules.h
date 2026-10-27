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

#ifndef TORCH_TPU_CSRC_COMMON_MLIR_TEST_MODULES_H_
#define TORCH_TPU_CSRC_COMMON_MLIR_TEST_MODULES_H_

#include "xla/pjrt/maybe_owning_mlir_module.h"

namespace torch_tpu {

// Returns a StableHLO module whose main function returns the sum of its
// `num_arguments` f32[128,128] arguments and donates its first argument.
// `num_arguments` must be at least 2.
[[nodiscard]] xla::MaybeOwningMlirModule MakeSumOfArgumentsModule(
    int num_arguments);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_MLIR_TEST_MODULES_H_
