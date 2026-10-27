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

#ifndef TORCH_TPU_CSRC_OPS_SCALED_DOT_PRODUCT_ATTENTION_FLASH_ATTENTION_KERNEL_H_
#define TORCH_TPU_CSRC_OPS_SCALED_DOT_PRODUCT_ATTENTION_FLASH_ATTENTION_KERNEL_H_

#include "absl/status/statusor.h"
#include "csrc/internal/mosaic/kiln/kiln.h"
#include "csrc/ops/scaled_dot_product_attention/flash_attention_config.h"

namespace torch_tpu {

// Creates the flash attention forward kernel.
absl::StatusOr<kiln::Kernel> CreateKernel(const FlashAttnConfig& config,
                                          const Tiling& tiling);

// Creates the flash attention backward DKV kernel.
absl::StatusOr<kiln::Kernel> CreateBackwardDkvKernel(
    const FlashAttnConfig& config, const Tiling& tiling);

// Creates the flash attention backward DQ kernel.
absl::StatusOr<kiln::Kernel> CreateBackwardDqKernel(
    const FlashAttnConfig& config, const Tiling& tiling);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_SCALED_DOT_PRODUCT_ATTENTION_FLASH_ATTENTION_KERNEL_H_
