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

#ifndef TORCH_TPU_CSRC_MLIR_PASSES_PARTITION_CONCAT_SOFTMAX_AND_DEINTERLEAVE_H_
#define TORCH_TPU_CSRC_MLIR_PASSES_PARTITION_CONCAT_SOFTMAX_AND_DEINTERLEAVE_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace torch_tpu {

// Creates an MLIR pass that:
// 1. Folds zero-sum shift-invariant pre-subtraction backward scatters on
//    softmax gradients and folds 2-result max+argmax pre-subtractions.
// 2. Partitions reciprocal-multiply forward softmax and backward
//    softmax_backward_data across small-tail concatenations (e.g. attention
//    sinks) without materializing unaligned concatenated tensors.
// 3. Lowers minor-dim-2 even/odd strided slice deinterleaving and backward
//    interleaving into 128-lane tile-aligned 2D MXU contractions.
std::unique_ptr<mlir::Pass> createPartitionConcatSoftmaxAndDeinterleavePass();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_MLIR_PASSES_PARTITION_CONCAT_SOFTMAX_AND_DEINTERLEAVE_H_
