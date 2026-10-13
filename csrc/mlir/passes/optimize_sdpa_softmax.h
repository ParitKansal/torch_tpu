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

#ifndef TORCH_TPU_CSRC_MLIR_PASSES_OPTIMIZE_SDPA_SOFTMAX_H_
#define TORCH_TPU_CSRC_MLIR_PASSES_OPTIMIZE_SDPA_SOFTMAX_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace torch_tpu {

// Creates an MLIR pass that optimizes SDPA forward and backward softmax graphs:
// 1. Rewrites backward ComputeGradP rowsum subtraction and non-Flash max-logit
//    subtraction to negate on the 3D reduced tensor before broadcast_in_dim so
//    XLA PrefuseReduceBroadcastReuse does not emit standalone O(L^2)
//    reduce-window output fusions.
// 2. Hoists 3D reciprocal 1/safe_sum_exp (and 3D all-masked row selection)
//    outside the 4D softmax multiplication in backward and unaligned forward
//    SDPA.
// 3. Decouples backward f32 shifted_logits and local sum_exp from forward in
//    monolithic SingleTraceTrainer graphs when XLA online-softmax applies.
// 4. Moves all-masked row Select after DotGeneral(softmax, V) and normalizes 5D
//    GQA V contraction to 4D when XLA online-softmax applies.
std::unique_ptr<mlir::Pass> createOptimizeSdpaSoftmaxPass();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_MLIR_PASSES_OPTIMIZE_SDPA_SOFTMAX_H_
