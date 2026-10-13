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

#ifndef TORCH_TPU_CSRC_MLIR_PASSES_FOLD_VIEW_TRANSPOSES_AND_HALF_SPLIT_ROPE_H_
#define TORCH_TPU_CSRC_MLIR_PASSES_FOLD_VIEW_TRANSPOSES_AND_HALF_SPLIT_ROPE_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace torch_tpu {

// Creates an MLIR pass that:
// 1. Folds TransposeOp(BroadcastInDimOp) into a single BroadcastInDimOp when
//    the composed broadcast dimensions are monotonic.
// 2. Cancels inverse TransposeOp chains across unit-dimension ReshapeOp and
//    SliceOp operations (such as as_strided_inverse followed by consumer view
//    decompositions).
// 3. Commutes minor-dimension half-slices past multi-head attention head-split
//    ReshapeOp + TransposeOp chains and unifies duplicate head-split views.
// 4. Lowers half-split rotary embeddings (rotate_half and _apply_rotary_emb
//    forward and backward) into rank-preserving skew-symmetric MXU DotGeneralOp
//    contractions without narrow half-head slices or pad-maximum concats.
std::unique_ptr<mlir::Pass> createFoldViewTransposesAndHalfSplitRopePass();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_MLIR_PASSES_FOLD_VIEW_TRANSPOSES_AND_HALF_SPLIT_ROPE_H_
