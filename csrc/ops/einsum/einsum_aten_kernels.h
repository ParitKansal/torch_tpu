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

#ifndef TORCH_TPU_CSRC_OPS_EINSUM_EINSUM_ATEN_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_EINSUM_EINSUM_ATEN_KERNELS_H_

#include "ATen/core/ATen_fwd.h"
#include "c10/util/string_view.h"

namespace torch_tpu {

// Implements `aten::einsum` (`einsum.default`) on `AutogradPrivateUse1`.
// Decomposes Python wrapper tensor subclasses (such as `DTensor`), functorch
// transforms, and forward AD via `at::native::einsum` while dispatching
// standard TPU tensors and `FakeTensor`s (`torch.compile` / `torch.export`)
// through `EinsumBackwardNode` and `AtenEinsum`.
at::Tensor AtenEinsumAutograd(c10::string_view equation, at::TensorList tensors,
                              at::OptionalIntArrayRef path);

// Implements `aten::einsum` (`einsum.default`) on `PrivateUse1`, lowering
// directly to `stablehlo.dot_general` without intermediate 3D `bmm` reshapes.
at::Tensor AtenEinsum(c10::string_view equation, at::TensorList tensors,
                      at::OptionalIntArrayRef path);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_EINSUM_EINSUM_ATEN_KERNELS_H_
