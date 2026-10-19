/*
 * Copyright 2025 Google LLC
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

#include "csrc/ops/copy_from/cpu_to_tpu.h"

#include <memory>
#include <optional>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "csrc/common/context_manager.h"
#include "csrc/common/context_states.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/layout_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/device_buffer_utils.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/pjrt/pjrt_transfer.h"
#include "csrc/pjrt/pjrt_utils.h"
#include "xla/layout.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

namespace {
// Copies the value out of the source CPU tensor to a materialized
// DeviceBufferRef.
absl::StatusOr<DeviceBufferRef> CopyCpuToTpuBuffer(const at::Tensor& src,
                                                   bool non_blocking) {
  at::Tensor contiguous_src_for_tpu = src.contiguous();
  ABSL_VLOG(1) << "[AtenCopyFrom] CPU -> TPU: Ensured CPU tensor is "
                  "contiguous with dtype "
               << ToString(src.scalar_type()) << ". Tensor: " << ToString(src);

  TT_ASSIGN_OR_RETURN(TpuLayout tpu_layout,
                      ResolveTpuLayout(contiguous_src_for_tpu));

  if (contiguous_src_for_tpu.numel() == 0) {
    // If there's no data to actually copy, then we can just create a zero-sized
    // buffer directly and avoid blocking on host/device synchronization.
    return CreateZeroSizeDeviceBufferRef(std::move(tpu_layout.sizes),
                                         tpu_layout.element_type);
  }

  // NOTE: we are intentionally using Tensor::data_ptr() here,
  // and **not** Tensor::storage().data_ptr().get().
  // On CPU, Tensor::data_ptr() will account for the storage offset, while
  // Tensor::storage().data_ptr().get() will not.
  return TpuMallocAndMemcpyHtoD(contiguous_src_for_tpu.data_ptr(),
                                tpu_layout.element_type, tpu_layout.sizes,
                                non_blocking);
}

// Returns `tensor`'s layout on the TPU if it isn't the default for its shape,
// e.g. the SparseCore layout of an embedding table.
std::optional<xla::Layout> GetNonDefaultLayout(const at::Tensor& tensor) {
  const absl::StatusOr<DeviceBufferRef> buffer = GetBuffer(tensor);
  // Buffers in pinned host memory have host layouts.
  if (!buffer.ok() || buffer->is_pinned_host() ||
      !buffer->shape().layout().has_value() ||
      IsDefaultDeviceLayout(*buffer->shape().layout(), buffer->element_type(),
                            buffer->dimensions())) {
    return std::nullopt;
  }
  const absl::StatusOr<xla::Layout> default_layout =
      GetDefaultDeviceLayout(buffer->element_type(), buffer->dimensions());
  if (!default_layout.ok()) return std::nullopt;
  // The buffer's layout records the dtype's width as its element size. Like
  // LayoutContext, use the default element size instead.
  xla::Layout layout = ToXlaLayout(*buffer->shape().layout());
  layout.set_element_size_in_bits(default_layout->element_size_in_bits());
  return layout;
}
}  // namespace

absl::Status CopyCpuToTpu(const at::Tensor& src, const at::Tensor& dest,
                          bool non_blocking) {
  ABSL_VLOG(1) << "[AtenCopyFrom] CPU -> TPU copy path for "
               << ToString(src, "src");
  at::Tensor src_with_dest_dtype = src;
  if (src.scalar_type() != dest.scalar_type()) {
    ABSL_VLOG(1) << "[AtenCopyFrom] CPU -> TPU: Converting CPU tensor "
                    "dtype from "
                 << ToString(src.scalar_type()) << " to "
                 << ToString(dest.scalar_type());
    // Do the type conversion on the host before transfer.
    src_with_dest_dtype = src.to(dest.scalar_type());
  }

  // The copy gives `dest` a new buffer. Unless a LayoutContext is active,
  // create it in `dest`'s current layout, so that e.g. load_state_dict() keeps
  // custom layouts.
  std::optional<xla::Layout> dest_layout;
  if (!GetContextState<LayoutContextState>().has_value()) {
    dest_layout = GetNonDefaultLayout(dest);
  }
  if (dest_layout.has_value()) {
    PushContextState<LayoutContextState>(
        std::make_shared<const xla::Layout>(*dest_layout));
  }
  absl::Cleanup exit_layout_context = [&dest_layout] {
    if (dest_layout.has_value()) PopContextState<LayoutContextState>();
  };

  TT_ASSIGN_OR_RETURN(DeviceBufferRef tpu_buf,
                      CopyCpuToTpuBuffer(src_with_dest_dtype, non_blocking),
                      _.SetPrepend() << "failed to transfer to 'tpu' device: ");
  TT_RETURN_IF_ERROR(AssignBufferToAtTensor(tpu_buf, dest));
  return absl::OkStatus();
}

}  // namespace torch_tpu
