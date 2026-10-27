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

#ifndef TORCH_TPU_CSRC_PJRT_PJRT_TRANSFER_H_
#define TORCH_TPU_CSRC_PJRT_PJRT_TRANSFER_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/eager/device_buffer.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "xla/future.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

// Allocates a TPU buffer of the given dimensions and element type, and copies
// the data from the host buffer into it.
// This is always non-blocking; the returned DeviceBufferRef can be used to
// track the transfer's completion.
// If backing_data is provided, it is used to keep the host data alive until the
// transfer completes. If backing_data is not provided, the host data must
// remain valid and unchanged until the transfer completes.
absl::StatusOr<DeviceBufferRef> TpuMallocAndMemcpyHtoD(
    const void* host_data, mlir::ElementType element_type,
    absl::Span<const int64_t> dimensions,
    std::vector<uint8_t> backing_data = {});

// Copies the data from the device directly into the provided host buffer.
// This is always non-blocking; the future may be awaited to make this blocking.
xla::Future<void> TpuMemcpyDtoHDirect(const DeviceBufferRef& buffer_ref,
                                      void* dst_ptr);

// Holds the completion future and RAII buffer hold for an asynchronous DMA
// copy.
struct AsyncDmaResult {
  xla::Future<> future;
  std::shared_ptr<void> buffer_hold;
};

// Performs asynchronous non-blocking Device-to-Host (D2H) DMA transfer
// from a TPU device buffer to a host memory pointer using the PjRt RawBuffer C
// API. Returns an AsyncDmaResult indicating copy completion along with an
// opaque hold object preserving device buffer lifetime until completion.
absl::StatusOr<AsyncDmaResult> TpuAsyncDmaCopyDtoH(
    const DeviceBufferRef& src_buffer, void* dst_host_ptr, int64_t copy_bytes);

// Performs asynchronous non-blocking Host-to-Device (H2D) DMA transfer
// from a host memory pointer into a TPU device buffer at a byte offset using
// the PjRt RawBuffer C API. Returns an AsyncDmaResult indicating copy
// completion along with an opaque hold object preserving device buffer lifetime
// until completion.
absl::StatusOr<AsyncDmaResult> TpuAsyncDmaCopyHtoD(
    const void* src_host_ptr, const DeviceBufferRef& dst_buffer,
    int64_t dst_byte_offset, int64_t copy_bytes);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_PJRT_PJRT_TRANSFER_H_
