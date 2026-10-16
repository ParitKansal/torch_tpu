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

#include "csrc/common/const_tensor.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

#include "ATen/SparseCsrTensorImpl.h"
#include "ATen/SparseTensorImpl.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/Device.h"
#include "c10/core/ScalarType.h"
#include "c10/core/SymInt.h"
#include "c10/core/SymIntArrayRef.h"
#include "c10/core/TensorImpl.h"
#include "c10/core/TensorOptions.h"
#include "csrc/common/error_utils.h"
#include "torch/headeronly/core/Layout.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

static absl::StatusOr<int64_t> WrapDim(std::string_view method, int64_t dim,
                                       int64_t ndim) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=no Python caller yet
      dim >= -ndim && dim < ndim, error::kIndexError)
      << "expected a dimension in [" << -ndim << ", " << ndim << ") for "
      << method << "(), got " << dim;
  return dim < 0 ? dim + ndim : dim;
}

absl::Status ConstTensor::ValidateDefined(std::string_view method) const {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=no Python caller yet
      defined(), error::kInvalidArgument)
      << "tensor for " << method << "() should be defined";
  return absl::OkStatus();
}

std::optional<c10::Device> ConstTensor::device() const {
  // Undefined tensors are the only ones without a device in practice.
  if (!defined()) {
    return std::nullopt;
  }
  return tensor_.device();
}

std::optional<c10::DeviceIndex> ConstTensor::get_device() const {
  const std::optional<c10::Device> device = this->device();
  if (!device.has_value()) {
    return std::nullopt;
  }
  return device->index();
}

c10::TensorOptions ConstTensor::options() const {
  const c10::TensorOptions options =
      c10::TensorOptions().dtype(dtype()).layout(layout());
  const std::optional<c10::Device> device = this->device();
  return device.has_value() ? options.device(*device) : options;
}

std::optional<size_t> ConstTensor::itemsize() const {
  if (!tensor_.unsafeGetTensorImpl()->dtype_initialized()) {
    return std::nullopt;
  }
  return dtype().itemsize();
}

std::optional<int64_t> ConstTensor::element_size() const {
  const std::optional<size_t> size = itemsize();
  if (!size.has_value()) {
    return std::nullopt;
  }
  return static_cast<int64_t>(*size);
}

std::optional<bool> ConstTensor::is_signed() const {
  const c10::ScalarType type = scalar_type();
  if (c10::isQIntType(type) || c10::isBitsType(type) ||
      type == c10::ScalarType::Undefined) {
    return std::nullopt;
  }
  return c10::isSignedType(type);  // MLIR_SIGNED_INT_OK=c10 dtype, not MLIR
}

std::optional<ConstTensor> ConstTensor::base() const {
  if (!is_view()) {
    return std::nullopt;
  }
  return ConstTensor(at::Tensor(tensor_._base()));
}

absl::StatusOr<int64_t> ConstTensor::dim() const {
  TT_RETURN_IF_ERROR(ValidateDefined("dim"));
  return tensor_.dim();
}

absl::StatusOr<c10::SymIntArrayRef> ConstTensor::sym_sizes() const {
  TT_RETURN_IF_ERROR(ValidateDefined("sym_sizes"));
  if (is_nested()) {
    return TT_ERROR(  // ERROR_COV_INFEASIBLE=no Python caller yet
               error::kInvalidArgument)
           << "tensor for sym_sizes() should not be nested";
  }
  return tensor_.sym_sizes();
}

absl::StatusOr<c10::SymInt> ConstTensor::sym_size(int64_t dim) const {
  TT_ASSIGN_OR_RETURN(const c10::SymIntArrayRef sizes, sym_sizes());
  TT_ASSIGN_OR_RETURN(
      const int64_t wrapped,
      WrapDim("sym_size", dim, static_cast<int64_t>(sizes.size())));
  return sizes[wrapped];
}

absl::StatusOr<c10::SymIntArrayRef> ConstTensor::sym_strides() const {
  TT_RETURN_IF_ERROR(ValidateDefined("sym_strides"));
  if (is_nested()) {
    return TT_ERROR(  // ERROR_COV_INFEASIBLE=no Python caller yet
               error::kInvalidArgument)
           << "tensor for sym_strides() should not be nested";
  }
  if (is_sparse_compressed()) {
    return TT_ERROR(  // ERROR_COV_INFEASIBLE=no Python caller yet
               error::kInvalidArgument)
           << "expected a non-sparse-compressed tensor for sym_strides(), "
              "got a "
           << layout() << " tensor";
  }
  return tensor_.sym_strides();
}

absl::StatusOr<c10::SymInt> ConstTensor::sym_stride(int64_t dim) const {
  TT_ASSIGN_OR_RETURN(const c10::SymIntArrayRef strides, sym_strides());
  TT_ASSIGN_OR_RETURN(
      const int64_t wrapped,
      WrapDim("sym_stride", dim, static_cast<int64_t>(strides.size())));
  return strides[wrapped];
}

absl::StatusOr<c10::SymInt> ConstTensor::sym_nbytes() const {
  TT_RETURN_IF_ERROR(ValidateDefined("sym_nbytes"));
  if (layout() == c10::kSparse) {
    return TT_ERROR(  // ERROR_COV_INFEASIBLE=no Python caller yet
               error::kInvalidArgument)
           << "tensor for sym_nbytes() should not be sparse COO";
  }
  return sym_numel() * c10::SymInt(static_cast<int64_t>(dtype().itemsize()));
}

std::optional<bool> ConstTensor::maybe_is_contiguous(
    c10::MemoryFormat memory_format) const {
  if (!defined() || has_symbolic_sizes_strides() || is_nested() ||
      layout() != c10::kStrided) {
    return std::nullopt;
  }
  return tensor_.is_contiguous(memory_format);
}

// The sparse accessors use dynamic_cast because wrapper subclasses can have the
// sparse dispatch keys without a sparse TensorImpl.
absl::StatusOr<int64_t> ConstTensor::sparse_dim() const {
  TT_RETURN_IF_ERROR(ValidateDefined("sparse_dim"));
  const c10::TensorImpl* impl = tensor_.unsafeGetTensorImpl();
  if (const auto* coo = dynamic_cast<const at::SparseTensorImpl*>(impl)) {
    return coo->sparse_dim();
  }
  if (const auto* csr = dynamic_cast<const at::SparseCsrTensorImpl*>(impl)) {
    return csr->sparse_dim();
  }
  if (layout() == c10::kStrided) {
    return 0;
  }
  return TT_ERROR(  // ERROR_COV_INFEASIBLE=no Python caller yet
             error::kInvalidArgument)
         << "expected a strided or sparse tensor for sparse_dim(), got a "
         << layout() << " tensor";
}

absl::StatusOr<int64_t> ConstTensor::dense_dim() const {
  TT_RETURN_IF_ERROR(ValidateDefined("dense_dim"));
  const c10::TensorImpl* impl = tensor_.unsafeGetTensorImpl();
  if (const auto* coo = dynamic_cast<const at::SparseTensorImpl*>(impl)) {
    return coo->dense_dim();
  }
  if (const auto* csr = dynamic_cast<const at::SparseCsrTensorImpl*>(impl)) {
    return csr->dense_dim();
  }
  if (layout() == c10::kStrided) {
    return tensor_.dim();
  }
  return TT_ERROR(  // ERROR_COV_INFEASIBLE=no Python caller yet
             error::kInvalidArgument)
         << "expected a strided or sparse tensor for dense_dim(), got a "
         << layout() << " tensor";
}

absl::StatusOr<bool> ConstTensor::is_coalesced() const {
  TT_RETURN_IF_ERROR(ValidateDefined("is_coalesced"));
  if (const auto* coo = dynamic_cast<const at::SparseTensorImpl*>(
          tensor_.unsafeGetTensorImpl())) {
    return coo->coalesced();
  }
  return TT_ERROR(  // ERROR_COV_INFEASIBLE=no Python caller yet
             error::kInvalidArgument)
         << "expected a sparse COO tensor for is_coalesced(), got a "
         << layout() << " tensor";
}

absl::StatusOr<c10::SymInt> ConstTensor::sym_nnz() const {
  TT_RETURN_IF_ERROR(ValidateDefined("sym_nnz"));
  const c10::TensorImpl* impl = tensor_.unsafeGetTensorImpl();
  if (const auto* coo = dynamic_cast<const at::SparseTensorImpl*>(impl)) {
    return coo->sym_nnz();
  }
  if (const auto* csr = dynamic_cast<const at::SparseCsrTensorImpl*>(impl)) {
    return csr->plain_indices().sym_size(-1);
  }
  return TT_ERROR(  // ERROR_COV_INFEASIBLE=no Python caller yet
             error::kInvalidArgument)
         << "expected a sparse tensor for sym_nnz(), got a " << layout()
         << " tensor";
}

}  // namespace torch_tpu
