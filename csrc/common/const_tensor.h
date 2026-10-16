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

#ifndef TORCH_TPU_CSRC_COMMON_CONST_TENSOR_H_
#define TORCH_TPU_CSRC_COMMON_CONST_TENSOR_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/Device.h"
#include "c10/core/DispatchKeySet.h"
#include "c10/core/SymInt.h"
#include "c10/core/SymIntArrayRef.h"
#include "c10/core/TensorImpl.h"
#include "c10/core/TensorOptions.h"
#include "c10/util/typeid.h"
#include "torch/headeronly/core/Layout.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

// A read-only handle to an at::Tensor that only exposes accessors which are
// guaranteed not to change the state of the tensor.
//
// at::Tensor's `const` methods give no such guarantee for the FakeTensors with
// symbolic sizes/strides that Dynamo traces with under torch.compile dynamic
// shapes: e.g. sizes()/strides() guard_int() every symbol, which silently
// specializes the dynamic dimensions, while numel()/storage_offset() throw.
// None of the accessors below guard on (and thereby specialize) a symbolic
// size or stride, none throw because a shape is symbolic, and none mutate the
// TensorImpl. Use ConstTensor when a function only inspects a tensor, e.g. for
// logging or debug printing. Note that comparing the returned SymInts with
// `==` (or other comparison operators) guards; print them instead.
//
// None of the accessors throw either: they return std::optional when there may
// be no value, and absl::StatusOr when the query may be invalid for the tensor.
//
// Caveat: tensor subclasses with a Python-custom sizes/strides/device policy
// forward some of these accessors to their `__torch_dispatch__`, which is
// guard-free by PyTorch convention but can run arbitrary Python code.
//
// ConstTensor holds a reference to the TensorImpl, so it is safe to store or
// capture. Converting an at::Tensor to a ConstTensor is implicit because it is
// always safe; converting back requires an explicit ToTensor() call.
class ConstTensor {
 public:
  // NOLINTNEXTLINE(google-explicit-constructor): Converting is always safe.
  ConstTensor(at::Tensor tensor) : tensor_(std::move(tensor)) {}

  // Cheap to copy and move: only the TensorImpl refcount changes.
  ConstTensor(const ConstTensor&) = default;
  ConstTensor& operator=(const ConstTensor&) = default;
  ConstTensor(ConstTensor&&) noexcept = default;
  ConstTensor& operator=(ConstTensor&&) noexcept = default;

  // Returns the wrapped tensor.
  //
  // WARNING: at::Tensor's `const` methods may specialize symbolic shapes or
  // throw on them (see the class comment). Prefer the accessors below.
  [[nodiscard]] at::Tensor ToTensor() const { return tensor_; }

  // Identity and metadata.
  [[nodiscard]] bool defined() const { return tensor_.defined(); }
  [[nodiscard]] bool is_same(const ConstTensor& other) const {
    return tensor_.is_same(other.tensor_);
  }
  [[nodiscard]] bool is_same(const at::Tensor& other) const {
    return tensor_.is_same(other);
  }
  [[nodiscard]] c10::ScalarType scalar_type() const {
    return tensor_.scalar_type();
  }
  [[nodiscard]] caffe2::TypeMeta dtype() const { return tensor_.dtype(); }
  // Returns nullopt for tensors without a device, e.g. undefined tensors.
  [[nodiscard]] std::optional<c10::Device> device() const;
  [[nodiscard]] std::optional<c10::DeviceIndex> get_device() const;
  [[nodiscard]] c10::Layout layout() const { return tensor_.layout(); }
  [[nodiscard]] c10::TensorOptions options() const;
  [[nodiscard]] c10::DispatchKeySet key_set() const {
    return tensor_.key_set();
  }
  // Return nullopt for tensors without a dtype, e.g. undefined tensors.
  [[nodiscard]] std::optional<size_t> itemsize() const;
  [[nodiscard]] std::optional<int64_t> element_size() const;
  [[nodiscard]] bool is_floating_point() const {
    return tensor_.is_floating_point();
  }
  [[nodiscard]] bool is_complex() const { return tensor_.is_complex(); }
  // Returns nullopt for dtypes without a signedness: quantized, bits and
  // undefined dtypes.
  [[nodiscard]] std::optional<bool> is_signed() const;

  // Device predicates.
  [[nodiscard]] bool is_cpu() const { return tensor_.is_cpu(); }
  [[nodiscard]] bool is_cuda() const { return tensor_.is_cuda(); }
  [[nodiscard]] bool is_xpu() const { return tensor_.is_xpu(); }
  [[nodiscard]] bool is_ipu() const { return tensor_.is_ipu(); }
  [[nodiscard]] bool is_xla() const { return tensor_.is_xla(); }
  [[nodiscard]] bool is_mtia() const { return tensor_.is_mtia(); }
  [[nodiscard]] bool is_hpu() const { return tensor_.is_hpu(); }
  [[nodiscard]] bool is_lazy() const { return tensor_.is_lazy(); }
  [[nodiscard]] bool is_hip() const { return tensor_.is_hip(); }
  [[nodiscard]] bool is_ve() const { return tensor_.is_ve(); }
  [[nodiscard]] bool is_mps() const { return tensor_.is_mps(); }
  [[nodiscard]] bool is_maia() const { return tensor_.is_maia(); }
  [[nodiscard]] bool is_vulkan() const { return tensor_.is_vulkan(); }
  [[nodiscard]] bool is_metal() const { return tensor_.is_metal(); }
  [[nodiscard]] bool is_meta() const { return tensor_.is_meta(); }
  [[nodiscard]] bool is_privateuseone() const {
    return tensor_.is_privateuseone();
  }

  // Layout and dispatch key predicates.
  [[nodiscard]] bool is_sparse() const { return tensor_.is_sparse(); }
  [[nodiscard]] bool is_sparse_csr() const { return tensor_.is_sparse_csr(); }
  // CSR, CSC, BSR or BSC.
  [[nodiscard]] bool is_sparse_compressed() const {
    return tensor_.unsafeGetTensorImpl()->is_sparse_compressed();
  }
  [[nodiscard]] bool is_mkldnn() const { return tensor_.is_mkldnn(); }
  [[nodiscard]] bool is_nested() const { return tensor_.is_nested(); }
  [[nodiscard]] bool is_quantized() const { return tensor_.is_quantized(); }
  [[nodiscard]] bool is_inference() const { return tensor_.is_inference(); }
  [[nodiscard]] bool is_conj() const { return tensor_.is_conj(); }
  [[nodiscard]] bool is_neg() const { return tensor_.is_neg(); }
  [[nodiscard]] bool is_zerotensor() const { return tensor_._is_zerotensor(); }
  [[nodiscard]] bool is_wrapped_number() const {
    return tensor_.unsafeGetTensorImpl()->is_wrapped_number();
  }
  [[nodiscard]] bool is_python_dispatch() const {
    return tensor_.unsafeGetTensorImpl()->is_python_dispatch();
  }
  [[nodiscard]] bool has_storage() const { return tensor_.has_storage(); }

  // Autograd. These read fields without creating autograd metadata. Undefined
  // tensors report the defaults of a tensor that never required grad.
  [[nodiscard]] bool requires_grad() const { return tensor_.requires_grad(); }
  [[nodiscard]] bool is_leaf() const { return !defined() || tensor_.is_leaf(); }
  [[nodiscard]] int64_t output_nr() const {
    return defined() ? tensor_.output_nr() : 0;
  }
  [[nodiscard]] bool retains_grad() const {
    return defined() && tensor_.retains_grad();
  }
  [[nodiscard]] std::optional<c10::ScalarType> grad_dtype() const {
    return defined() ? tensor_.grad_dtype() : scalar_type();
  }
  // Unlike at::Tensor::grad(), this does not warn for non-leaf tensors.
  [[nodiscard]] ConstTensor grad() const {
    return tensor_.unsafeGetTensorImpl()->grad();
  }
  [[nodiscard]] bool is_view() const { return defined() && tensor_.is_view(); }
  // Returns nullopt if the tensor is not a view.
  [[nodiscard]] std::optional<ConstTensor> base() const;

  // Shape. These return symbolic values as is, without specializing them.
  [[nodiscard]] bool has_symbolic_sizes_strides() const {
    return tensor_.unsafeGetTensorImpl()->has_symbolic_sizes_strides();
  }
  // Fails for undefined tensors.
  absl::StatusOr<int64_t> dim() const;
  // Fail for undefined and nested tensors.
  absl::StatusOr<c10::SymIntArrayRef> sym_sizes() const;
  // Also fails with OutOfRange if `dim` is not in [-dim(), dim()).
  absl::StatusOr<c10::SymInt> sym_size(int64_t dim) const;
  // Fail for undefined, nested and sparse compressed tensors. Sparse COO
  // tensors have all-zero strides.
  absl::StatusOr<c10::SymIntArrayRef> sym_strides() const;
  // Also fails with OutOfRange if `dim` is not in [-dim(), dim()).
  absl::StatusOr<c10::SymInt> sym_stride(int64_t dim) const;
  [[nodiscard]] c10::SymInt sym_numel() const { return tensor_.sym_numel(); }
  // Fails for undefined and sparse COO tensors.
  absl::StatusOr<c10::SymInt> sym_nbytes() const;
  [[nodiscard]] c10::SymInt sym_storage_offset() const {
    return tensor_.sym_storage_offset();
  }

  // Returns whether the tensor is contiguous in `memory_format`, or nullopt if
  // the tensor is undefined, has symbolic sizes/strides, is nested or has a
  // non-strided layout.
  // at::Tensor::sym_is_contiguous() is deliberately not exposed: computing the
  // contiguity of backed symbolic shapes evaluates (and guards on) size and
  // stride comparisons.
  [[nodiscard]] std::optional<bool> maybe_is_contiguous(
      c10::MemoryFormat memory_format = c10::MemoryFormat::Contiguous) const;

  // Sparse. sparse_dim() and dense_dim() are 0 and dim() for strided tensors.
  absl::StatusOr<int64_t> sparse_dim() const;
  absl::StatusOr<int64_t> dense_dim() const;
  // Fails for tensors that are not sparse COO.
  absl::StatusOr<bool> is_coalesced() const;
  // Returns the number of specified elements of a sparse COO or sparse
  // compressed tensor. Unlike at::Tensor::_nnz(), this does not guard on a
  // symbolic number of elements. Fails for tensors that are not sparse.
  absl::StatusOr<c10::SymInt> sym_nnz() const;

 private:
  absl::Status ValidateDefined(std::string_view method) const;

  at::Tensor tensor_;  // UNINITIALIZED_TENSOR_OK: Set by the constructor.
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_CONST_TENSOR_H_
