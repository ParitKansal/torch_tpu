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

#ifndef TORCH_TPU_CSRC_EAGER_DEVICE_ASSIGNMENT_PIPELINE_H_
#define TORCH_TPU_CSRC_EAGER_DEVICE_ASSIGNMENT_PIPELINE_H_

#include <cstdint>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/ops/op_names.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"

namespace torch_tpu {

struct OpDeviceAssignment {
  OpName op_name;
  absl::Span<const int64_t> device_assignment;
};

struct PallasDeviceAssignment {
  absl::Span<const int64_t> device_assignment;
};

// Resolves a single target device assignment across `ops` and `pallas_ops`.
// If `pallas_ops` is non-empty, its device assignment serves as the
// authoritative anchor; otherwise the first non-empty op device assignment
// does. Ops whose device assignments are permutations of the same device set
// resolve to that one assignment, and their collective attributes are remapped
// later (see `RemapReplicaOrderInOperation`). Ops on different device sets
// return kInvalidArgument: collectives over process groups with different
// members cannot share an executable. Conflicting Pallas mesh device
// assignments throw.
absl::StatusOr<std::vector<int64_t>>  // INT_VEC_OK
ResolveDeviceAssignment(
    absl::Span<const OpDeviceAssignment> ops,
    absl::Span<const PallasDeviceAssignment> pallas_ops = {});

// Resolves a single target device assignment across all deferred operations in
// `execution_order` by delegating to `ResolveDeviceAssignment`.
absl::StatusOr<std::vector<int64_t>>  // INT_VEC_OK
ResolveDeviceAssignmentForTraversal(
    absl::Span<const SharedDeviceBufferList> execution_order,
    absl::Span<const int64_t> pallas_device_assignment = {});

// Remaps the rank-local collective attributes (`replica_groups`,
// `source_target_pairs`, and CustomCall `backend_config` replica groups) of
// `root_op` and its nested ops when `op_da` is a permutation of `resolved_da`:
// rank r of the op becomes the partition of `resolved_da` holding device
// `op_da[r]`. A no-op if `op_da` is empty or equals `resolved_da`. Returns an
// error if `op_da` is not a permutation of `resolved_da`.
absl::Status RemapReplicaOrderInOperation(
    mlir::Operation* absl_nonnull root_op, absl::Span<const int64_t> op_da,
    absl::Span<const int64_t> resolved_da);

// Applies `RemapReplicaOrderInOperation` to all MLIR operations emitted for
// `deferred_op` (starting at `first_new_op` in the current block, plus any
// newly added functions in `new_func_ops`) when the op's device assignment is
// a permutation of `resolved_da`. Returns an error if any remap fails.
absl::Status ApplyReplicaOrderRemapping(
    mlir::Operation* absl_nullable first_new_op,
    absl::Span<const mlir::func::FuncOp> new_func_ops,
    const DeferredOp& deferred_op, absl::Span<const int64_t> resolved_da);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EAGER_DEVICE_ASSIGNMENT_PIPELINE_H_
