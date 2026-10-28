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

#include "csrc/internal/mosaic/kiln/lib/pipeline.h"

#include <cstddef>
#include <cstdint>
#include <limits>

#include "absl/functional/function_ref.h"
#include "absl/log/absl_check.h"
#include "csrc/internal/mosaic/kiln/core/control_flow.h"
#include "csrc/internal/mosaic/kiln/core/dma.h"
#include "csrc/internal/mosaic/kiln/core/dtype.h"
#include "csrc/internal/mosaic/kiln/core/index_arg.h"
#include "csrc/internal/mosaic/kiln/core/memory.h"
#include "csrc/internal/mosaic/kiln/core/ref.h"
#include "csrc/internal/mosaic/kiln/core/scalar.h"
#include "csrc/internal/mosaic/kiln/core/semaphore.h"
#include "csrc/internal/mosaic/kiln/core/shape.h"
#include "csrc/internal/mosaic/kiln/core/specs.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace torch_tpu::kiln {
namespace {

// Pipelined buffers have two slots: one being computed on, one being copied.
constexpr int64_t kSlots = 2;

// The VMEM slots of a pipelined HBM operand, with one DMA semaphore per slot.
struct Slots {
  Ref vmem;
  SemaphoreArray sems;
};

// Allocates the VMEM slots for blocks of `hbm` (a `{2, block_shape...}`
// buffer), plus one DMA semaphore per slot.
Slots AllocaSlots(const Ref& hbm, const Dimensions& block_shape) {
  ABSL_CHECK(hbm.memory_space() == MemorySpace::kHbm)  // CRASH_OK
      << "pipelined operands must live in HBM";
  ABSL_CHECK_GE(hbm.rank(), 1)  // CRASH_OK
      << "pipelined HBM operands must be at least 1D";
  ABSL_CHECK_EQ(  // CRASH_OK
      hbm.rank(), static_cast<int64_t>(block_shape.size()))
      << "the block shape must have the rank of its HBM operand";
  Dimensions dims = {kSlots};
  dims.append(block_shape.begin(), block_shape.end());
  if (block_shape.size() == 1) {
    // A `{2, B}` buffer would be tiled two rows at a time, so a single slot
    // (one row) can't be sliced out. Put the slots side by side in a flat
    // `{2 * B}` buffer instead, as Pallas `BufferedRef` does.
    ABSL_CHECK_LE(  // CRASH_OK
        kSlots * block_shape[0], std::numeric_limits<int32_t>::max())
        << "1D pipelined buffer exceeds int32_t max";
    dims = {kSlots * block_shape[0]};
  }
  return Slots{.vmem = AllocaVmem(Shape(dims, hbm.dtype())),
               .sems = AllocaDmaSemaphore({kSlots})};
}

// The `region` of `hbm`.
Ref HbmWindow(const Ref& hbm, const BlockRegion& region) {
  ABSL_CHECK(region.offsets.size() == region.sizes.size() &&  // CRASH_OK
             static_cast<int64_t>(region.sizes.size()) == hbm.rank())
      << "a block region needs one offset and one extent per dimension";
  llvm::SmallVector<Indexer, 4> indexers;
  for (size_t d = 0; d < region.sizes.size(); ++d) {
    indexers.push_back(Slice(region.offsets[d], region.sizes[d]));
  }
  return hbm.At(indexers);
}

// The leading `sizes` part of slot `slot` of `vmem` (the whole slot when
// `sizes` is the block shape).
Ref SlotWindow(const Ref& vmem, Scalar slot, const Dimensions& block_shape,
               llvm::ArrayRef<IndexArg> sizes) {
  if (block_shape.size() == 1) {
    const int32_t extent = static_cast<int32_t>(block_shape[0]);
    return vmem(Slice(slot * extent, sizes[0]));
  }
  llvm::SmallVector<Indexer, 4> indexers = {slot};
  for (const IndexArg& size : sizes) {
    indexers.push_back(Slice(0, size));
  }
  return vmem.At(indexers);
}

Ref Slot(const Ref& vmem, Scalar slot, const Dimensions& block_shape) {
  llvm::SmallVector<IndexArg, 4> sizes(block_shape.begin(), block_shape.end());
  return SlotWindow(vmem, slot, block_shape, sizes);
}

// Whether a pipelined operand is copied into VMEM (an input) or out to HBM.
enum class Direction { kToVmem, kToHbm };

// Allocates the slots of each of `operands`.
llvm::SmallVector<Slots, 4> AllocaSlots(
    llvm::ArrayRef<PipelineOperand> operands) {
  llvm::SmallVector<Slots, 4> slots;
  for (const PipelineOperand& operand : operands) {
    ABSL_CHECK(operand.region != nullptr)  // CRASH_OK
        << "PipelineOperand::region must be set";
    slots.push_back(AllocaSlots(operand.hbm, operand.block_shape));
  }
  return slots;
}

// The copy of block `step` of `operand` between HBM and its slot.
AsyncCopyDescriptor Copy(const PipelineOperand& operand, const Slots& slots,
                         Scalar step, Direction direction) {
  const BlockRegion region = operand.region(step);
  const Scalar slot = step % kSlots;
  Ref hbm = HbmWindow(operand.hbm, region);
  Ref vmem = SlotWindow(slots.vmem, slot, operand.block_shape, region.sizes);
  return direction == Direction::kToVmem
             ? MakeAsyncCopy(hbm, vmem, slots.sems(slot))
             : MakeAsyncCopy(vmem, hbm, slots.sems(slot));
}

// Starts the copies of block `step` of each of `operands`.
void StartCopies(llvm::ArrayRef<PipelineOperand> operands,
                 llvm::ArrayRef<Slots> slots, Scalar step,
                 Direction direction) {
  for (size_t i = 0; i < operands.size(); ++i) {
    Copy(operands[i], slots[i], step, direction).Start();
  }
}

// Waits for the copies of block `step` of each of `operands`.
void WaitCopies(llvm::ArrayRef<PipelineOperand> operands,
                llvm::ArrayRef<Slots> slots, Scalar step, Direction direction) {
  for (size_t i = 0; i < operands.size(); ++i) {
    Copy(operands[i], slots[i], step, direction).Wait();
  }
}

// The slots of block `step` of each of `operands`.
llvm::SmallVector<Ref, 4> SlotRefs(llvm::ArrayRef<PipelineOperand> operands,
                                   llvm::ArrayRef<Slots> slots, Scalar step) {
  llvm::SmallVector<Ref, 4> refs;
  for (size_t i = 0; i < operands.size(); ++i) {
    refs.push_back(Slot(slots[i].vmem, step % kSlots, operands[i].block_shape));
  }
  return refs;
}

}  // namespace

// TODO(b/576608541): Follow-ups:
// - consecutive steps accumulating into one output block (e.g. the K steps
//   of a matmul);
// - Pallas-style block indices (`index_map`), with offsets and tile-rounded
//   edge sizes derived from the HBM shape;
// - a configurable number of slots (`pl.Buffered`);
// - skipping the copy of a block that is unchanged from the previous step.
void EmitPipeline(IndexArg begin_arg, IndexArg end_arg,
                  llvm::ArrayRef<PipelineOperand> inputs,
                  llvm::ArrayRef<PipelineOperand> outputs,
                  absl::FunctionRef<void(const PipelineStep&)> body) {
  const Scalar begin = begin_arg.ToScalar();
  const Scalar end = end_arg.ToScalar();
  ABSL_CHECK(begin.dtype() == i32 && end.dtype() == i32)  // CRASH_OK
      << "pipeline bounds must be i32 Scalars";

  RunScoped([&] {  // REF_CAPTURE_OK=called inline
    const llvm::SmallVector<Slots, 4> input_slots = AllocaSlots(inputs);
    const llvm::SmallVector<Slots, 4> output_slots = AllocaSlots(outputs);
    If(begin < end, [&] {  // REF_CAPTURE_OK=called inline
      StartCopies(inputs, input_slots, begin, Direction::kToVmem);
      For(begin, end, [&](Scalar step) {  // REF_CAPTURE_OK=called inline
        If(step + 1 < end, [&] {          // REF_CAPTURE_OK=called inline
          StartCopies(inputs, input_slots, step + 1, Direction::kToVmem);
        });
        WaitCopies(inputs, input_slots, step, Direction::kToVmem);
        // The output slots of this step were last copied out at step - 2.
        if (!outputs.empty()) {
          If(step >= begin + kSlots, [&] {  // REF_CAPTURE_OK=called inline
            WaitCopies(outputs, output_slots, step - kSlots, Direction::kToHbm);
          });
        }
        body(PipelineStep{.step = step,
                          .inputs = SlotRefs(inputs, input_slots, step),
                          .outputs = SlotRefs(outputs, output_slots, step)});
        StartCopies(outputs, output_slots, step, Direction::kToHbm);
      });

      // The loop waited for every output copy but the last two.
      if (!outputs.empty()) {
        If(end - 1 > begin, [&] {  // REF_CAPTURE_OK=called inline
          WaitCopies(outputs, output_slots, end - 2, Direction::kToHbm);
        });
        WaitCopies(outputs, output_slots, end - 1, Direction::kToHbm);
      }
    });
  });
}

}  // namespace torch_tpu::kiln
