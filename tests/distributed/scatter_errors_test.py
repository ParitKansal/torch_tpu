# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from absl.testing import absltest
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from torch_tpu._internal.distributed import multiprocessing
from tests import error_testing as et
from tests.distributed import distributed_utils


def run_scatter_wrong_number_inputs(num_inputs: int) -> None:
  """Runs scatter with number of inputs not equal to process group size."""
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  assert num_inputs != world_size
  src_rank = 0
  size = 5
  output = torch.empty(size, device="tpu", dtype=torch.float32)
  if rank == src_rank:
    inputs = [
        torch.zeros(size, device="tpu", dtype=torch.float32)
        for _ in range(num_inputs)
    ]
    expected_msg = (
        "distributed.scatter(): expected the number of input tensors on the"
        f" root rank to be equal to the group size, got {num_inputs} tensors"
        f" and {world_size} processes"
    )
    with et.assert_raises_message(RuntimeError, tpu=expected_msg):
      torch.distributed.scatter(output, inputs, src=src_rank)
  # Intentionally do NOT call scatter on non-root ranks.
  # The root-rank validation error prevents scatter() from starting; calling
  # scatter() on non-root ranks would block waiting for the root rank.


def run_scatter_wrong_shape_output(
    input_shape: list[int],
    output_shape: list[int],
) -> None:
  """Runs scatter with differently-shaped input and output tensors."""
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  assert input_shape != output_shape
  src_rank = 0
  output = torch.empty(output_shape, device="tpu", dtype=torch.float32)
  if rank == src_rank:
    inputs = [
        torch.zeros(input_shape, device="tpu", dtype=torch.float32)
        for _ in range(world_size)
    ]
    expected_msg = (
        "distributed.scatter(): expected output tensor shape to match input"
        f" tensor shape, got {output_shape} and {input_shape}"
    )
    with et.assert_raises_message(RuntimeError, tpu=expected_msg):
      torch.distributed.scatter(output, inputs, src=src_rank)
  # Intentionally do NOT call scatter on non-root ranks.


def run_scatter_mismatch_input_shapes(
    shape: list[int],
    mismatch_shape: list[int],
) -> None:
  """Runs scatter with one input having a different shape than the others."""
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  assert shape != mismatch_shape
  src_rank = 0
  output = torch.empty(shape, device="tpu", dtype=torch.float32)
  if rank == src_rank:
    inputs = [
        torch.zeros(shape, device="tpu", dtype=torch.float32)
        for _ in range(world_size)
    ]
    inputs[0] = torch.zeros(mismatch_shape, device="tpu", dtype=torch.float32)
    expected_msg = (
        "distributed.scatter(): failed to validate input tensors on root rank:"
        f" expected all tensors to have the same shape, got {mismatch_shape}"
        f" at index 0 and {shape} at index 1"
    )
    with et.assert_raises_message(RuntimeError, tpu=expected_msg):
      torch.distributed.scatter(output, inputs, src=src_rank)
  # Intentionally do NOT call scatter on non-root ranks.


def run_scatter_errors() -> None:
  dist.init_process_group(backend="tpu_dist")
  world_size = int(os.environ["WORLD_SIZE"])
  run_scatter_wrong_number_inputs(num_inputs=world_size + 1)
  run_scatter_wrong_shape_output(input_shape=[2, 3], output_shape=[3, 3])
  run_scatter_mismatch_input_shapes(shape=[6, 2], mismatch_shape=[8, 6, 2])


class ScatterErrorsTest(et.TpuOnlyDistributedErrorTestBase):

  _world_size = distributed_utils.get_tpu_device_count()

  def test_scatter_errors(self):
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=run_scatter_errors,
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
