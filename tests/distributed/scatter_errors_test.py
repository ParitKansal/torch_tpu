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

from absl.testing import absltest
import torch
from torch_tpu._internal.distributed import multiprocessing
from tests import error_testing as et
from tests.distributed import continuous_test_utils


class ScatterErrorsTest(
    continuous_test_utils.TpuMultiProcContinuousTest,
    et.TpuOnlyDistributedErrorTestBase,
):

  world_size = 8

  def test_scatter_wrong_number_inputs(self) -> None:
    """Runs scatter with number of inputs not equal to process group size."""
    num_inputs = 9
    assert num_inputs != self.world_size
    src_rank = 0
    size = 5
    output = torch.empty(size, device="tpu", dtype=torch.float32)
    if self.rank == src_rank:
      inputs = [
          torch.zeros(size, device="tpu", dtype=torch.float32)
          for _ in range(num_inputs)
      ]
      expected_msg = (
          "distributed.scatter(): expected the number of input tensors on the"
          f" root rank to be equal to the group size, got {num_inputs} tensors"
          f" and {self.world_size} processes"
      )
      with et.assert_raises_message(RuntimeError, tpu=expected_msg):
        torch.distributed.scatter(output, inputs, src=src_rank)
    # Intentionally do NOT call scatter on non-root ranks.
    # The root-rank validation error prevents scatter() from starting; calling
    # scatter() on non-root ranks would block waiting for the root rank.

  def test_scatter_wrong_shape_output(self) -> None:
    """Runs scatter with differently-shaped input and output tensors."""
    input_shape = [2, 3]
    output_shape = [3, 3]
    assert input_shape != output_shape
    src_rank = 0
    output = torch.empty(output_shape, device="tpu", dtype=torch.float32)
    if self.rank == src_rank:
      inputs = [
          torch.zeros(input_shape, device="tpu", dtype=torch.float32)
          for _ in range(self.world_size)
      ]
      expected_msg = (
          "distributed.scatter(): expected output tensor shape to match input"
          f" tensor shape, got {output_shape} and {input_shape}"
      )
      with et.assert_raises_message(RuntimeError, tpu=expected_msg):
        torch.distributed.scatter(output, inputs, src=src_rank)
    # Intentionally do NOT call scatter on non-root ranks.

  def test_scatter_mismatch_input_shapes(self) -> None:
    """Runs scatter with one input having a different shape than the others."""
    shape = [6, 2]
    mismatch_shape = [8, 6, 2]
    assert shape != mismatch_shape
    src_rank = 0
    output = torch.empty(shape, device="tpu", dtype=torch.float32)
    if self.rank == src_rank:
      inputs = [
          torch.zeros(shape, device="tpu", dtype=torch.float32)
          for _ in range(self.world_size)
      ]
      inputs[0] = torch.zeros(mismatch_shape, device="tpu", dtype=torch.float32)
      expected_msg = (
          "distributed.scatter(): failed to validate input tensors on root"
          " rank: expected all tensors to have the same shape, got"
          f" {mismatch_shape} at index 0 and {shape} at index 1"
      )
      with et.assert_raises_message(RuntimeError, tpu=expected_msg):
        torch.distributed.scatter(output, inputs, src=src_rank)
    # Intentionally do NOT call scatter on non-root ranks.


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
