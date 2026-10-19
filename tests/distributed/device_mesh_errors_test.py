# Copyright 2026 Google LLC
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

"""Tests for topology-aware device mesh error handling."""

import dataclasses
import os
from typing import Final
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import distributed as dist
import torch.distributed.tensor as dt
import torch.multiprocessing as mp
from torch_tpu import _loader
from torch_tpu._internal.distributed import device_mesh
from torch_tpu._internal.utils import test_utils as utils
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests.distributed import device_mesh_test_utils
from tests.distributed import distributed_utils

_SHAPE_MISMATCH: Final[str] = "Tensor shape mismatch"
_SFLAG_TIMEOUT: Final[str] = "Wait timeout on sflag"
_ZERO_ALL_GATHER_DIM: Final[str] = (
    "dimension size of operand at 'all_gather_dim' cannot be zero"
)


def _replicate_tensor(
    tensor_size: int,
    compiled: bool,
    ndims: int = 1,
    pass_shape_stride: bool = False,
    trailing_dim_size: int = 4,
) -> None:
  """Redistributes a tensor from shard to replicate."""
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  mesh = dt.init_device_mesh("tpu", (world_size,))

  trailing_dims = [trailing_dim_size] * (ndims - 1)
  global_shape = (
      torch.Size([tensor_size] + trailing_dims) if pass_shape_stride else None
  )
  global_stride = (
      torch.empty(global_shape, device="meta").stride()
      if pass_shape_stride
      else None
  )

  def shard_and_redistribute(x):
    dtensor = dt.DTensor.from_local(
        x,
        device_mesh=mesh,
        placements=[dt.Shard(0)],
        shape=global_shape,
        stride=global_stride,
    )
    return dtensor.redistribute(mesh, [dt.Replicate()]).to_local()

  chunk_size = (tensor_size + world_size - 1) // world_size
  start = min(rank * chunk_size, tensor_size)
  end = min((rank + 1) * chunk_size, tensor_size)
  local_size = end - start
  local_shape = [local_size] + trailing_dims
  local_tensor = torch.ones(*local_shape, dtype=torch.float32, device="tpu")

  if compiled:
    func = torch.compile(
        shard_and_redistribute, backend="tpu", options={"serializable": False}
    )
  else:
    func = shard_and_redistribute

  output = func(local_tensor)
  out = output.cpu()

  if pass_shape_stride or rank == 0:
    expected_shape = [tensor_size] + trailing_dims
    expected = torch.ones(*expected_shape, dtype=torch.float32)
    utils.assert_close(out, expected)


class DeviceMeshErrorsTest(seed_test_utils.MultiProcessRepeatableTest):

  def setUp(self):
    super().setUp()
    _loader._init_device("tpu")  # INIT_DEVICE_OK=CPU runner

  def test_pre_initialization_errors(self):
    # If the process group is not initialized, topology_aware_mesh should
    # raise a RuntimeError.
    with mock.patch("torch.distributed.is_initialized", return_value=False):
      with self.assertRaisesRegex(
          RuntimeError, "process group must be initialized"
      ):
        torch.tpu.distributed.topology_aware_mesh(mesh_shape=(8, 8))

  def test_multi_slice_rejects_per_slice_mesh_shape(self):
    infos = device_mesh_test_utils.v7x_multislice_infos(num_slices=2)

    with self.assertRaisesRegex(ValueError, "must divide mesh_shape"):
      device_mesh_test_utils.patched_call(
          infos,
          lambda: torch.tpu.distributed.topology_aware_mesh(
              mesh_shape=(1, 8),
              dcn_mesh_shape=(2, 1),
          ),
      )

  def test_multi_slice_rejects_mesh_shape_smaller_than_world(self):
    infos = device_mesh_test_utils.v7x_multislice_infos(num_slices=2)

    with self.assertRaisesRegex(ValueError, "describes the full mesh"):
      device_mesh_test_utils.patched_call(
          infos,
          lambda: torch.tpu.distributed.topology_aware_mesh(
              mesh_shape=(2, 4),
              dcn_mesh_shape=(2, 1),
          ),
      )

  def test_multi_slice_rejects_all_ones_dcn_mesh_shape(self):
    # (1, 1) asks for no DCN split, which two slices cannot satisfy. It has to
    # be rejected rather than falling through to the single-slice path.
    infos = device_mesh_test_utils.v7x_multislice_infos(num_slices=2)

    with self.assertRaisesRegex(ValueError, "covers 1 slices"):
      device_mesh_test_utils.patched_call(
          infos,
          lambda: torch.tpu.distributed.topology_aware_mesh(
              mesh_shape=(2, 8),
              dcn_mesh_shape=(1, 1),
          ),
      )

  def test_multi_slice_rejects_all_ones_dcn_mesh_shape_3d(self):
    infos = device_mesh_test_utils.v7x_multislice_infos(num_slices=2)

    with self.assertRaisesRegex(ValueError, "covers 1 slices"):
      device_mesh_test_utils.patched_call(
          infos,
          lambda: torch.tpu.distributed.topology_aware_mesh(
              mesh_shape=(2, 2, 4),
              dcn_mesh_shape=(1, 1, 1),
          ),
      )

  def test_multi_slice_requires_megascale(self):
    infos = device_mesh_test_utils.v7x_multislice_infos(
        num_slices=2, has_slice_index=False
    )

    with self.assertRaisesRegex(RuntimeError, "megascale_port"):
      device_mesh_test_utils.patched_call(
          infos,
          lambda: torch.tpu.distributed.topology_aware_mesh(
              mesh_shape=(2, 8),
              dcn_mesh_shape=(2, 1),
          ),
      )

  def test_rejects_partial_slice_index_reporting(self):
    infos = device_mesh_test_utils.v7x_multislice_infos(num_slices=2)
    # One rank came up without the MegaScale client, so it cannot say which
    # slice it is on and the job's layout is only partly observable.
    infos[3] = dataclasses.replace(infos[3], has_slice_index=False)

    with self.assertRaisesRegex(RuntimeError, "15 of 16 torch ranks"):
      device_mesh_test_utils.patched_call(
          infos,
          lambda: torch.tpu.distributed.topology_aware_mesh(mesh_shape=(2, 8)),
      )

  def test_multi_slice_rejects_ragged_slices(self):
    infos = device_mesh_test_utils.v7x_multislice_infos(num_slices=2)
    # Drop one device from slice 1 and pad slice 0 to keep the world size, so
    # only the per-slice balance check can catch it.
    infos = infos[:9] + infos[10:] + [infos[9]]
    infos[-1] = dataclasses.replace(infos[-1], slice_index=0, id=100099)

    with self.assertRaisesRegex(ValueError, "same number of devices"):
      device_mesh_test_utils.patched_call(
          infos,
          lambda: torch.tpu.distributed.topology_aware_mesh(
              mesh_shape=(2, 8),
              dcn_mesh_shape=(2, 1),
          ),
      )

  def test_infer_dcn_mesh_shape_refuses_to_split_shard_axis(self):
    # dp_replicate=2 cannot absorb 4 slices. Falling through to dp_shard would
    # split shard groups across DCN, so this has to be refused instead.
    with self.assertRaisesRegex(ValueError, "Refusing to fall back"):
      device_mesh._infer_dcn_mesh_shape((1, 2, 64, 1, 1), 4)

  def test_infer_dcn_mesh_shape_rejects_all_trivial_mesh(self):
    with self.assertRaisesRegex(ValueError, "every axis has extent 1"):
      device_mesh._infer_dcn_mesh_shape((1, 1, 1), 4)

  def test_infer_dcn_mesh_shape_rejects_indivisible_mesh(self):
    with self.assertRaisesRegex(ValueError, "Cannot spread mesh_shape"):
      device_mesh._infer_dcn_mesh_shape((1, 3, 5, 1, 1), 4)


class UnevenShardingErrorsTest(seed_test_utils.MultiProcessRepeatableTest):

  @parameterized.named_parameters(
      (
          "1d_compiled_fail_small",
          1,
          True,
          4,
          f"{_SHAPE_MISMATCH}|{_SFLAG_TIMEOUT}",
      ),
      (
          "1d_eager_fail_small",
          1,
          False,
          4,
          f"{_SHAPE_MISMATCH}|{_SFLAG_TIMEOUT}|{_ZERO_ALL_GATHER_DIM}",
      ),
      (
          "1d_compiled_fail_uneven",
          1,
          True,
          12,
          f"{_SHAPE_MISMATCH}|{_SFLAG_TIMEOUT}",
      ),
      (
          "1d_eager_fail_uneven",
          1,
          False,
          12,
          f"{_SHAPE_MISMATCH}|{_SFLAG_TIMEOUT}|{_ZERO_ALL_GATHER_DIM}",
      ),
      ("2d_compiled_fail_small", 2, True, 4, _SFLAG_TIMEOUT),
      (
          "2d_eager_fail_small",
          2,
          False,
          4,
          f"{_SFLAG_TIMEOUT}|{_ZERO_ALL_GATHER_DIM}",
      ),
      ("2d_compiled_fail_uneven", 2, True, 12, _SFLAG_TIMEOUT),
      (
          "2d_eager_fail_uneven",
          2,
          False,
          12,
          f"{_SFLAG_TIMEOUT}|{_ZERO_ALL_GATHER_DIM}",
      ),
  )
  def test_replicate_tensor_without_shape(
      self, ndims, compiled, tensor_size, expected_regex
  ):
    with self.assertRaisesRegex(Exception, expected_regex):
      distributed_utils.dist_run(
          nproc_per_node=8,
          fn=_replicate_tensor,
          tensor_size=tensor_size,
          compiled=compiled,
          ndims=ndims,
          pass_shape_stride=False,
      )

  @parameterized.named_parameters(
      ("1d_without_shape_stride_small", 1, 4),
      ("1d_without_shape_stride_uneven", 1, 12),
      ("2d_without_shape_stride_small", 2, 4),
      ("2d_without_shape_stride_uneven", 2, 12),
  )
  def test_replicate_tensor_debug_eager_without_shape_stride(
      self, ndims, tensor_size
  ):
    def run_fn():
      with mock.patch.dict(os.environ, {"TORCH_TPU_ENABLE_DEBUG_CHECKS": "1"}):
        distributed_utils.dist_run(
            nproc_per_node=8,
            fn=_replicate_tensor,
            tensor_size=tensor_size,
            compiled=False,
            ndims=ndims,
            pass_shape_stride=False,
        )

    with self.assertRaisesRegex(Exception, "Inconsistent tensor metadata"):
      run_fn()


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
