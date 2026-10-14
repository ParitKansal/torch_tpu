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

"""Regression test for eager collectives when TPU is touched before process group creation."""

from absl.testing import absltest
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from torch_tpu._internal.utils import test_utils as utils
from torch_tpu._internal.distributed import multiprocessing
from tests.distributed import continuous_test_utils
from tests.distributed import distributed_utils


class CollectiveOpsDeviceOrderTest(
    continuous_test_utils.TpuMultiProcContinuousTest
):
  world_size = 8

  @classmethod
  def _init_pg(cls, rank: int, world_size: int, rdvz_file: str | None) -> None:
    multiprocessing.parse_absl_flags()
    distributed_utils.set_worker_env(
        local_rank=rank,
        nproc_per_node=world_size,
    )
    warmup = torch.zeros(1, device="tpu")
    _ = warmup.cpu()
    torch.accelerator.synchronize()
    super()._init_pg(rank, world_size, rdvz_file)

  def test_collectives_with_pre_pg_warmup(self) -> None:
    explicit_world_pg = dist.new_group(
        ranks=list(range(self.world_size)), backend="tpu_dist"
    )
    expected = torch.arange(self.world_size, dtype=torch.float32)

    # all_reduce
    summed = torch.full(
        (self.world_size,), float(self.rank), device="tpu", dtype=torch.float32
    )
    dist.all_reduce(summed)
    utils.assert_close(
        summed.cpu(),
        torch.full(
            (self.world_size,), self.world_size * (self.world_size - 1) / 2.0
        ),
    )

    # all_gather (default group)
    gathered_default = torch.empty(
        self.world_size, device="tpu", dtype=torch.float32
    )
    dist.all_gather_into_tensor(
        gathered_default,
        torch.full((1,), float(self.rank), device="tpu", dtype=torch.float32),
    )
    utils.assert_close(gathered_default.cpu(), expected)

    # all_gather (explicit world group)
    gathered_new_group = torch.empty(
        self.world_size, device="tpu", dtype=torch.float32
    )
    dist.all_gather_into_tensor(
        gathered_new_group,
        torch.full((1,), float(self.rank), device="tpu", dtype=torch.float32),
        group=explicit_world_pg,
    )
    utils.assert_close(gathered_new_group.cpu(), expected)

    # all_to_all_single
    send = torch.full(
        (self.world_size,), float(self.rank), device="tpu", dtype=torch.float32
    )
    recv = torch.empty(self.world_size, device="tpu", dtype=torch.float32)
    dist.all_to_all_single(recv, send)
    utils.assert_close(recv.cpu(), expected)


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
