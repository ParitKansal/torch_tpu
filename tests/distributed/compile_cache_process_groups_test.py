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

"""Tests that the compile cache keeps apart programs of different groups.

Graphs name process groups by creation order, so two programs whose same-named
groups have different members must not load each other's cached executable.
Groups that a program does not use do not affect its cache key.
"""

import os

from absl.testing import absltest
import torch
from torch import distributed as dist
from torch._dynamo.utils import counters
from torch._inductor.runtime.cache_dir_utils import temporary_cache_dir
from torch_tpu._internal.distributed import spmd_util
from torch_tpu._internal.utils import test_utils as utils
from torch_tpu._internal.distributed import multiprocessing
from tests.distributed import continuous_test_utils

_WORLD = 8

# Each layout splits the world into two groups of 4, created in list order, so
# the group of rank 0 is named "1" in both.
_LAYOUTS = {
    "halves": [[0, 1, 2, 3], [4, 5, 6, 7]],
    "columns": [[0, 2, 4, 6], [1, 3, 5, 7]],
}


_LAYOUT_GROUPS: dict[str, list[dist.ProcessGroup]] = {}


def _get_layout_groups(layout: str) -> list[dist.ProcessGroup]:
  if layout not in _LAYOUT_GROUPS:
    _LAYOUT_GROUPS[layout] = [
        dist.new_group(ranks=ranks) for ranks in _LAYOUTS[layout]
    ]
  return _LAYOUT_GROUPS[layout]


def run_group_all_reduce(
    layout: str,
    cache_dir: str,
    cache_hit: bool,
    unused_group: bool = False,
) -> None:
  """Compiles a group all-reduce and checks its result and the cache use.

  Args:
    layout: The key in _LAYOUTS of the groups to create.
    cache_dir: The directory holding one compile cache per rank. Ranks of a
      group compile the same program, so a shared cache would let one rank load
      another's entry within a run.
    cache_hit: Whether the program is expected to load from the cache.
    unused_group: Whether to also create the groups of the "columns" layout,
      which the program does not use.
  """
  torch._dynamo.reset()
  counters.clear()
  rank = int(os.environ["RANK"])
  assert dist.get_world_size() == _WORLD

  my_group, my_ranks = None, None
  for ranks, group in zip(_LAYOUTS[layout], _get_layout_groups(layout)):
    if rank in ranks:
      my_group, my_ranks = group, ranks
  if unused_group:
    _get_layout_groups("columns")

  @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
  @spmd_util.spmd_safe
  def group_sum(x: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    dist.all_reduce(out, group=my_group)
    return out

  with temporary_cache_dir(os.path.join(cache_dir, str(rank))):
    out = group_sum(torch.tensor([float(rank)], device="tpu")).cpu()
  utils.assert_close(out, torch.tensor([float(sum(my_ranks))]))
  aot = counters["aot_autograd"]
  assert aot["autograd_cache_hit"] == int(cache_hit), dict(aot)

  if dist.is_initialized():
    torch.accelerator.synchronize()
    dist.barrier()


class CompileCacheProcessGroupsTest(
    continuous_test_utils.TpuMultiProcContinuousTest
):
  world_size = _WORLD

  def test_group_layouts_get_their_own_cache_entries(self):
    cache_dir = self.create_tempdir().full_path
    for layout, cache_hit in (
        ("halves", False),
        ("columns", False),
        ("halves", True),
    ):
      run_group_all_reduce(
          layout=layout,
          cache_dir=cache_dir,
          cache_hit=cache_hit,
      )

  def test_unused_groups_do_not_change_the_cache_key(self):
    cache_dir = self.create_tempdir().full_path
    for unused_group, cache_hit in ((False, False), (True, True)):
      run_group_all_reduce(
          layout="halves",
          cache_dir=cache_dir,
          cache_hit=cache_hit,
          unused_group=unused_group,
      )


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
