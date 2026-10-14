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

"""Unit tests for the distributed Muse Glimmer SimpleFSDP training benchmark."""

import types
from absl.testing import absltest
import torch
from torch import nn
import torch.multiprocessing as mp
from torch_tpu._internal.utils import test_utils
from torch_tpu._internal.distributed import multiprocessing
from perflab import context as context_lib
from perflab import target as target_lib
from perflab.distributed_models import muse_glimmer
from perflab.model_utils import torchtitan_batching
from perflab.model_utils import torchtitan_muse_glimmer
from tests import seed_test_utils
from tests.distributed import distributed_utils

_WORLD_SIZE = 2


def _worker_cpu_training_step() -> None:
  """SimpleFSDP training steps (B=1 and B=2) of the debug model on gloo / CPU."""
  torch.distributed.init_process_group(backend="gloo", init_method="env://")
  try:
    seq_len = 64
    rank = torch.distributed.get_rank()
    config = torchtitan_muse_glimmer.text_config(
        "debugmodel", num_layers=4, seq_len=seq_len
    )
    model = muse_glimmer.build_model(
        config, "cpu", _WORLD_SIZE, torch.float32, seq_len=seq_len
    )
    tokens, labels = torchtitan_muse_glimmer.make_inputs(
        config.vocab_size, seq_len, rank, "cpu", batch_size=1
    )
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    before = [p.detach().to_local().clone() for p in model.parameters()]

    loss = torchtitan_muse_glimmer.cross_entropy_loss(
        model, (tokens,), {"labels": labels}
    )
    loss.backward()
    opt.step()

    assert torch.isfinite(loss), f"Non-finite loss {loss}"
    after = [p.detach().to_local() for p in model.parameters()]
    assert any(
        not torch.equal(b, a) for b, a in zip(before, after)
    ), "No parameter changed after the optimizer step"

    # Also verify a B=2 step through SimpleFSDP + activation checkpointing.
    opt.zero_grad()
    tokens_b2, labels_b2 = torchtitan_muse_glimmer.make_inputs(
        config.vocab_size, seq_len, rank, "cpu", batch_size=2
    )
    positions_b2 = torch.arange(seq_len, dtype=torch.long).repeat(2)
    before_b2 = [p.detach().to_local().clone() for p in model.parameters()]
    loss_b2 = torchtitan_muse_glimmer.cross_entropy_loss(
        model,
        (tokens_b2,),
        {"labels": labels_b2, "positions": positions_b2},
    )
    loss_b2.backward()
    opt.step()
    assert torch.isfinite(loss_b2), f"Non-finite B=2 loss {loss_b2}"
    after_b2 = [p.detach().to_local() for p in model.parameters()]
    assert any(
        not torch.equal(b, a) for b, a in zip(before_b2, after_b2)
    ), "No parameter changed after the B=2 optimizer step"
  finally:
    torch.distributed.destroy_process_group()


class _LogitsModel(nn.Module):
  """Returns fixed logits, ignoring its input."""

  def __init__(self, logits: torch.Tensor):
    super().__init__()
    self.logits = logits

  def forward(self, tokens: torch.Tensor) -> torch.Tensor:
    del tokens  # Unused.
    return self.logits


class DistributedMuseGlimmerBenchmarkTest(
    seed_test_utils.MultiProcessRepeatableTest
):

  def test_rejects_unsupported_platforms(self):
    for platform in (
        target_lib.Platform.V7_1X1X1,
        target_lib.Platform.V7_2X2X2,
        target_lib.Platform.B200_1,
    ):
      ctx = context_lib.Context(
          target=target_lib.make_target(platform),
          run_scope=context_lib.RunScope.PRESUBMIT,
      )
      with self.assertRaisesRegex(
          target_lib.UnsupportedBenchmark, "not supported on platform"
      ):
        muse_glimmer.muse_glimmer_30b_4_layers_simple_fsdp_training(ctx)

  def test_requires_initialized_process_group(self):
    ctx = context_lib.Context(
        target=target_lib.make_target(target_lib.Platform.V7_1X1X1_2TC),
        run_scope=context_lib.RunScope.PRESUBMIT,
    )
    with self.assertRaisesRegex(
        target_lib.UnsupportedBenchmark, "initialized torch.distributed"
    ):
      muse_glimmer.muse_glimmer_30b_4_layers_simple_fsdp_training(ctx)

  def test_cross_entropy_loss(self):
    logits = torch.randn(5, 7, dtype=torch.bfloat16)
    labels = torch.randint(0, 7, (5,))
    loss = torchtitan_muse_glimmer.cross_entropy_loss(
        _LogitsModel(logits), (torch.zeros(5),), {"labels": labels}
    )
    self.assertEqual(loss.dtype, torch.float32)
    test_utils.assert_close(
        loss, nn.functional.cross_entropy(logits.float(), labels)
    )

  def test_text_config_is_30b_hybrid_layout(self):
    config = torchtitan_muse_glimmer.text_config(
        "30B", num_layers=4, seq_len=2048
    )
    self.assertEqual(config.dim, 6656)
    self.assertEqual(config.vocab_size, 202048)
    self.assertIsNone(config.vision_encoder)
    self.assertEqual(
        [layer.attention.rope is not None for layer in config.layers],
        [True, True, True, False],
    )
    self.assertEqual(
        [layer.attention.window_size for layer in config.layers],
        [2048, 2048, 2048, None],
    )

  def test_simple_fsdp_training_step_ws2_cpu(self):
    distributed_utils.dist_run(_WORLD_SIZE, _worker_cpu_training_step)

  def test_batch_size_for_target(self):
    for platform in (
        target_lib.Platform.V7_1X1X1_2TC,
        target_lib.Platform.V7_1X2X1,
        target_lib.Platform.V7_2X2X1,
        target_lib.Platform.CPU,
    ):
      self.assertEqual(
          torchtitan_batching.batch_size_for_target(
              target_lib.make_target(platform)
          ),
          1,
      )
    for platform in (
        target_lib.Platform.B200_2,
        target_lib.Platform.B200_4,
        target_lib.Platform.B200_8,
    ):
      self.assertEqual(
          torchtitan_batching.batch_size_for_target(
              target_lib.make_target(platform)
          ),
          2,
      )
    tpu_target = target_lib.make_target(target_lib.Platform.V7_2X2X1)
    gpu_target = target_lib.make_target(target_lib.Platform.B200_4)
    self.assertEqual(
        8 * torchtitan_batching.batch_size_for_target(tpu_target) * 2048,
        16384,
    )
    self.assertEqual(
        4 * torchtitan_batching.batch_size_for_target(gpu_target) * 2048,
        16384,
    )
    with self.assertRaisesRegex(ValueError, "unsupported device kind"):
      torchtitan_batching.batch_size_for_target(
          types.SimpleNamespace(device_kind="unknown")  # pyrefly: ignore[bad-argument-type]
      )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
