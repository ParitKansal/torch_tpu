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

"""Unit tests for KV cache implementations and mark_dynamic."""

from unittest import mock
from absl.testing import absltest
import torch
import kv_cache
import model
import runner_dynamic


class KvCacheTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # pylint: disable=protected-access
    torch._dynamo.reset()

  def test_dynamic_cache_mark_dynamic(self):
    # pylint: disable=protected-access
    cache = kv_cache.DynamicCache()
    k1 = torch.randn(1, 2, 8, 16)
    v1 = torch.randn(1, 2, 8, 16)
    cache.update(k1, v1, layer_idx=0)
    cache.mark_dynamic()

    self.assertIn(2, cache.keys[0]._dynamo_dynamic_indices)
    self.assertIn(2, cache.values[0]._dynamo_dynamic_indices)

  def test_dynamic_cache_multi_layer_mark_dynamic(self):
    # pylint: disable=protected-access
    cache = kv_cache.DynamicCache()
    num_layers = 4
    for layer_idx in range(num_layers):
      k = torch.randn(1, 2, 8, 16)
      v = torch.randn(1, 2, 8, 16)
      cache.update(k, v, layer_idx=layer_idx)

    cache.mark_dynamic()

    for layer_idx in range(num_layers):
      self.assertIn(2, cache.keys[layer_idx]._dynamo_dynamic_indices)
      self.assertIn(2, cache.values[layer_idx]._dynamo_dynamic_indices)

  def test_dynamic_cache_empty_mark_dynamic(self):
    cache = kv_cache.DynamicCache()
    # Should be a safe no-op on an empty cache.
    cache.mark_dynamic()

  def test_dynamic_cache_mismatched_keys_values(self):
    cache = kv_cache.DynamicCache()
    cache.keys.append(torch.randn(1, 2, 8, 16))
    # cache.values is empty, length mismatch

    with self.assertRaisesRegex(ValueError, "Mismatched keys .* and values"):
      cache.mark_dynamic()

  def test_dynamic_cache_invalid_tensor_rank(self):
    cache = kv_cache.DynamicCache()
    # 3D tensors instead of required 4D
    cache.keys.append(torch.randn(2, 8, 16))
    cache.values.append(torch.randn(2, 8, 16))

    with self.assertRaisesRegex(
        ValueError, "Expected 4D key and value tensors"
    ):
      cache.mark_dynamic()

  def test_static_cache_mark_dynamic(self):
    cache = kv_cache.StaticCache(
        num_layers=1,
        batch_size=1,
        num_kv_heads=2,
        max_cache_len=16,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
    )
    # mark_dynamic on StaticCache should be a no-op and not raise.
    cache.mark_dynamic()

  def test_decode_recompilation_eliminated_with_mark_dynamic(self):
    torch._dynamo.reset()

    class DummyModel(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(16, 16)

      def forward(
          self, x: torch.Tensor, cache: kv_cache.Cache
      ) -> tuple[torch.Tensor, kv_cache.Cache]:
        k = x[:, :, :8].unsqueeze(1)
        v = x[:, :, 8:].unsqueeze(1)
        k, v = cache.update(k, v, layer_idx=0)
        out = self.proj(x) + k.sum() + v.sum()
        return out, cache

    dummy_model = DummyModel()
    compile_count = 0

    def counting_backend(gm, example_inputs):
      del example_inputs
      nonlocal compile_count
      compile_count += 1
      return gm

    compiled_model = torch.compile(
        dummy_model,
        backend=counting_backend,
        dynamic=None,
    )

    # Initial prefill
    cache = kv_cache.DynamicCache()
    init_k = torch.randn(1, 1, 4, 8)
    init_v = torch.randn(1, 1, 4, 8)
    cache.update(init_k, init_v, layer_idx=0)

    # Call mark_dynamic once before decode (no bounds):
    cache.mark_dynamic()

    # Step 1: seq_len = 4 -> compiles for the first time
    x1 = torch.randn(1, 1, 16)
    _, cache = compiled_model(x1, cache)
    self.assertEqual(compile_count, 1)

    # Step 2: seq_len = 5 -> dynamic shape guard matches, NO recompilation
    x2 = torch.randn(1, 1, 16)
    _, cache = compiled_model(x2, cache)
    self.assertEqual(compile_count, 1)

    # Step 3: seq_len = 6 -> dynamic shape guard matches, NO recompilation
    x3 = torch.randn(1, 1, 16)
    _, cache = compiled_model(x3, cache)
    self.assertEqual(compile_count, 1)

  def test_muse_glimmer_decode_with_mark_dynamic(self):
    torch._dynamo.reset()
    cfg = model.Config(
        vocab_size=32,
        num_layers=4,
        dmodel=32,
        dff=64,
        dhead=8,
        num_q_heads=4,
        num_kv_heads=2,
        prefill_seq_len=4,
        num_decode_steps=4,
    )
    m = model.MuseGlimmer30B(cfg)

    # Prefill eager
    prompt = torch.randint(0, cfg.vocab_size, (1, cfg.prefill_seq_len))
    logits, past_kv = m(prompt)

    # Compile with dynamic=None and eager backend (avoids C++ compiler
    # dependency in hermetic sandbox).
    compiled_model = torch.compile(m, backend="eager", dynamic=None)

    # Mark dynamic once before the decode loop (bare, no bounds).
    past_kv.mark_dynamic()

    next_tok = torch.argmax(logits[:, -1:, :], dim=-1)
    for _ in range(cfg.num_decode_steps):
      logits, past_kv = compiled_model(next_tok, past_key_values=past_kv)
      next_tok = torch.argmax(logits[:, -1:, :], dim=-1)

    self.assertEqual(
        past_kv.get_seq_length(), cfg.prefill_seq_len + cfg.num_decode_steps
    )

  def test_muse_glimmer_constraint_violation_when_max_val_exceeds_buffer(self):
    torch._dynamo.reset()
    cfg = model.Config(
        vocab_size=32,
        num_layers=4,
        dmodel=32,
        dff=64,
        dhead=8,
        num_q_heads=4,
        num_kv_heads=2,
        prefill_seq_len=4,
        num_decode_steps=4,
    )
    m = model.MuseGlimmer30B(cfg)

    prompt = torch.randint(0, cfg.vocab_size, (1, cfg.prefill_seq_len))
    logits, past_kv = m(prompt)
    compiled_model = torch.compile(m, backend="eager", dynamic=None)

    # If max_val is set to prefill_seq_len + num_decode_steps (8), Dynamo's
    # buffer slice guard past_len <= 7 will detect a constraint violation.
    invalid_max_val = cfg.prefill_seq_len + cfg.num_decode_steps
    for idx in range(len(past_kv.keys)):
      if past_kv.window(idx) is None:
        torch._dynamo.mark_dynamic(
            past_kv.keys[idx], 2, min=4, max=invalid_max_val
        )
        torch._dynamo.mark_dynamic(
            past_kv.values[idx], 2, min=4, max=invalid_max_val
        )
    next_tok = torch.argmax(logits[:, -1:, :], dim=-1)

    with self.assertRaises(
        torch.fx.experimental.symbolic_shapes.ConstraintViolationError
    ):
      compiled_model(next_tok, past_key_values=past_kv)

  def test_dynamic_cache_swa_layers_not_marked_dynamic(self):
    # pylint: disable=protected-access
    cache = kv_cache.DynamicCache(swa_size=4, swa_layers=(True, False))
    # Layer 0 is SWA (window=4), Layer 1 is full-attention.
    k = torch.randn(1, 1, 8, 16)
    v = torch.randn(1, 1, 8, 16)
    cache.update(k, v, layer_idx=0)
    cache.update(k, v, layer_idx=1)

    # Layer 0 holds the fixed window of 4 tokens.
    self.assertEqual(cache.keys[0].shape[-2], 4)
    # Layer 1 holds all 8 tokens.
    self.assertEqual(cache.keys[1].shape[-2], 8)

    cache.mark_dynamic()

    # Layer 0 is SWA, so it remains static (not marked dynamic).
    self.assertNotIn(2, getattr(cache.keys[0], "_dynamo_dynamic_indices", ()))

    # Layer 1 is full-attention, so it is marked dynamic along dim 2.
    self.assertIn(2, getattr(cache.keys[1], "_dynamo_dynamic_indices", ()))

  @mock.patch.object(torch.accelerator, "synchronize")
  def test_runner_dynamic_model_generate_mark_dynamic(self, mock_sync):
    del mock_sync
    torch._dynamo.reset()
    cfg = model.Config(
        vocab_size=32,
        num_layers=4,
        dmodel=32,
        dff=64,
        dhead=8,
        num_q_heads=4,
        num_kv_heads=2,
        prefill_seq_len=4,
        num_decode_steps=4,
    )
    m = model.MuseGlimmer30B(cfg)
    compiled_m = torch.compile(m, backend="eager", dynamic=None)
    inputs = torch.randint(0, cfg.vocab_size, (1, cfg.prefill_seq_len))
    output_text, metrics = runner_dynamic.model_generate(
        compiled_m,
        inputs,
        tokenizer=None,
        num_decode_steps=cfg.num_decode_steps,
        mark_dynamic=True,
    )
    self.assertIsInstance(output_text, str)
    self.assertIn("prefill_time_ms", metrics)
    self.assertLen(metrics["decode_step_times_ms"], cfg.num_decode_steps)

  @mock.patch.object(torch.accelerator, "synchronize")
  def test_runner_dynamic_swa_boundary_crossing(self, mock_sync):
    del mock_sync
    torch._dynamo.reset()
    cfg = model.Config(
        vocab_size=32,
        num_layers=4,
        dmodel=32,
        dff=64,
        dhead=8,
        num_q_heads=4,
        num_kv_heads=2,
        swa_size=4,
        prefill_seq_len=2,
        num_decode_steps=4,
    )
    m = model.MuseGlimmer30B(cfg)
    compiled_m = torch.compile(m, backend="eager", dynamic=None)
    inputs = torch.randint(0, cfg.vocab_size, (1, cfg.prefill_seq_len))
    output_text, metrics = runner_dynamic.model_generate(
        compiled_m,
        inputs,
        tokenizer=None,
        num_decode_steps=cfg.num_decode_steps,
        mark_dynamic=True,
    )
    self.assertIsInstance(output_text, str)
    self.assertLen(metrics["decode_step_times_ms"], cfg.num_decode_steps)

  def test_runner_dynamic_model_generate_short_input_guard(self):
    cfg = model.Config(vocab_size=32, num_layers=4, dmodel=32, dhead=8)
    m = model.MuseGlimmer30B(cfg)
    # Length 1 inputs should trigger the ValueError guard when
    # mark_dynamic=True.
    inputs = torch.randint(0, cfg.vocab_size, (1, 1))
    with self.assertRaisesRegex(
        ValueError, "Dynamo requires dynamic dimension size >= 2"
    ):
      runner_dynamic.model_generate(
          m, inputs, tokenizer=None, num_decode_steps=2, mark_dynamic=True
      )


if __name__ == "__main__":
  absltest.main()
