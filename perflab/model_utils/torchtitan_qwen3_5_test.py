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

from absl.testing import absltest
import torch
from torch_tpu._internal.utils import test_utils
from perflab.model_utils import torchtitan_qwen3_5
from tests import seed_test_utils
from transformers.models.qwen3_5 import modeling_qwen3_5


def _recurrent_gated_delta(q, k, v, g, beta):
  """Per-token reference (HF `torch_recurrent_gated_delta_rule` math)."""
  q = torchtitan_qwen3_5._l2norm(q.float()) * q.shape[-1] ** -0.5  # pylint: disable=protected-access
  k = torchtitan_qwen3_5._l2norm(k.float())  # pylint: disable=protected-access
  v, g, beta = v.float(), g.float(), beta.float()
  bs, seq_len, n_heads, d_k = k.shape
  state = torch.zeros(bs, n_heads, d_k, v.shape[-1])
  outs = []
  for t in range(seq_len):
    state = state * g[:, t].exp()[..., None, None]
    kv_mem = (state * k[:, t, :, :, None]).sum(-2)
    delta = (v[:, t] - kv_mem) * beta[:, t, :, None]
    state = state + k[:, t, :, :, None] * delta[:, :, None, :]
    outs.append((state * q[:, t, :, :, None]).sum(-2))
  return torch.stack(outs, dim=1)


class TorchtitanQwen35Test(seed_test_utils.RepeatableTest):

  def test_chunked_matches_recurrent(self):
    bs, seq_len, n_heads, d_k, d_v = 2, 100, 3, 16, 8
    q = torch.randn(bs, seq_len, n_heads, d_k)
    k = torch.randn(bs, seq_len, n_heads, d_k)
    v = torch.randn(bs, seq_len, n_heads, d_v)
    g = -torch.rand(bs, seq_len, n_heads)
    beta = torch.rand(bs, seq_len, n_heads)
    # seq_len 100 with chunk 32 exercises the padding of the last chunk.
    out = torchtitan_qwen3_5.chunked_gated_delta(
        q, k, v, g, beta, chunk_size=32
    )
    test_utils.assert_close(
        out, _recurrent_gated_delta(q, k, v, g, beta), atol=1e-4, rtol=1e-4
    )

  def test_grouped_kernel_matches_hf(self):
    seq_len, n_key_heads, n_value_heads, d_k, d_v = 100, 2, 4, 16, 8
    q = torch.randn(seq_len, n_key_heads, d_k)
    k = torch.randn(seq_len, n_key_heads, d_k)
    v = torch.randn(seq_len, n_value_heads, d_v)
    g = -torch.rand(seq_len, n_value_heads)
    beta = torch.rand(seq_len, n_value_heads)
    out = torchtitan_qwen3_5._pure_torch_gated_delta_kernel_forward(  # pylint: disable=protected-access
        None, q, k, v, g, beta
    )
    # HF expects (bs, seqlen, heads, dim) with key heads already repeated.
    repeats = n_value_heads // n_key_heads
    expected, _ = modeling_qwen3_5.torch_chunk_gated_delta_rule(
        q.repeat_interleave(repeats, dim=1).unsqueeze(0),
        k.repeat_interleave(repeats, dim=1).unsqueeze(0),
        v.unsqueeze(0),
        g.unsqueeze(0),
        beta.unsqueeze(0),
        use_qk_l2norm_in_kernel=True,
    )
    test_utils.assert_close(out, expected.squeeze(0), atol=1e-4, rtol=1e-4)

  def test_debug_model_forward_backward(self):
    seq_len = 96
    config = torchtitan_qwen3_5.text_config(
        "debugmodel", num_layers=4, seq_len=seq_len
    )
    self.assertIsNone(config.vision_encoder)
    self.assertLen(config.layers, 4)
    model = torchtitan_qwen3_5.build_model(config, "cpu", torch.float32)
    # 3 GatedDeltaNet layers + 1 attention layer, all swapped to pure torch.
    kernels = [
        m
        for m in model.modules()
        if isinstance(m, torchtitan_qwen3_5.qwen3_5_gdn.GatedDeltaKernel)
    ]
    self.assertLen(kernels, 3)
    for kernel in kernels:
      self.assertEqual(
          kernel.forward.__func__,
          torchtitan_qwen3_5._pure_torch_gated_delta_kernel_forward,
      )

    tokens = torch.randint(0, config.vocab_size, (seq_len,))
    logits = model(tokens)
    self.assertEqual(tuple(logits.shape), (seq_len, config.vocab_size))
    loss = torch.nn.functional.cross_entropy(logits.float(), tokens)
    loss.backward()
    self.assertTrue(torch.isfinite(loss))
    for name, param in model.named_parameters():
      self.assertIsNotNone(param.grad, name)
      self.assertTrue(torch.isfinite(param.grad).all(), name)

  def test_b1_inner_gated_delta_net_matches_upstream(self):
    seq_len = 32
    config = torchtitan_qwen3_5.text_config(
        "debugmodel", num_layers=1, seq_len=seq_len
    )
    torch.manual_seed(42)
    swapped_model = torchtitan_qwen3_5.build_model(
        config, "cpu", torch.float32, seq_len=seq_len
    )
    swapped_gdn = swapped_model.layers["0"].attn.inner_gated_delta_net

    with torch.device("meta"):
      upstream_model = config.build()
    upstream_model.to_empty(device="cpu")
    with torch.no_grad():
      upstream_model.init_states(buffer_device=torch.device("cpu"))
    upstream_model.load_state_dict(swapped_model.state_dict())
    upstream_gdn = upstream_model.layers["0"].attn.inner_gated_delta_net
    # Swap only the kernel on the upstream reference so InnerGatedDeltaNet.forward
    # stays the unmodified upstream method.
    upstream_gdn.kernel.forward = torchtitan_qwen3_5.types.MethodType(
        torchtitan_qwen3_5._pure_torch_gated_delta_kernel_forward,  # pylint: disable=protected-access
        upstream_gdn.kernel,
    )

    tokens = torch.randint(0, config.vocab_size, (seq_len,))
    with torch.no_grad():
      expected = upstream_model(tokens)
      actual_b1 = swapped_model(tokens)
      # B=1 path delegates directly to upstream InnerGatedDeltaNet.forward.
      self.assertTrue(torch.equal(actual_b1, expected))
      # Also verify the batched (B > 1) conv + kernel code path on duplicated
      # [tokens, tokens] matches the B=1 upstream output for both slices.
      actual_b2 = swapped_model(torch.cat([tokens, tokens], dim=0))
      test_utils.assert_close(
          actual_b2[:seq_len], expected, atol=1e-6, rtol=1e-6
      )
      test_utils.assert_close(
          actual_b2[seq_len:], expected, atol=1e-6, rtol=1e-6
      )
      self.assertEqual(swapped_gdn._seq_len, seq_len)  # pylint: disable=protected-access

  def test_batch_size_2_forward_and_backward_equivalence(self):
    seq_len = 64
    config = torchtitan_qwen3_5.text_config(
        "debugmodel", num_layers=4, seq_len=seq_len
    )
    torch.manual_seed(123)
    model = torchtitan_qwen3_5.build_model(
        config, "cpu", torch.float32, seq_len=seq_len
    )

    gen0 = torch.Generator().manual_seed(10)
    seq0 = torch.randint(0, config.vocab_size, (seq_len,), generator=gen0)
    labels0 = torch.randint(0, config.vocab_size, (seq_len,), generator=gen0)
    gen1 = torch.Generator().manual_seed(20)
    seq1 = torch.randint(0, config.vocab_size, (seq_len,), generator=gen1)
    labels1 = torch.randint(0, config.vocab_size, (seq_len,), generator=gen1)

    model.zero_grad()
    logits0 = model(seq0)
    loss0 = torch.nn.functional.cross_entropy(logits0.float(), labels0)
    loss0.backward()
    grads0 = {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
    }

    model.zero_grad()
    logits1 = model(seq1)
    loss1 = torch.nn.functional.cross_entropy(logits1.float(), labels1)
    loss1.backward()
    grads1 = {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
    }

    model.zero_grad()
    tokens_b2 = torch.cat([seq0, seq1], dim=0)
    labels_b2 = torch.cat([labels0, labels1], dim=0)
    positions_b2 = torch.arange(seq_len, dtype=torch.long).repeat(2)
    logits_b2 = model(tokens_b2, positions=positions_b2)
    self.assertEqual(tuple(logits_b2.shape), (2 * seq_len, config.vocab_size))

    # Forward equivalence: each sequence slice in B=2 matches its B=1 forward.
    # Calibrated to 1e-5 for fp32 matmul/SDPA batch-tiling reduction order.
    logits_b2_unpacked = logits_b2.reshape(2, seq_len, config.vocab_size)
    test_utils.assert_close(
        logits_b2_unpacked[0], logits0, atol=1e-5, rtol=1e-5
    )
    test_utils.assert_close(
        logits_b2_unpacked[1], logits1, atol=1e-5, rtol=1e-5
    )

    loss_b2 = torch.nn.functional.cross_entropy(logits_b2.float(), labels_b2)
    test_utils.assert_close(
        loss_b2, 0.5 * (loss0 + loss1), atol=1e-5, rtol=1e-5
    )
    loss_b2.backward()

    # Gradient equivalence: B=2 mean-reduced CE gradient equals the average of
    # the two B=1 gradients. Calibrated to 1e-5 for fp32 backward accumulation
    # across 2*S tokens vs two S-token reductions.
    for name, param in model.named_parameters():
      self.assertIsNotNone(param.grad, name)
      expected_grad = 0.5 * (grads0[name] + grads1[name])
      test_utils.assert_close(param.grad, expected_grad, atol=1e-5, rtol=1e-5)

  def test_invalid_seq_len_and_non_divisible_tokens_raise(self):
    seq_len = 32
    config = torchtitan_qwen3_5.text_config(
        "debugmodel", num_layers=4, seq_len=seq_len
    )
    torch.manual_seed(7)
    model = torchtitan_qwen3_5.build_model(
        config, "cpu", torch.float32, seq_len=seq_len
    )
    # T < S delegates to the single-sequence path without raising.
    short_tokens = torch.randint(0, config.vocab_size, (seq_len // 2,))
    with torch.no_grad():
      short_logits = model(short_tokens)
    self.assertEqual(
        tuple(short_logits.shape), (seq_len // 2, config.vocab_size)
    )
    # Calling the swap helpers again preserves _orig_forward rather than
    # wrapping the wrapper.
    torchtitan_qwen3_5.use_pure_torch_gated_delta(model, seq_len=seq_len)
    torchtitan_qwen3_5.use_batched_sdpa(model, seq_len=seq_len)
    with torch.no_grad():
      self.assertTrue(torch.equal(model(short_tokens), short_logits))

    with self.assertRaisesRegex(ValueError, "must be divisible by seq_len"):
      model(torch.randint(0, config.vocab_size, (seq_len + 1,)))
    with self.assertRaisesRegex(ValueError, "seq_len must be >= 1"):
      torchtitan_qwen3_5.use_pure_torch_gated_delta(model, seq_len=0)
    with self.assertRaisesRegex(ValueError, "seq_len must be >= 1"):
      torchtitan_qwen3_5.use_batched_sdpa(model, seq_len=0)

  def test_custom_subclass_forward_preserved_by_swaps(self):
    seq_len = 16
    config = torchtitan_qwen3_5.text_config(
        "debugmodel", num_layers=4, seq_len=seq_len
    )
    model = torchtitan_qwen3_5.build_model(
        config, "cpu", torch.float32, seq_len=seq_len
    )
    gdn_calls = 0
    gdn = model.layers["0"].attn.inner_gated_delta_net
    base_gdn_forward = gdn._orig_forward  # pylint: disable=protected-access

    def custom_gdn_forward(*args, **kwargs):
      nonlocal gdn_calls
      gdn_calls += 1
      return base_gdn_forward(*args, **kwargs)

    delattr(gdn, "_orig_forward")
    gdn.forward = custom_gdn_forward
    torchtitan_qwen3_5.use_pure_torch_gated_delta(model, seq_len=seq_len)

    sdpa_calls = 0
    sdpa = model.layers["3"].attn.inner_attention
    base_sdpa_forward = sdpa._orig_forward  # pylint: disable=protected-access

    def custom_sdpa_forward(*args, **kwargs):
      nonlocal sdpa_calls
      sdpa_calls += 1
      return base_sdpa_forward(*args, **kwargs)

    delattr(sdpa, "_orig_forward")
    sdpa.forward = custom_sdpa_forward
    torchtitan_qwen3_5.use_batched_sdpa(model, seq_len=seq_len)

    with torch.no_grad():
      model(torch.randint(0, config.vocab_size, (seq_len,)))
      self.assertEqual(gdn_calls, 1)
      self.assertEqual(sdpa_calls, 1)
      model(
          torch.randint(0, config.vocab_size, (2 * seq_len,)),
          positions=torch.arange(seq_len, dtype=torch.long).repeat(2),
      )
      self.assertEqual(sdpa_calls, 2)


if __name__ == "__main__":
  absltest.main()
