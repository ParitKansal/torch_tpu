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

import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F
import kv_cache
import rope


def is_swa_layer(layer_idx: int) -> bool:
  """Returns True if `layer_idx` uses sliding-window attention.

  Every fourth layer uses full attention, the others use SWA.
  """
  return (layer_idx % 4) != 3


@dataclasses.dataclass
class Config:
  # General model config
  vocab_size: int = 202048
  num_layers: int = 52
  dmodel: int = 6656
  dff: int = 19968
  dhead: int = 128
  num_q_heads: int = 32
  num_kv_heads: int = 2

  # Details
  swa_size: int = 2048
  rope_freq_base: float = 500000.0
  rmsnorm_eps: float = 1e-5
  post_norm_eps: float = 1e-8
  qk_scale_factor: float = 3.87
  final_logit_softcapping: float = 20.0
  output_multiplier: float = 0.19611613513818404

  # Perf evaluation config
  prefill_seq_len: int = 128  # Default prefill length (overridden by prompt)
  num_decode_steps: int = 10  # Number of decode steps

  def swa_layers(self) -> tuple[bool, ...]:
    """Returns the per-layer sliding-window flags, indexed by layer."""
    return tuple(is_swa_layer(i) for i in range(self.num_layers))


# TODO: Add a small parity test comparing prefill and decode logits
# against HuggingFace (via load_weights_from_hf) for both StaticCache and
# DynamicCache.
class MuseGlimmer30B(nn.Module):

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

    self.embed_tokens = nn.Embedding(config.vocab_size, config.dmodel)
    self.embed_norm = nn.RMSNorm(
        (config.dmodel), eps=config.rmsnorm_eps, elementwise_affine=False
    )
    self.layers = nn.ModuleList(
        DecoderBlock(config, layer_idx=i) for i in range(config.num_layers)
    )
    self.norm = nn.RMSNorm((config.dmodel,), eps=config.rmsnorm_eps)
    self.lm_head = nn.Linear(config.dmodel, config.vocab_size, bias=False)

    max_len = config.prefill_seq_len + config.num_decode_steps
    rope_sin, rope_cos = rope.precompute_rope_sin_cos(
        seq_len=max_len,
        dim_head=config.dhead,
        rope_theta=config.rope_freq_base,
    )
    self.rope_sin = nn.Buffer(rope_sin, persistent=False)
    self.rope_cos = nn.Buffer(rope_cos, persistent=False)

    causal_mask = torch.ones(1, 1, max_len, max_len).tril()
    swa_mask = causal_mask.triu(-config.swa_size + 1)
    self.causal_mask = nn.Buffer(
        torch.where(causal_mask == 1, 0, float("-inf")), persistent=False
    )
    self.swa_mask = nn.Buffer(
        torch.where(swa_mask == 1, 0, float("-inf")), persistent=False
    )

  def forward(
      self,
      tok: torch.Tensor,
      past_key_values: kv_cache.Cache | None = None,
  ) -> tuple[torch.Tensor, kv_cache.Cache]:
    if past_key_values is None:
      past_key_values = kv_cache.DynamicCache(
          self.config.swa_size,
          self.config.swa_layers(),
          max_cache_len=self.causal_mask.shape[-1],
      )
    past_len = past_key_values.get_seq_length()
    seq_len = tok.shape[1]
    if isinstance(past_key_values, kv_cache.StaticCache):
      cache_position = torch.arange(seq_len, device=tok.device) + past_len
      rope_sin_cos = (
          self.rope_sin.index_select(2, cache_position),
          self.rope_cos.index_select(2, cache_position),
      )
      causal_mask = self.causal_mask.index_select(2, cache_position)
      swa_mask = self.swa_mask.index_select(2, cache_position)
      # SWA buffers only hold the window; pick the mask columns of the absolute
      # positions their slots hold after this step.
      swa_mask = swa_mask.index_select(
          3, past_key_values.swa_key_positions(seq_len)
      )
    else:
      # SDPA with is_causal=True aligns the causal mask to the top-left corner,
      # which is only valid when the cache is empty (past_len == 0).
      assert past_len == 0 or seq_len == 1
      kv_len = past_len + seq_len
      rope_sin_cos = (
          self.rope_sin[:, :, past_len:kv_len, :],
          self.rope_cos[:, :, past_len:kv_len, :],
      )
      causal_mask = None
      if past_len == 0:
        # Prefill: SWA layers attend all prompt keys (`update()` returns them)
        # under the band mask.
        swa_mask = self.swa_mask[:, :, :seq_len, :seq_len]
      else:
        # Decode: the SWA buffers hold the window; pick the mask columns of the
        # positions their slots hold, as in the StaticCache branch. The mask is
        # applied unconditionally: while the window is still filling up, the
        # empty slots hold zero keys and must be masked, and once it is full
        # the mask is all zeros but keeping it is what leaves the decode step
        # without a branch on the sequence length.
        swa_mask = self.swa_mask[:, :, past_len:kv_len, :].index_select(
            3, past_key_values.swa_key_positions(seq_len)
        )

    h = self.embed_norm(self.embed_tokens(tok))
    for layer in self.layers:
      attn_mask = swa_mask if layer.self_attn.is_swa else causal_mask
      h = layer(
          h,
          rope_sin_cos,
          past_key_values=past_key_values,
          attn_mask=attn_mask,
      )
    h = self.norm(h)

    alpha = self.config.output_multiplier
    c = self.config.final_logit_softcapping
    logits = F.tanh(self.lm_head(h[:, -1:, :]) * (alpha / c)) * c
    return logits, past_key_values


class DecoderBlock(nn.Module):

  def __init__(self, config: Config, layer_idx: int):
    super().__init__()
    self.config = config
    self.layer_idx = layer_idx

    self.self_attn = Attention(
        config, is_swa=is_swa_layer(layer_idx), layer_idx=layer_idx
    )
    self.mlp = FeedForwardSwiGLU(config)

    dm = config.dmodel
    f32 = torch.float32
    eps_pre, eps_post = config.rmsnorm_eps, config.post_norm_eps
    # NOTE: These should use centered rmsnorm. This is relevant for weight decay
    # and also weight loading from HF checkpoint.
    self.input_layernorm = nn.RMSNorm((dm,), eps=eps_pre, dtype=f32)
    self.post_attention_layernorm = nn.RMSNorm((dm,), eps=eps_post, dtype=f32)
    self.pre_feedforward_layernorm = nn.RMSNorm((dm,), eps=eps_pre, dtype=f32)
    self.post_feedforward_layernorm = nn.RMSNorm((dm,), eps=eps_post, dtype=f32)

  def forward(
      self,
      h: torch.Tensor,
      rope_sin_cos: tuple[torch.Tensor, torch.Tensor],
      past_key_values: kv_cache.Cache,
      attn_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    h = h + self.post_attention_layernorm(
        self.self_attn(
            self.input_layernorm(h),
            rope_sin_cos,
            past_key_values=past_key_values,
            attn_mask=attn_mask,
        )
    )
    h = h + self.post_feedforward_layernorm(
        self.mlp(self.pre_feedforward_layernorm(h))
    )
    return h


class Attention(torch.nn.Module):
  """Grouped-query Attention."""

  def __init__(self, config: Config, is_swa: bool, layer_idx: int = 0):
    super().__init__()
    self.config = config
    self.is_swa = is_swa
    self.layer_idx = layer_idx

    dm, dh = config.dmodel, config.dhead
    num_q_heads = config.num_q_heads
    num_kv_heads = config.num_kv_heads

    self.q_proj = nn.Linear(dm, num_q_heads * dh, bias=False)
    self.k_proj = nn.Linear(dm, num_kv_heads * dh, bias=False)
    self.v_proj = nn.Linear(dm, num_kv_heads * dh, bias=False)

    self.gate_proj = nn.Linear(dm, num_q_heads * dh, bias=False)
    self.o_proj = nn.Linear(num_q_heads * dh, dm, bias=False)

    # Glimmer has QK RMSNorm without learned scale parameters
    self.qk_norm = torch.nn.RMSNorm(
        normalized_shape=(dh,),
        eps=config.rmsnorm_eps,
        dtype=torch.float32,
        elementwise_affine=False,
    )

  def forward(
      self,
      h: torch.Tensor,
      rope_sin_cos: tuple[torch.Tensor, torch.Tensor],
      past_key_values: kv_cache.Cache,
      attn_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    local_bsz, seq_len, _ = h.shape

    dh = self.config.dhead
    num_q_heads = self.config.num_q_heads
    num_kv_heads = self.config.num_kv_heads

    q = self.q_proj(h).view(local_bsz, seq_len, num_q_heads, dh)
    k = self.k_proj(h).view(local_bsz, seq_len, num_kv_heads, dh)
    v = self.v_proj(h).view(local_bsz, seq_len, num_kv_heads, dh)

    qk_scale = self.config.qk_scale_factor
    q = self.qk_norm(q.transpose(1, 2).contiguous()) * qk_scale
    k = self.qk_norm(k.transpose(1, 2).contiguous())
    v = v.transpose(1, 2).contiguous()

    if self.is_swa:
      # In Muse Glimmer 30B, RoPE is only applied in SWA layers; full attention
      # layers use NoPE.
      rs, rc = rope_sin_cos
      q = rope.apply_rope(q, rs, rc)
      k = rope.apply_rope(k, rs, rc)

    k, v = past_key_values.update(k, v, self.layer_idx)
    is_causal = attn_mask is None and seq_len > 1

    attn_out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=is_causal, enable_gqa=True
    )
    attn_out = attn_out.transpose(1, 2)
    attn_out = attn_out.reshape(local_bsz, seq_len, num_q_heads * dh)

    gate = F.sigmoid(self.gate_proj(h))
    out = self.o_proj(attn_out * gate)
    return out


class FeedForwardSwiGLU(nn.Module):
  """Simple FFN with SwiGLU activation."""

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

    dm, df = config.dmodel, config.dff
    self.up_proj = nn.Linear(dm, df, bias=False)
    self.gate_proj = nn.Linear(dm, df, bias=False)
    self.down_proj = nn.Linear(df, dm, bias=False)

  def forward(self, h):
    h = self.up_proj(h) * F.silu(self.gate_proj(h))
    h = self.down_proj(h)
    return h


@torch.no_grad()
def init_nn_module(m: torch.nn.Module):
  for p in m.parameters():
    p.data.copy_(torch.randn_like(p.data) * 0.05)
