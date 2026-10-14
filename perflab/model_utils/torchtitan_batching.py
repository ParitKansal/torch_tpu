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

"""Shared batching helpers for packed `[T]` TorchTitan models in perflab."""

import types
from typing import Any

import torch
from torch import nn
from perflab import target as target_lib
from torchtitan.models.common import attention as common_attention


def batch_size_for_target(target: target_lib.Target) -> int:
  """Returns sequences per process so each chip processes 2 sequences/step.

  On TPU, each v7 chip runs 2 processes (one per TensorCore), so 1 sequence per
  process yields 2 sequences per chip. On GPU, each B200 runs 1 process, so 2
  sequences per process yields 2 sequences per chip. CPU uses 1 sequence per
  process for unit testing.

  Args:
    target: Benchmark target platform/device description.
  """
  if target.device_kind in (
      target_lib.DeviceKind.TPU,
      target_lib.DeviceKind.CPU,
  ):
    return 1
  if target.device_kind == target_lib.DeviceKind.CUDA:
    return 2
  raise ValueError(f"unsupported device kind: {target.device_kind}")


def infer_seq_len(config: Any) -> int | None:
  """Infers the configured `max_context_length` from a TorchTitan config."""
  for layer in getattr(config, "layers", ()):
    attention = getattr(layer, "attention", None)
    rope = getattr(attention, "rope", None)
    if rope is not None:
      return rope.max_context_length
  return None


def make_inputs(
    vocab_size: int,
    seq_len: int,
    rank: int,
    device: str,
    batch_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Returns packed `(batch_size * seq_len,)` token / label sequences for `rank`.

  torchtitan takes packed `(T,)` tokens. The data differs per rank, as in real
  data-parallel training.

  Args:
    vocab_size: Token ids are drawn from `[0, vocab_size)`.
    seq_len: Tokens per sequence.
    rank: Seeds the per-rank data.
    device: Device to place the tensors on.
    batch_size: Number of packed sequences of length `seq_len` per rank.
  """
  if batch_size < 1:
    raise ValueError(f"batch_size must be >= 1, got {batch_size}")
  num_tokens = batch_size * seq_len
  generator = torch.Generator().manual_seed(1234 + rank)
  tokens, labels = [
      torch.randint(
          0, vocab_size, (num_tokens,), dtype=torch.long, generator=generator
      ).to(device)
      for _ in range(2)
  ]
  return tokens, labels


def _batched_sdpa_forward(
    self: common_attention.ScaledDotProductInnerAttention,
    q_BLHK: torch.Tensor,  # pylint: disable=invalid-name
    k_BLHK: torch.Tensor,  # pylint: disable=invalid-name
    v_BLHV: torch.Tensor,  # pylint: disable=invalid-name
    *,
    attention_masks: common_attention.AttentionMasksType | None = None,
    scale: float | None = None,
    enable_gqa: bool = False,
    is_causal: bool = True,
    **kwargs: Any,
) -> torch.Tensor:
  """Drop-in `ScaledDotProductInnerAttention.forward` for packed `(B * S, ...)`."""
  seq_len = getattr(self, "_seq_len", None)
  orig_forward = getattr(self, "_orig_forward", None)
  if orig_forward is None:
    orig_forward = types.MethodType(
        common_attention.ScaledDotProductInnerAttention.forward, self
    )
  if q_BLHK.ndim != 3 or seq_len is None or q_BLHK.shape[0] <= seq_len:
    return orig_forward(
        q_BLHK,
        k_BLHK,
        v_BLHV,
        attention_masks=attention_masks,
        scale=scale,
        enable_gqa=enable_gqa,
        is_causal=is_causal,
        **kwargs,
    )
  num_tokens = q_BLHK.shape[0]
  if num_tokens % seq_len != 0:
    raise ValueError(
        f"num_tokens ({num_tokens}) must be divisible by seq_len ({seq_len})"
    )
  batch_size = num_tokens // seq_len
  out_BLHV = orig_forward(  # pylint: disable=invalid-name
      q_BLHK.reshape(batch_size, seq_len, *q_BLHK.shape[1:]),
      k_BLHK.reshape(batch_size, seq_len, *k_BLHK.shape[1:]),
      v_BLHV.reshape(batch_size, seq_len, *v_BLHV.shape[1:]),
      attention_masks=attention_masks,
      scale=scale,
      enable_gqa=enable_gqa,
      is_causal=is_causal,
      **kwargs,
  )
  return out_BLHV.reshape(num_tokens, *out_BLHV.shape[2:])


def use_batched_sdpa(model: nn.Module, *, seq_len: int | None = None) -> int:
  """Swaps every `ScaledDotProductInnerAttention` in `model` for batched `[T]`.

  Args:
    model: A built torchtitan model.
    seq_len: Per-sequence token length `S` when inputs are packed `(B * S,)`.

  Returns:
    The number of SDPA modules swapped.
  """
  if seq_len is not None and seq_len < 1:
    raise ValueError(f"seq_len must be >= 1, got {seq_len}")
  count = 0
  for module in model.modules():
    if isinstance(module, common_attention.ScaledDotProductInnerAttention):
      if not hasattr(module, "_orig_forward"):
        module._orig_forward = module.forward  # pylint: disable=protected-access
      module._seq_len = seq_len  # pylint: disable=protected-access
      module.forward = types.MethodType(_batched_sdpa_forward, module)
      count += 1
  return count
