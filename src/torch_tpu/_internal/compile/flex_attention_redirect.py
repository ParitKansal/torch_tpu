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

"""Runs PyTorch flex_attention on TPU tensors.

Upstream flex_attention rejects TPU tensors and has no TPU kernel for its
higher-order op. `patch()` lets TPU tensors through and redirects the common
case (no score_mod, no LSE requested) to scaled_dot_product_attention with an
additive bias built from the BlockMask, which TorchTPU lowers to its flash
attention kernel. Everything else falls back to upstream's eager math path.
"""

# pylint: disable=protected-access

import torch
from torch._higher_order_ops.flex_attention import flex_attention_backward
from torch.nn.attention import flex_attention
import torch.nn.functional as F


def equal_square_tile_size(mask: torch.Tensor) -> int | None:
  """Returns the document length d if `mask` folds, else None.

  `mask` is a [T, T] bool attention mask. It folds if it is block-diagonal
  with T // d identical d x d tiles, i.e. T // d equal-length documents that
  each use the same within-document mask and never see each other.
  The candidate d is the first token that cannot see the token before it
  (the start of the second document). The fold is accepted only if the
  top-left d x d tile repeated along the diagonal reproduces `mask` exactly.
  The guess can miss a fold (e.g. anti-causal patterns, where every token
  hides its predecessor) but never accepts a mask that would change results.
  """
  size = mask.shape[0]
  idx = torch.arange(1, size, device=mask.device)
  boundaries = (~mask[idx, idx - 1]).nonzero()
  if boundaries.numel() == 0:
    return None
  tile = int(boundaries[0]) + 1
  if size % tile:
    return None
  expected = torch.block_diag(*[mask[:tile, :tile]] * (size // tile))
  return tile if torch.equal(expected, mask) else None


def patch() -> None:
  """Patches PyTorch flex_attention to support TPU tensors."""
  original_validate = flex_attention._validate_device

  def _validate_device(query, key, value):
    if query.device.type != "tpu":
      original_validate(query, key, value)

  flex_attention._validate_device = _validate_device

  original_from_kv_blocks = flex_attention.BlockMask.from_kv_blocks.__func__

  @classmethod
  def _from_kv_blocks_tpu(cls, *args, **kwargs):
    # On TPU, a mask that is block-diagonal with identical tiles (equal-length
    # packed documents) is tagged with the tile size so attention can run per
    # document (see `_flex_attention_tpu`). Runs eagerly at mask creation.
    block_mask = original_from_kv_blocks(cls, *args, **kwargs)
    kv_num_blocks = block_mask.kv_num_blocks
    q_len, kv_len = block_mask.seq_lengths
    if (
        kv_num_blocks.device.type == "tpu"
        and not torch.compiler.is_compiling()
        and not isinstance(kv_num_blocks, torch._subclasses.FakeTensor)
        and tuple(kv_num_blocks.shape[:2]) == (1, 1)
        and q_len == kv_len
        and block_mask.mask_mod is not flex_attention.noop_mask
    ):
      with torch.no_grad():
        mask = flex_attention.create_mask(
            block_mask.mask_mod,
            1,
            1,
            q_len,
            kv_len,
            device=kv_num_blocks.device,
        )
        doc_len = equal_square_tile_size(mask[0, 0])
      if doc_len is not None:
        block_mask.__dict__["_tpu_doc_len"] = doc_len
    return block_mask

  flex_attention.BlockMask.from_kv_blocks = _from_kv_blocks_tpu

  dispatch_key = torch._C.DispatchKey.CompositeExplicitAutograd
  original_bwd = flex_attention_backward.py_kernels.pop(dispatch_key)

  @flex_attention_backward.py_impl(dispatch_key)
  def _sdpa_dense_backward(query, key, value, *args, **kwargs):
    with query.device:
      return original_bwd(query, key, value, *args, **kwargs)

  original_flex_attention = flex_attention.flex_attention

  def _get_sdpa_attn_bias(
      block_mask: flex_attention.BlockMask,
      q_len: int,
      kv_len: int,
      dtype: torch.dtype,
      device: torch.device,
  ) -> torch.Tensor:
    # Cached on the BlockMask. Under torch.compile Dynamo records the cache
    # write as a side effect, so the bias is built in-graph once and later
    # graphs read it as an input.
    cache = block_mask.__dict__.setdefault("_tpu_sdpa_mask_cache", {})
    key = (q_len, kv_len, dtype, device)
    if key not in cache:
      b_mask, h_mask = block_mask.kv_num_blocks.shape[:2]
      with torch.no_grad():
        bool_mask = flex_attention.create_mask(
            block_mask.mask_mod,
            b_mask,
            h_mask,
            q_len,
            kv_len,
            device=device,
        )
        attn_bias = torch.zeros(
            bool_mask.shape, dtype=dtype, device=device
        ).masked_fill_(~bool_mask, float("-inf"))
      cache[key] = attn_bias
    return cache[key]

  def _flex_attention_tpu(
      query: torch.Tensor,
      key: torch.Tensor,
      value: torch.Tensor,
      score_mod=None,
      block_mask=None,
      scale=None,
      enable_gqa: bool = False,
      return_lse: bool = False,
      kernel_options=None,
      *,
      return_aux=None,
  ):
    need_aux = return_lse or (
        return_aux is not None
        and (
            getattr(return_aux, "lse", False)
            or getattr(return_aux, "max_scores", False)
        )
    )
    if (
        query.device.type == "tpu"
        and score_mod is None
        and not need_aux
        and query.dtype in (torch.bfloat16, torch.float32)
    ):
      q_len = query.shape[-2]
      kv_len = key.shape[-2]
      # Set when the BlockMask was created (`_from_kv_blocks_tpu`): the mask is
      # block-diagonal with identical `doc_len` tiles, so attention runs per
      # tile with the tiles folded into the batch dim.
      doc_len = getattr(block_mask, "_tpu_doc_len", None)
      num_docs = 1
      if doc_len and query.shape[0] == 1:
        if q_len != kv_len or q_len % doc_len:
          raise ValueError(
              f"_tpu_doc_len={doc_len} must divide q_len=kv_len"
              f" ({q_len}, {kv_len})."
          )
        num_docs = q_len // doc_len
      attn_bias = (
          _get_sdpa_attn_bias(
              block_mask,
              q_len // num_docs,
              kv_len // num_docs,
              query.dtype,
              query.device,
          )
          if block_mask is not None
          else None
      )
      if num_docs > 1:

        def fold(
            tensor: torch.Tensor,
        ) -> torch.Tensor:  # [1,H,T,D] -> [N,H,T/N,D]
          return tensor.reshape(
              tensor.shape[1], num_docs, doc_len, -1
          ).transpose(0, 1)

        query, key, value = fold(query), fold(key), fold(value)
      out = F.scaled_dot_product_attention(
          query,
          key,
          value,
          attn_mask=attn_bias,
          dropout_p=0.0,
          is_causal=False,
          scale=scale,
          enable_gqa=enable_gqa,
      )
      if num_docs > 1:
        out = out.transpose(0, 1).reshape(1, -1, q_len, out.shape[-1])
      if return_aux is not None:
        return out, flex_attention.AuxOutput()
      return out
    return original_flex_attention(
        query,
        key,
        value,
        score_mod=score_mod,
        block_mask=block_mask,
        scale=scale,
        enable_gqa=enable_gqa,
        return_lse=return_lse,
        kernel_options=kernel_options,
        return_aux=return_aux,
    )

  flex_attention.flex_attention = _flex_attention_tpu
