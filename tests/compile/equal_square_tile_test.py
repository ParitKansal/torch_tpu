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

"""Tests for the equal-tile detection used by the TPU flex_attention fold."""

from absl.testing import absltest
import torch
from torch_tpu._internal.compile import flex_attention_redirect
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils

_detect = flex_attention_redirect.equal_square_tile_size


def _tril(n: int) -> torch.Tensor:
  return torch.ones(n, n, dtype=torch.bool).tril()


def _band(n: int, window: int) -> torch.Tensor:
  return _tril(n) & ~torch.ones(n, n, dtype=torch.bool).tril(-window)


def _packed(*tiles: torch.Tensor) -> torch.Tensor:
  return torch.block_diag(*tiles)


class EqualSquareTileSizeTest(seed_test_utils.RepeatableTest):

  def test_equal_causal_documents(self):
    self.assertEqual(_detect(_packed(*[_tril(4)] * 4)), 4)

  def test_equal_sliding_window_documents(self):
    self.assertEqual(_detect(_packed(*[_band(4, 2)] * 4)), 4)

  def test_tile_not_aligned_to_power_of_two(self):
    self.assertEqual(_detect(_packed(*[_tril(3)] * 4)), 3)

  def test_single_document_has_no_tile(self):
    self.assertIsNone(_detect(_tril(16)))

  def test_full_attention_has_no_tile(self):
    self.assertIsNone(_detect(torch.ones(16, 16, dtype=torch.bool)))

  def test_unequal_documents(self):
    self.assertIsNone(_detect(_packed(_tril(6), _tril(10))))

  def test_documents_with_different_content(self):
    odd = _tril(4)
    odd[3, 1] = False
    self.assertIsNone(_detect(_packed(_tril(4), odd, _tril(4), _tril(4))))

  def test_cross_document_attention(self):
    mask = _packed(*[_tril(4)] * 4)
    mask[5, 2] = True
    self.assertIsNone(_detect(mask))

  def test_fold_matches_full_attention(self):
    # The redirect uses tile 0 as the bias for every folded document; check
    # that is numerically the same as attending over the packed sequence.
    docs, doc_len, heads, dim = 4, 8, 2, 16
    mask = _packed(*[_band(doc_len, 3)] * docs)
    tile = _detect(mask)
    self.assertEqual(tile, doc_len)
    q, k, v = (torch.randn(1, heads, docs * doc_len, dim) for _ in range(3))
    full = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask
    )

    def fold(t):
      return t.reshape(heads, docs, doc_len, dim).transpose(0, 1)

    folded = torch.nn.functional.scaled_dot_product_attention(
        fold(q), fold(k), fold(v), attn_mask=mask[:tile, :tile]
    )
    unfolded = folded.transpose(0, 1).reshape(1, heads, docs * doc_len, dim)
    utils.assert_close(unfolded, full)


if __name__ == "__main__":
  absltest.main()
