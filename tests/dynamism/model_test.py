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

"""Model tests with dynamic shapes through torch.compile()."""

from absl.testing import absltest
import torch
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils
import transformers


class ModelTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    if not torch.accelerator.is_available():
      self.skipTest("TPU accelerator not available in this test environment.")
    tt_testing.reset_eager_state()
    torch.compiler.reset()
    self.device = torch.accelerator.current_accelerator()

  def test_gemma4_decode_dynamic(self):
    cfg = transformers.Gemma4TextConfig(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        global_head_dim=128,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
        sliding_window=64,
        use_cache=True,
        dtype="bfloat16",
        layer_types=["sliding_attention", "full_attention"],
    )
    model = transformers.Gemma4ForCausalLM(cfg).to(dtype=torch.bfloat16).eval()
    prompt = torch.randint(0, 64, (1, 16))

    def decode(fn, input_ids):
      tokens = []
      logits = []
      with torch.inference_mode():
        out = fn(input_ids=input_ids, use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        logits.append(out.logits.cpu())
        for _ in range(4):
          tok = torch.argmax(out.logits[:, -1:, :], dim=-1)
          tokens.append(tok.cpu())
          out = fn(
              input_ids=tok,
              past_key_values=past,
              use_cache=True,
              logits_to_keep=1,
          )
          past = out.past_key_values
          logits.append(out.logits.cpu())
      return tokens, logits, past.get_seq_length()

    eager_tokens, eager_logits, eager_seq_len = decode(model, prompt)

    compiled = torch.compile(model.to(self.device), backend="tpu", dynamic=None)
    compiled_tokens, compiled_logits, compiled_seq_len = decode(
        compiled, prompt.to(self.device)
    )

    self.assertEqual(compiled_seq_len, 20)
    self.assertEqual(compiled_seq_len, eager_seq_len)
    # bfloat16 logits differ by up to ~0.025 between eager and compiled on TPU
    # v6e.
    utils.assert_close(compiled_logits, eager_logits, rtol=3e-2, atol=3e-2)
    utils.assert_close(compiled_tokens, eager_tokens)


if __name__ == "__main__":
  absltest.main()
