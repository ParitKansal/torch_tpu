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

"""HuggingFace text encoder and encoder-decoder benchmark definitions."""

from typing import Any

import torch
from perflab import context as context_lib
from perflab import model_helpers
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import torch_device_ops


def _modify_config_hook(config: Any) -> Any:
  if hasattr(config, "use_cache"):
    config.use_cache = False
  return config


def _load_hf_text_model(
    ctx: context_lib.Context,
    model_name: str,
    batch_size: int,
    seq_len: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
  """Loads a HuggingFace text model and generates token inputs.

  Encoder-decoder models get decoder inputs of the same sequence length.

  Args:
    ctx: The benchmark context.
    model_name: HuggingFace model identifier.
    batch_size: Number of sequences per step.
    seq_len: Tokens per sequence.

  Returns:
    A tuple of (model, inputs dictionary).
  """
  return model_helpers.load_hf_llm(
      model_name,
      dtype=torch_device_ops.get_torch_dtype(ctx.dtype),
      device_str=ctx.device_kind.value,
      is_training=False,
      seq_len=seq_len,
      batch_size=batch_size,
      modify_config_hook=_modify_config_hook,
  )


# Full-scope batch sizes are the smallest powers of two (at most 64) at which
# compiled device time is at least 85% of the step time on v7_1x1x1 (80% if 85%
# is not reached), as for the generated benchmarks.


@registry_lib.register_benchmark(
    stepper=step_lib.StepperType.FORWARD,
    presubmit=registry_lib.PresubmitPolicy.RUN,
)
def bert_base_forward(ctx: context_lib.Context):
  """Benchmark factory for BERT-Base (uncased) forward."""
  if ctx.run_scope == context_lib.RunScope.PRESUBMIT:
    batch_size = 1
    seq_len = 128
  else:
    batch_size = 16
    # 512 is the maximum number of BERT positions.
    seq_len = 512
  model, inputs = _load_hf_text_model(
      ctx,
      "EhsanAghazadeh/bert-base-uncased-random-weights",
      batch_size=batch_size,
      seq_len=seq_len,
  )
  return model, (), inputs
