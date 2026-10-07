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

"""Simple end-to-end demo of Muse Glimmer 30B prefill and decode with StaticCache."""

from collections.abc import Mapping
import time
from typing import Any

import torch
from torch._functorch._aot_autograd import autograd_cache
import checkpoint_utils
import kv_cache
import model
import summary_utils
import transformers

_PROMPT = "The future of artificial intelligence is"


def model_generate(
    model: Any,
    config: model.Config,
    initial_inputs: torch.Tensor,
    tokenizer: transformers.PreTrainedTokenizerBase | None,
    prefix: str = "",
) -> tuple[str, Mapping[str, Any]]:
  """Generates tokens using MuseGlimmer30B with StaticCache."""
  num_decode_steps = config.num_decode_steps
  with torch.inference_mode():
    past_key_values = kv_cache.StaticCache(
        num_layers=config.num_layers,
        batch_size=initial_inputs.shape[0],
        num_kv_heads=config.num_kv_heads,
        max_cache_len=config.prefill_seq_len + num_decode_steps,
        head_dim=config.dhead,
        dtype=torch.bfloat16,
        device=initial_inputs.device,
        swa_size=config.swa_size,
        swa_layers=config.swa_layers(),
    )

    prefill_start_time = time.perf_counter()
    logits, past_key_values = model(
        initial_inputs,
        past_key_values=past_key_values,
    )
    torch.accelerator.synchronize()
    prefill_end_time = time.perf_counter()

    prefill_time_ms = (prefill_end_time - prefill_start_time) * 1000
    print(f"{prefix}Prefill done in {prefill_time_ms:.2f} ms", flush=True)
    generated_tokens = [initial_inputs]
    steps_completed = 0
    decode_step_times_ms = []

    for _ in range(num_decode_steps):
      # Greedy sampling
      next_token = torch.argmax(logits, dim=-1)
      generated_tokens.append(next_token)

      step_start_time = time.perf_counter()
      logits, past_key_values = model(
          next_token,
          past_key_values=past_key_values,
      )
      torch.accelerator.synchronize()
      step_end_time = time.perf_counter()

      step_time_ms = (step_end_time - step_start_time) * 1000
      decode_step_times_ms.append(step_time_ms)
      steps_completed += 1
      print(
          f"{prefix}Decode step {steps_completed}/{num_decode_steps} done in"
          f" {step_time_ms:.2f} ms",
          flush=True,
      )

    output_tokens = torch.cat([t.to("cpu") for t in generated_tokens], dim=1)

    avg_decode_time_no_warmup_ms = None

    # First 3 decode steps are considered warmup.
    decode_warmup_steps = 3
    if steps_completed > decode_warmup_steps:
      non_warmup_times = decode_step_times_ms[decode_warmup_steps:]
      avg_decode_time_no_warmup_ms = sum(non_warmup_times) / len(
          non_warmup_times
      )

    avg_decode_time_ms = None
    if steps_completed > 0:
      avg_decode_time_ms = sum(decode_step_times_ms) / len(decode_step_times_ms)

  if tokenizer is not None:
    output_text = str(
        tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    )
  else:
    generated_ids = output_tokens[0, initial_inputs.shape[1] :].tolist()
    output_text = str(generated_ids)
  print(f"{prefix}Prompt: {_PROMPT}", flush=True)
  print(f"{prefix}Output: {output_text}", flush=True)

  metrics = {
      "prefill_time_ms": prefill_time_ms,
      "decode_step_times_ms": decode_step_times_ms,
      "avg_decode_time_per_token_ms": avg_decode_time_ms,
      "avg_decode_time_per_token_no_warmup_ms": avg_decode_time_no_warmup_ms,
  }
  return output_text, metrics


def run_worker(
    device: str,
    *,
    use_random_weights: bool = False,
    prefill_seq_len: int | None = None,
    num_decode_steps: int | None = None,
):
  """Initializes Muse Glimmer 30B and runs prefill + decode with StaticCache.

  Args:
    device: Target accelerator device ("tpu" or "cuda").
    use_random_weights: If True, initializes the model with random weights and
      uses synthetic random input token IDs instead of loading the checkpoint
      and tokenizer.
    prefill_seq_len: Target prefill sequence length. When set with a tokenizer,
      repeats and truncates the tokenized prompt to `prefill_seq_len` tokens;
      when `use_random_weights=True`, generates `prefill_seq_len` random tokens
      (falling back to `model.Config().prefill_seq_len` if None).
    num_decode_steps: Number of decode steps to run. If None, falls back to the
      default `num_decode_steps` in `model.Config()`.
  """
  torch._dynamo.reset()
  autograd_cache.AOTAutogradCache.clear()
  compile_backend = "tpu" if device == "tpu" else "inductor"
  torch._logging.set_logs()  # pylint: disable=protected-access
  transformers.logging.set_verbosity_error()
  torch.manual_seed(42)
  torch.set_default_dtype(torch.bfloat16)

  tokenizer = None if use_random_weights else checkpoint_utils.load_tokenizer()
  if tokenizer is not None:
    prompt_ids = tokenizer(_PROMPT, add_special_tokens=False).input_ids
    if tokenizer.bos_token_id is not None:
      prompt_ids = [tokenizer.bos_token_id] + prompt_ids
    if prefill_seq_len is not None:
      repeats = (prefill_seq_len + len(prompt_ids) - 1) // len(prompt_ids)
      prompt_ids = (prompt_ids * repeats)[:prefill_seq_len]
    inputs = torch.tensor([prompt_ids], dtype=torch.long)
  else:
    default_cfg = model.Config()
    seq_len = (
        prefill_seq_len
        if prefill_seq_len is not None
        else default_cfg.prefill_seq_len
    )
    inputs = torch.randint(
        0,
        default_cfg.vocab_size,
        (1, seq_len),
        dtype=torch.long,
    )

  config_kwargs: dict[str, Any] = {"prefill_seq_len": inputs.shape[1]}
  if num_decode_steps is not None:
    config_kwargs["num_decode_steps"] = num_decode_steps
  config = model.Config(**config_kwargs)
  print(f"Model config: {config}", flush=True)

  with torch.device(device):
    muse_model = model.MuseGlimmer30B(config)

  if use_random_weights:
    model.init_nn_module(muse_model)
  else:
    checkpoint_utils.load_weights_from_checkpoint(muse_model)

  torch.accelerator.synchronize()

  muse_model_compiled = torch.compile(
      muse_model,
      backend=compile_backend,
      dynamic=False,
  )

  label = f"Compiled Static {device.upper()}"
  _, metrics = model_generate(
      muse_model_compiled,
      config,
      inputs.to(device),
      tokenizer,
      prefix=f"[{label}] ",
  )

  print(
      f"\n=== Results for batch_size={inputs.shape[0]},"
      f" prefill_seq_len={inputs.shape[1]} ===",
      flush=True,
  )
  summary_utils.print_summary_table({label: metrics}, config.num_decode_steps)
