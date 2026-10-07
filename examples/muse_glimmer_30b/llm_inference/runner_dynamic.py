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

"""Simple end-to-end demo of Muse Glimmer 30B prefill and decode with DynamicCache."""

from collections.abc import Mapping
import time
from typing import Any
import warnings

import torch
from torch._functorch._aot_autograd import autograd_cache
import checkpoint_utils
import model
import summary_utils
import transformers

_PROMPT = "The future of artificial intelligence is"


def model_generate(
    model: Any,
    initial_inputs: torch.Tensor,
    tokenizer: transformers.PreTrainedTokenizerBase | None,
    num_decode_steps: int,
    prefix: str = "",
    mark_dynamic: bool = True,
) -> tuple[str, Mapping[str, Any]]:
  """Generates tokens using MuseGlimmer30B with DynamicCache.

  Args:
    model: Compiled MuseGlimmer30B model.
    initial_inputs: Prefill input token IDs of shape (batch_size, seq_len).
    tokenizer: Tokenizer for decoding generated tokens, or None.
    num_decode_steps: Number of decode steps to execute.
    prefix: Logging prefix string.
    mark_dynamic: If True, uses `torch._dynamo.mark_dynamic` to mark the
      DynamicCache KV cache sequence length dimension dynamic.

  Returns:
    Tuple of decoded text and performance metrics mapping.

  Raises:
    ValueError: If initial_inputs.shape[1] < 2 when mark_dynamic=True.
  """
  if mark_dynamic and initial_inputs.shape[1] < 2:
    raise ValueError(
        f"initial_inputs seq_len ({initial_inputs.shape[1]}) must be >= 2 for "
        "mark_dynamic (Dynamo requires dynamic dimension size >= 2)."
    )

  with torch.inference_mode():
    prefill_start_time = time.perf_counter()
    logits, past_key_values = model(initial_inputs)
    torch.accelerator.synchronize()
    prefill_end_time = time.perf_counter()

    prefill_time_ms = (prefill_end_time - prefill_start_time) * 1000
    print(f"{prefix}Prefill done in {prefill_time_ms:.2f} ms", flush=True)

    if mark_dynamic and past_key_values is not None:
      past_key_values.mark_dynamic()

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

    # First 3 decode steps are considered warmup (2 steps if mark_dynamic).
    decode_warmup_steps = 2 if mark_dynamic else 3
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
  print(f'{prefix}Prompt: "{_PROMPT}"', flush=True)
  print(f'{prefix}Output: "{output_text}"', flush=True)

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
    enable_bounded_dynamism: bool = True,
    mark_dynamic: bool = True,
    prefill_seq_len: int | None = None,
    num_decode_steps: int | None = None,
    num_dynamic_buckets: int | None = None,
    bucket_size: int | None = None,
):
  """Initializes Muse Glimmer 30B and runs prefill + decode with DynamicCache.

  Args:
    device: Target accelerator device ("tpu" or "cuda").
    use_random_weights: If True, initializes the model with random weights and
      uses synthetic random input token IDs instead of loading the checkpoint
      and tokenizer.
    enable_bounded_dynamism: If True, compiles with `dynamic=None` (bounded
      dynamism); otherwise compiles with `dynamic=False`.
    mark_dynamic: If True and enable_bounded_dynamism=True, marks the KV cache
      sequence length dimension as dynamic using mark_dynamic to eliminate
      decode recompilations.
    prefill_seq_len: Target prefill sequence length. When set with a tokenizer,
      repeats and truncates the tokenized prompt to `prefill_seq_len` tokens;
      when `use_random_weights=True`, generates `prefill_seq_len` random tokens
      (falling back to `model.Config().prefill_seq_len` if None).
    num_decode_steps: Number of decode steps to run. If None, falls back to the
      default `num_decode_steps` in `model.Config()`.
    num_dynamic_buckets: Number of dynamic buckets passed in `torch.compile`
      `options` when `device == "tpu"` and `enable_bounded_dynamism=True`. If
      None, omits the option and uses the TPU compiler backend default.
    bucket_size: Bucket step size passed in `torch.compile` `options` when
      `device == "tpu"` and `enable_bounded_dynamism=True`. If None, omits the
      option and uses the TPU compiler backend default.
  """
  torch._dynamo.reset()
  autograd_cache.AOTAutogradCache.clear()
  compile_backend = "tpu" if device == "tpu" else "inductor"
  torch._logging.set_logs()  # pylint: disable=protected-access
  transformers.logging.set_verbosity_error()
  warnings.filterwarnings(
      "ignore",
      category=UserWarning,
      message=r".*FLASH_ATTENTION SDPBackend.*",
  )
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
  device_inputs = inputs.to(device)

  torch._dynamo.config.recompile_limit = 64

  use_mark_dynamic = enable_bounded_dynamism and mark_dynamic
  compile_options: dict[str, Any] = {}
  if enable_bounded_dynamism:
    compile_dynamic = None
    if device == "tpu":
      if num_dynamic_buckets is not None:
        compile_options["num_dynamic_buckets"] = num_dynamic_buckets
      if bucket_size is not None:
        compile_options["bucket_size"] = bucket_size
    if compile_options:
      opts_str = ", ".join(f"{k}={v}" for k, v in compile_options.items())
      label = (
          f"dynamic=None (BD + mark_dynamic, {opts_str})"
          if use_mark_dynamic
          else f"dynamic=None (BD, {opts_str})"
      )
    else:
      label = (
          "dynamic=None (BD + mark_dynamic)"
          if use_mark_dynamic
          else "dynamic=None (BD)"
      )
    prefix = f"[{device.upper()} dynamic=None] "
  else:
    compile_dynamic = False
    label = "dynamic=False"
    prefix = f"[{device.upper()} dynamic=False] "

  print(
      f"\n--- Running with {label} ---",
      flush=True,
  )
  compiled_model = torch.compile(
      muse_model,
      backend=compile_backend,
      dynamic=compile_dynamic,
      options=compile_options or None,
  )
  _, metrics = model_generate(
      compiled_model,
      device_inputs,
      tokenizer,
      config.num_decode_steps,
      prefix=prefix,
      mark_dynamic=use_mark_dynamic,
  )

  print(
      f"\n=== Results for batch_size={inputs.shape[0]},"
      f" prefill_seq_len={inputs.shape[1]} ===",
      flush=True,
  )
  summary_utils.print_summary_table(
      {label: metrics},
      config.num_decode_steps,
  )
