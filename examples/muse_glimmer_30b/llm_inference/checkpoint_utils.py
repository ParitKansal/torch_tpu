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

"""Utilities for loading HuggingFace Muse Glimmer 30B weights and tokenizer."""

import time
from etils import epath
import torch
import model
import transformers

DEFAULT_MODEL_NAME = "meta-models/Muse-Glimmer-30B"
LOCAL_TMP_HF_DATA_PATH = epath.Path("/tmp/weights/transformers")


def _resolve_model_path(
    model_name: str = DEFAULT_MODEL_NAME,
) -> epath.Path | str:
  """Resolves the checkpoint source, preferring /tmp over HF Hub.

  Resolution order:
  1. Local `/tmp/weights/transformers/<model_name>` if it exists. Users with
     pre-downloaded weights at a custom location can place or symlink them to
     this `/tmp` path if needed.
  2. Fallback to `model_name` so Hugging Face `from_pretrained` loads from the
     local HF cache (`HF_HOME`) or downloads directly from Hugging Face Hub
     using
     `model_name`.

  Args:
    model_name: Hugging Face repository ID or subdirectory name.

  Returns:
    Existing directory `epath.Path`, or `model_name` string for Hugging Face Hub
    fallback.
  """
  local_path = LOCAL_TMP_HF_DATA_PATH / model_name
  if local_path.exists():
    return local_path
  return model_name


def load_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
) -> transformers.PreTrainedTokenizerBase | None:
  """Loads the HuggingFace Muse Glimmer tokenizer if available."""
  pretrained_source = _resolve_model_path(model_name)
  try:
    return transformers.AutoTokenizer.from_pretrained(pretrained_source)
  except OSError:
    return None


@torch.no_grad()
def copy_params_from_hf_layer(
    src: torch.nn.Module, tgt: torch.nn.Module
) -> None:
  """Copies parameters from a single HF MuseGlimmerTextDecoderLayer to DecoderBlock."""
  src_params = sorted(src.named_parameters(), key=lambda p: p[0])
  tgt_params = sorted(tgt.named_parameters(), key=lambda p: p[0])
  assert len(src_params) == len(
      tgt_params
  ), f"Parameter count mismatch: {len(src_params)} vs {len(tgt_params)}"
  for (sn, sp), (tn, tp) in zip(src_params, tgt_params, strict=True):
    assert sn == tn, f"Parameters must match: {sn}, {tn}"
    if sn.endswith("_layernorm.weight"):
      # HF uses a manually implemented 'centered RMSNorm', correct for that.
      tp.data.copy_(sp.data.to(tp.dtype) + 1.0)
    else:
      tp.data.copy_(sp.data.to(tp.dtype))


@torch.no_grad()
def load_weights_from_hf(
    hf_model: transformers.MuseGlimmerForConditionalGeneration,
    m: model.MuseGlimmer30B,
) -> None:
  """Copies text decoder and LM head weights from HF MuseGlimmer into MuseGlimmer30B."""
  hf_text = hf_model.model.language_model
  m.embed_tokens.weight.data.copy_(
      hf_text.embed_tokens.weight.data.to(m.embed_tokens.weight.dtype)
  )
  for hf_layer, our_layer in zip(hf_text.layers, m.layers, strict=True):
    copy_params_from_hf_layer(hf_layer, our_layer)
  # This is standard RMSNorm.
  m.norm.weight.data.copy_(hf_text.norm.weight.data.to(m.norm.weight.dtype))
  m.lm_head.weight.data.copy_(
      hf_model.lm_head.weight.data.to(m.lm_head.weight.dtype)
  )


class _TextOnlyMuseGlimmer(transformers.MuseGlimmerForConditionalGeneration):
  """MuseGlimmerForConditionalGeneration without vision submodules."""

  def __init__(self, config):
    super().__init__(config)
    del self.model.vision_tower
    del self.model.vision_adapter
    del self.model.vision_projection


@torch.no_grad()
def load_weights_from_checkpoint(
    m: model.MuseGlimmer30B,
    model_name: str = DEFAULT_MODEL_NAME,
) -> None:
  """Loads HuggingFace Muse Glimmer checkpoint into MuseGlimmer30B."""
  pretrained_source = _resolve_model_path(model_name)

  print(
      f"Loading HuggingFace checkpoint from {pretrained_source}...", flush=True
  )
  t0 = time.perf_counter()
  hf_model = _TextOnlyMuseGlimmer.from_pretrained(
      pretrained_source, dtype=torch.bfloat16
  )
  load_weights_from_hf(hf_model, m)
  del hf_model
  if torch.accelerator.is_available():
    torch.accelerator.synchronize()
  elapsed_s = time.perf_counter() - t0
  print(
      f"Loaded HuggingFace checkpoint in {elapsed_s:.2f} s",
      flush=True,
  )
