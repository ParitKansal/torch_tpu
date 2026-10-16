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

"""Signature-driven input synthesis.

Inputs are resolved in three tiers:
  1. Signature reflection: every `forward` argument listed in `_RULES` is
     generated from config dimensions.
  2. Modality default: used only when the signature has no named argument
     (e.g. `forward(*args, **kwargs)`).
  3. Quirk builder: a `ModelQuirk.input_builder` runs last and may add or
     replace entries for models whose inputs cannot be derived generically.

Usage, e.g. in a provider's `ModuleSpec.sample_inputs_factory`:

  ctx = inputs.InputContext(config=config, shape=shape, device=device)
  kwargs = inputs.generate(
      model_cls.forward, ctx, modality=modality, builder=builder
  )
  return (), kwargs  # `ModuleSpec` expects (args, kwargs).

`forward` may be the unbound `forward` of the model class: only its signature
is read, so inputs can be generated before the model is instantiated.
`modality` is the model's resolved modality; it is required, although it only
changes the result for opaque signatures. `builder` is the model's
`ModelQuirk.input_builder`, or None for most models.

Defaults match tests/transformers_inputs.py (seed, sequence length, vocab,
EOS) so that generated inputs stay bit-identical across the migration. This
module has no model-specific branches; model-specific logic lives in quirks/.
"""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import inspect
from typing import Any

import torch
from tests.model_registry.core import config_utils
from tests.model_registry.types import Modality

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

_SEED = 42

# Text length.
_MAX_SEQ_LEN = 512
_MAX_SEQ_LEN_HEURISTIC_CAP = 1_000_000
_SEQ_LEN_FIELDS = (  # Fields that bound the text length, in priority order.
    "max_text_len",
    "max_position_embeddings",
    "max_target_positions",
    "n_positions",
    "seq_length",
    "max_seq_len",
    "model_max_length",
)
# Position-embedding sizes that RoBERTa-style models offset by pad_token_id.
_PADDED_POSITION_FIELDS = ("max_position_embeddings", "max_target_positions")

# Defaults when the config does not specify a dimension.
_DEFAULT_VOCAB_SIZE = 32000
_DEFAULT_EOS_TOKEN_ID = 2
_DEFAULT_IMAGE_SIZE = 224
_DEFAULT_NUM_CHANNELS = 3
_DEFAULT_AUDIO_SAMPLES = 16000
_DEFAULT_NUM_MEL_BINS = 80
_DEFAULT_MEL_FRAMES = 3000
_DEFAULT_SAMPLE_SIZE = 32
_DEFAULT_LATENT_CHANNELS = 4
_DEFAULT_TIMESTEP = 1


# -----------------------------------------------------------------------------
# Dimension heuristics. Each mirrors its tests/transformers_inputs.py
# counterpart so that generated inputs stay bit-identical.
# -----------------------------------------------------------------------------


def _vision_sub_config(config: Any) -> Any:
  """Returns the first of `vision_config`, `encoder`, `backbone_config`."""
  for key in ("vision_config", "encoder", "backbone_config"):
    sub = config_utils.get_field(config, key)
    if sub:
      return sub
  return None


def max_seq_len(config: Any) -> int:
  """Returns the largest safe text length for `config`.

  Mirrors `_get_max_seq_len`: the top-level config and its `text_config`,
  `encoder` and `decoder` are each searched for the first bounding field, and
  the smallest bound wins. Models such as RoBERTa offset learned position ids
  by `pad_token_id + 1`, so that offset is subtracted from position-embedding
  sizes to keep ids in range.

  Args:
    config: A raw `config.json` dict or a `transformers.PretrainedConfig`.

  Returns:
    The maximum sequence length, or 512 if the config does not bound it.
  """
  top_pad = config_utils.get_field(config, "pad_token_id")
  configs = [config] + [
      sub
      for key in ("text_config", "encoder", "decoder")
      if (sub := config_utils.get_field(config, key)) is not None
  ]
  bounds = []
  for cfg in configs:
    for key in _SEQ_LEN_FIELDS:
      value = config_utils.as_int(
          config_utils.get_field(cfg, key), default=0, min_value=1
      )
      if not 0 < value < _MAX_SEQ_LEN_HEURISTIC_CAP:
        continue
      if key in _PADDED_POSITION_FIELDS:
        pad = config_utils.as_int(
            config_utils.get_field(cfg, "pad_token_id") or top_pad,
            default=0,
            min_value=0,
        )
        if pad > 0 and value > pad + 1:
          value -= pad + 1
      bounds.append(value)
      break
  return min(bounds) if bounds else _MAX_SEQ_LEN


def image_hw(config: Any) -> tuple[int, int]:
  """Returns the input image `(height, width)` for `config`.

  Lookup order mirrors `_parse_image_size`: the vision sub-config first, then
  the top level, then `input_size` / `pretrained_cfg.input_size` (timm-style).
  A two-element `image_size` such as YOLOS's `[512, 864]` stays non-square,
  as in the legacy vision builder. Model-specific sizes belong in quirks.

  Args:
    config: A raw `config.json` dict or a `transformers.PretrainedConfig`.

  Returns:
    The image height and width.
  """
  vision = _vision_sub_config(config)
  for raw in (
      config_utils.get_field(vision, "image_size"),
      config_utils.get_field(config, "image_size"),
  ):
    if isinstance(raw, (list, tuple)) and raw:
      h = config_utils.as_int(raw[0], min_value=1)
      w = config_utils.as_int(raw[-1], min_value=1)
      if h is not None and w is not None:
        return h, w
    elif (size := config_utils.as_int(raw, min_value=1)) is not None:
      return size, size
  pretrained = config_utils.get_field(config, "pretrained_cfg")
  raw = config_utils.get_field(config, "input_size") or config_utils.get_field(
      pretrained, "input_size"
  )
  if isinstance(raw, (list, tuple)) and raw:
    raw = raw[-1]
  size = config_utils.as_int(raw, default=_DEFAULT_IMAGE_SIZE, min_value=1)
  return size, size


def num_channels(config: Any) -> int:
  """Returns the image channel count; mirrors `_get_num_channels`."""
  vision = _vision_sub_config(config)
  for raw in (
      config_utils.get_field(vision, "num_channels"),
      config_utils.get_field(config, "num_channels"),
  ):
    if (channels := config_utils.as_int(raw, min_value=1)) is not None:
      return channels
  return _DEFAULT_NUM_CHANNELS


# -----------------------------------------------------------------------------
# InputContext
# -----------------------------------------------------------------------------


def _new_generator() -> torch.Generator:
  return torch.Generator(device="cpu").manual_seed(_SEED)


@dataclasses.dataclass(frozen=True)
class InputContext:
  """Everything an input rule or quirk builder needs to build tensors.

  Random tensors are drawn from a CPU generator seeded with 42 and then moved
  to `device`, so the same config yields bit-identical inputs on every run and
  every device (GPU goldens, repeated CPU / TPU runs).

  Attributes:
    config: A raw `config.json` dict or a `transformers.PretrainedConfig`.
    shape: Optional `(batch, seq_len, ...)` override.
    device: Device the tensors are placed on.
    dtype: Floating-point dtype of generated tensors.
    generator: Seeded CPU generator; one per context.
  """

  config: Any
  shape: Sequence[int] | None = None
  device: str | torch.device = "cpu"
  dtype: torch.dtype = torch.float32
  generator: torch.Generator = dataclasses.field(
      default_factory=_new_generator, compare=False, repr=False
  )

  # --- Config-derived dimensions. ---

  @property
  def batch_size(self) -> int:
    return int(self.shape[0]) if self.shape else 1

  def field(self, *keys: str, default: Any = None) -> Any:
    """Returns the first present config field among `keys`."""
    return config_utils.get_first_field(self.config, keys, default)

  def text_len(self) -> int:
    """`min(shape[1], max_seq_len)` if a shape is given, else at most 512."""
    limit = max_seq_len(self.config)
    if self.shape is not None and len(self.shape) > 1:
      return min(int(self.shape[1]), limit)
    return min(limit, _MAX_SEQ_LEN)

  def vocab_size(self) -> int:
    # `src_vocab_size` first: for asymmetric encoder-decoders such as FSMT,
    # `vocab_size` aliases the target vocab, which the encoder cannot embed.
    vocab = self.field("src_vocab_size", "vocab_size")
    return config_utils.as_int(vocab, _DEFAULT_VOCAB_SIZE, min_value=1)

  def decoder_vocab_size(self) -> int:
    # Mirrors legacy `_add_encoder_decoder_inputs`: the target vocab of
    # asymmetric encoder-decoders (e.g. FSMT) can be smaller than the source.
    decoder = config_utils.get_field(self.config, "decoder")
    text = config_utils.get_field(self.config, "text_config")
    for raw in (
        config_utils.get_field(self.config, "tgt_vocab_size"),
        config_utils.get_field(decoder, "vocab_size"),
        config_utils.get_field(self.config, "vocab_size"),
        config_utils.get_field(text, "vocab_size"),
    ):
      if (vocab := config_utils.as_int(raw, min_value=1)) is not None:
        return vocab
    return _DEFAULT_VOCAB_SIZE

  def decoder_len(self) -> int:
    """`text_len()`, further capped by `max_target_positions` (as legacy)."""
    max_target = config_utils.as_int(
        config_utils.get_field(self.config, "max_target_positions"),
        min_value=1,
    )
    length = self.text_len()
    return length if max_target is None else min(length, max_target)

  def eos_token_id(self) -> int:
    eos = self.field("eos_token_id", default=_DEFAULT_EOS_TOKEN_ID)
    if isinstance(eos, (list, tuple)):
      eos = eos[0] if eos else _DEFAULT_EOS_TOKEN_ID
    return config_utils.as_int(eos, _DEFAULT_EOS_TOKEN_ID)

  # --- Tensor factories. Random ones draw from the seeded generator. ---

  def randn(self, *size: int) -> torch.Tensor:
    t = torch.randn(*size, generator=self.generator, dtype=self.dtype)
    return t.to(self.device)

  def randint(self, high: int, *size: int) -> torch.Tensor:
    t = torch.randint(0, high, size, generator=self.generator)
    return t.to(self.device)

  def ones(self, *size: int) -> torch.Tensor:
    return torch.ones(*size, device=self.device, dtype=torch.long)


Inputs = dict[str, torch.Tensor]
InputBuilder = Callable[[InputContext, Inputs], None]
_Rule = Callable[[InputContext], torch.Tensor]


# -----------------------------------------------------------------------------
# Argument rules: one function per supported `forward` argument.
# -----------------------------------------------------------------------------


def _input_ids(ctx: InputContext) -> torch.Tensor:
  """Random token ids that end with EOS, like a real tokenized prompt."""
  ids = ctx.randint(ctx.vocab_size(), ctx.batch_size, ctx.text_len())
  eos = ctx.eos_token_id()
  if 0 <= eos < ctx.vocab_size():
    ids[:, -1] = eos
  return ids


def _decoder_input_ids(ctx: InputContext) -> torch.Tensor:
  return ctx.randint(
      ctx.decoder_vocab_size(), ctx.batch_size, ctx.decoder_len()
  )


def _attention_mask(ctx: InputContext) -> torch.Tensor:
  return ctx.ones(ctx.batch_size, ctx.text_len())


def _pixel_values(ctx: InputContext) -> torch.Tensor:
  h, w = image_hw(ctx.config)
  return ctx.randn(ctx.batch_size, num_channels(ctx.config), h, w)


def _input_values(ctx: InputContext) -> torch.Tensor:
  return ctx.randn(ctx.batch_size, _DEFAULT_AUDIO_SAMPLES)


def _input_features(ctx: InputContext) -> torch.Tensor:
  mel_bins = config_utils.as_int(
      ctx.field("num_mel_bins", "feature_size"),
      _DEFAULT_NUM_MEL_BINS,
      min_value=1,
  )
  max_src = config_utils.as_int(ctx.field("max_source_positions"), min_value=1)
  frames = _DEFAULT_MEL_FRAMES if max_src is None else 2 * max_src
  return ctx.randn(ctx.batch_size, mel_bins, frames)


def _diffusion_sample(ctx: InputContext) -> torch.Tensor:
  size = config_utils.as_int(
      ctx.field("sample_size"), _DEFAULT_SAMPLE_SIZE, min_value=1
  )
  channels = config_utils.as_int(
      ctx.field("in_channels"), _DEFAULT_LATENT_CHANNELS, min_value=1
  )
  return ctx.randn(ctx.batch_size, channels, size, size)


def _timestep(ctx: InputContext) -> torch.Tensor:
  return torch.tensor([_DEFAULT_TIMESTEP], device=ctx.device)


# -----------------------------------------------------------------------------
# Rule tables
# -----------------------------------------------------------------------------

# Maps a `forward` argument name to the rule that builds it. Arguments not
# listed here (inputs_embeds, position_ids, token_type_ids, labels, ...) are
# left unset so that HF falls back to its own defaults. `token_type_ids` and
# `pixel_mask` are deliberately absent: HF defaults them to zeros and ones
# respectively, and the legacy path never generates them, so omitting them
# keeps the generated keys (and GPU golden `inputs_hash`) unchanged.
_RULES: Mapping[str, _Rule] = {
    # Text.
    "input_ids": _input_ids,
    "decoder_input_ids": _decoder_input_ids,
    "attention_mask": _attention_mask,
    # Vision.
    "pixel_values": _pixel_values,
    # Audio.
    "input_values": _input_values,
    "input_features": _input_features,
    # Diffusion.
    "sample": _diffusion_sample,
    "timestep": _timestep,
}

# Masks are only generated together with the tensor they describe. Many
# forwards (e.g. ViT, Whisper) accept `attention_mask` without taking
# `input_ids`; a `[B, S]` token mask would be wrong for them.
_COMPANION_OF: Mapping[str, str] = {
    "attention_mask": "input_ids",
}

# Primary arguments per modality, used when the signature is opaque.
_MODALITY_DEFAULTS: Mapping[Modality, tuple[str, ...]] = {
    Modality.TEXT: ("input_ids", "attention_mask"),
    Modality.CAUSAL_LM: ("input_ids", "attention_mask"),
    Modality.SEQ2SEQ: ("input_ids", "attention_mask", "decoder_input_ids"),
    Modality.VISION: ("pixel_values",),
    Modality.AUDIO: ("input_values",),
    Modality.MULTIMODAL: ("input_ids", "attention_mask", "pixel_values"),
    Modality.DIFFUSION: ("sample", "timestep"),
}


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def forward_arg_names(forward: Callable[..., Any]) -> list[str]:
  """Returns the named (non-variadic) parameters of `forward`."""
  return [
      p.name
      for p in inspect.signature(forward).parameters.values()
      if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
  ]


def _select_args(forward: Callable[..., Any], modality: Modality) -> list[str]:
  """Returns the arguments to generate, in `forward` order (Tiers 1 and 2)."""
  arg_names = forward_arg_names(forward)
  if not arg_names:
    # Tier 2: opaque signature, e.g. `forward(*args, **kwargs)`.
    return list(_MODALITY_DEFAULTS.get(modality, ("input_ids",)))
  # Tier 1. A readable signature with no known argument (e.g. PatchTST's
  # `past_values`) yields nothing here; a quirk builder must supply it.
  names = [n for n in arg_names if n in _RULES]
  return [
      n for n in names if n not in _COMPANION_OF or _COMPANION_OF[n] in names
  ]


def generate(
    forward: Callable[..., Any],
    ctx: InputContext,
    *,
    modality: Modality,
    builder: InputBuilder | None = None,
) -> Inputs:
  """Builds keyword inputs for `forward`.

  Tensors are generated in `forward` argument order, so the seeded random
  stream is consumed deterministically.

  Args:
    forward: The model's forward callable (e.g. `model.forward`).
    ctx: Config, shape, device, dtype and seeded generator.
    modality: The model's resolved modality. Required, so that a caller cannot
      silently fall back to text inputs; it only changes the result when the
      signature is opaque (Tier 2).
    builder: Optional `ModelQuirk.input_builder` for this model (Tier 3).

  Returns:
    A dict of keyword arguments for `forward`.
  """
  inputs = {name: _RULES[name](ctx) for name in _select_args(forward, modality)}
  if builder is not None:
    builder(ctx, inputs)
  return inputs
