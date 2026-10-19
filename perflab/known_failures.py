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

"""Perflab model benchmarks that are known to fail.

Benchmark runners skip the executions that match an entry in `KNOWN_FAILURES`
the same way as the run modes in `BenchmarkSpec.skipped_run_modes`:
`torch_benchmark_test` and `torchax_benchmark_test` apply `--skip_behavior` to
both (e.g. `--skip_behavior=run_skipped` runs only the skipped executions to
check whether they still fail), and `distributed_runner` always skips them.
"""

import dataclasses

from perflab import common
from perflab import mode as mode_lib
from perflab import target as target_lib


@dataclasses.dataclass(frozen=True, kw_only=True)
class KnownFailure:
  """A model benchmark known to fail under specific execution coordinates.

  Attributes:
    benchmark: Registered `BenchmarkSpec.name` in `perflab/models` or
      `perflab/distributed_models`.
    framework: Framework (`TORCH` for TorchTPU/GPU or `TORCHAX` for TorchAX).
    device_kind: Optional device kind filter (e.g. `TPU` or `CUDA`). `None`
      matches all device kinds.
    run_modes: Optional set of `RunMode` values that fail. `None` matches all
      run modes.
    reason: Human-readable explanation of why the benchmark fails.
  """

  benchmark: str
  framework: mode_lib.Framework
  device_kind: target_lib.DeviceKind | None = None
  run_modes: frozenset[common.RunMode] | None = None
  reason: str

  def matches(
      self,
      benchmark: str,
      *,
      framework: mode_lib.Framework,
      device_kind: target_lib.DeviceKind,
      run_mode: common.RunMode,
  ) -> bool:
    """Returns whether this entry applies to the given execution coordinates."""
    if self.benchmark != benchmark or self.framework != framework:
      return False
    if self.device_kind is not None and self.device_kind != device_kind:
      return False
    if self.run_modes is not None and run_mode not in self.run_modes:
      return False
    return True


KNOWN_FAILURES: tuple[KnownFailure, ...] = (
    # TorchTPU.
    #
    # TODO(b/571695076): Remove once the 4096 parameter training step fits.
    #
    # The eager-default, eager-optimized and compiled training steps exceed the
    # XLA:TPU entry computation parameter limit.
    KnownFailure(
        benchmark="synthetic_4096_params_training",
        framework=mode_lib.Framework.TORCH,
        device_kind=target_lib.DeviceKind.TPU,
        run_modes=frozenset({
            common.RunMode.EAGER_DEFAULT,
            common.RunMode.EAGER_OPTIMIZED,
            common.RunMode.COMPILED,
        }),
        reason=(
            "the fused forward, backward and AdamW step needs more XLA"
            " parameters (~20.5k) than supported (~16.3k)"
        ),
    ),
    # TODO(b/574632209): Remove once XLA:TPU cl/1001433059 lands. It fixes
    # `TpuConvolutionRewriter` for non-`f0b_i0o->0bf` dimension orders.
    # Repro: //perflab:distributed_dev_benchmarks_ws8
    #   --test_filter=qwen3_8_27b_4_layers_simple_fsdp_training_compiled
    KnownFailure(
        benchmark="qwen3_8_27b_4_layers_simple_fsdp_training",
        framework=mode_lib.Framework.TORCH,
        device_kind=target_lib.DeviceKind.TPU,
        run_modes=frozenset({common.RunMode.COMPILED}),
        reason=(
            "XLA:TPU HLO shape mismatch in qwen3_8_L147_split1_bwd: Expected"
            " instruction to have shape equal to bf16[2048,2048,2048], actual"
            " shape is bf16[2048,2048,1] on multiply(%slice, %silu_backward)"
            " (b/574632209)"
        ),
    ),
    # TODO(b/574560944): Remove once XLA:TPU cl/1000409835 lands. It fixes the
    # HloDeduplicationKey hash collision from cl/999454734 (b/573904712).
    #
    # The ragged-dot custom fusions of the MoE experts fail to compile on v7
    # with a `program_name_collision` RET_CHECK.
    KnownFailure(
        benchmark="gemma_4_26b_a4b_6_layers_training",
        framework=mode_lib.Framework.TORCH,
        device_kind=target_lib.DeviceKind.TPU,
        run_modes=frozenset({common.RunMode.COMPILED}),
        reason=(
            "XLA:TPU program_name_collision: HloDeduplicationKey hash collision"
            " between ragged-dot fusions"
        ),
    ),
    KnownFailure(
        benchmark="gpt_oss_20b_4_layers_training",
        framework=mode_lib.Framework.TORCH,
        device_kind=target_lib.DeviceKind.TPU,
        run_modes=frozenset({common.RunMode.COMPILED}),
        reason=(
            "XLA:TPU program_name_collision: HloDeduplicationKey hash collision"
            " between ragged-dot fusions"
        ),
    ),
    # TODO(b/568037427): Remove the TorchAX entries as their failures are fixed.
    #
    # `torch._grouped_mm`, used by the transformers MoE experts, has no
    # TorchAX lowering.
    KnownFailure(
        benchmark="gemma_4_26b_a4b_6_layers_forward",
        framework=mode_lib.Framework.TORCHAX,
        reason=(
            "aten::_grouped_mm has no lowering (on v5e_1x1, HBM OOM while"
            " moving the model to the device)"
        ),
    ),
    KnownFailure(
        benchmark="gemma_4_26b_a4b_6_layers_training",
        framework=mode_lib.Framework.TORCHAX,
        reason="aten::_grouped_mm has no lowering",
    ),
    KnownFailure(
        benchmark="gpt_oss_120b_2_layers_training",
        framework=mode_lib.Framework.TORCHAX,
        reason="aten::_grouped_mm has no lowering",
    ),
    KnownFailure(
        benchmark="gpt_oss_120b_4_layers_forward",
        framework=mode_lib.Framework.TORCHAX,
        reason="aten::_grouped_mm has no lowering",
    ),
    KnownFailure(
        benchmark="gpt_oss_20b_4_layers_training",
        framework=mode_lib.Framework.TORCHAX,
        reason="aten::_grouped_mm has no lowering",
    ),
    KnownFailure(
        benchmark="gpt_oss_20b_forward",
        framework=mode_lib.Framework.TORCHAX,
        reason="aten::_grouped_mm has no lowering",
    ),
    KnownFailure(
        benchmark="nemotron_3_nano_30b_6_layers_forward",
        framework=mode_lib.Framework.TORCHAX,
        reason="aten::_grouped_mm has no lowering",
    ),
    KnownFailure(
        benchmark="qwen3_coder_30b_moe_forward",
        framework=mode_lib.Framework.TORCHAX,
        reason="aten::_grouped_mm has no lowering",
    ),
    KnownFailure(
        benchmark="qwen3_5_397b_4_layers_forward",
        framework=mode_lib.Framework.TORCHAX,
        reason="aten::_grouped_mm has no lowering",
    ),
    # The model forward needs a concrete value from a traced tensor (for a
    # shape or Python control flow), which fails under `jax.jit`.
    KnownFailure(
        benchmark="bart_large_mnli_forward_gen",
        framework=mode_lib.Framework.TORCHAX,
        reason=(
            "torch.unique_consecutive() on a traced EOS count in"
            " modeling_bart.py"
        ),
    ),
    KnownFailure(
        benchmark="clap_htsat_fused_forward_gen",
        framework=mode_lib.Framework.TORCHAX,
        reason="torch.where() with a data-dependent size in modeling_clap.py",
    ),
    KnownFailure(
        benchmark="gemma_4_e4b_it_forward_gen",
        framework=mode_lib.Framework.TORCHAX,
        reason=(
            "boolean indexing with a traced pooler_mask in modeling_gemma4.py"
        ),
    ),
    KnownFailure(
        benchmark="glm_ocr_forward_gen",
        framework=mode_lib.Framework.TORCHAX,
        reason="grid_thw.tolist() in transformers get_vision_position_ids()",
    ),
    KnownFailure(
        benchmark="janus_pro_1b_forward_gen",
        framework=mode_lib.Framework.TORCHAX,
        reason=(
            "aten::_is_all_true in torch_compilable_check on a traced image"
            " token count in modeling_janus.py"
        ),
    ),
    KnownFailure(
        benchmark="lahja_sa_ahmad_v1_forward_gen",
        framework=mode_lib.Framework.TORCHAX,
        reason="torch.arange(predicted_lengths.max()) in modeling_vits.py",
    ),
    KnownFailure(
        benchmark="llava_1_5_7b_hf_forward_gen",
        framework=mode_lib.Framework.TORCHAX,
        reason=(
            "torch_compilable_check and masked_scatter on a traced image mask"
            " in modeling_llava.py"
        ),
    ),
    KnownFailure(
        benchmark="sam3_litetext_s0_forward_gen",
        framework=mode_lib.Framework.TORCHAX,
        reason=(
            "torch.arange() with a traced height in modeling_sam3_lite_text.py"
        ),
    ),
    KnownFailure(
        benchmark="vjepa2_vitl_fpc64_256_forward_gen",
        framework=mode_lib.Framework.TORCHAX,
        reason=(
            "Tensor.repeat() with a traced patch count in modeling_vjepa2.py"
        ),
    ),
    # Out of memory.
    KnownFailure(
        benchmark="gemma_4_31b_6_layers_forward",
        framework=mode_lib.Framework.TORCHAX,
        reason="host OOM while moving the model to the device",
    ),
    # TODO(b/571695076): Remove once the 4096 parameter training step compiles
    # within the subprocess timeout.
    KnownFailure(
        benchmark="synthetic_4096_params_training",
        framework=mode_lib.Framework.TORCH,
        device_kind=target_lib.DeviceKind.CUDA,
        run_modes=frozenset({common.RunMode.COMPILED}),
        reason=(
            "Inductor compilation of the training step exceeds the 900 s"
            " subprocess timeout on b200_1"
        ),
    ),
    # TODO(b/563472806): Remove once internal builds pick up
    # https://github.com/llvm/llvm-project/pull/225808.
    #
    # The LLVM of internal builds expands a vectorized integer division by a
    # constant into a v2i16 smul_lohi or umul_lohi that its NVPTX backend cannot
    # select, so Triton aborts while compiling an Inductor kernel of these
    # models. The Triton wheel used in OSS is not affected.
    # copybara:uncomment_begin(internal builds only)
    # KnownFailure(
        # benchmark="blip_image_captioning_base_train_gen",
        # framework=mode_lib.Framework.TORCH,
        # device_kind=target_lib.DeviceKind.CUDA,
        # run_modes=frozenset({common.RunMode.COMPILED}),
        # reason="LLVM cannot select v2i16 [su]mul_lohi in a Triton kernel",
    # ),
    # KnownFailure(
        # benchmark="conformer_training",
        # framework=mode_lib.Framework.TORCH,
        # device_kind=target_lib.DeviceKind.CUDA,
        # run_modes=frozenset({common.RunMode.COMPILED}),
        # reason="LLVM cannot select v2i16 [su]mul_lohi in a Triton kernel",
    # ),
    # KnownFailure(
        # benchmark="dinov2_small_forward_gen",
        # framework=mode_lib.Framework.TORCH,
        # device_kind=target_lib.DeviceKind.CUDA,
        # run_modes=frozenset({common.RunMode.COMPILED}),
        # reason="LLVM cannot select v2i16 [su]mul_lohi in a Triton kernel",
    # ),
    # KnownFailure(
        # benchmark="dinov2_small_train_gen",
        # framework=mode_lib.Framework.TORCH,
        # device_kind=target_lib.DeviceKind.CUDA,
        # run_modes=frozenset({common.RunMode.COMPILED}),
        # reason="LLVM cannot select v2i16 [su]mul_lohi in a Triton kernel",
    # ),
    # KnownFailure(
        # benchmark="dpt_hybrid_midas_forward_gen",
        # framework=mode_lib.Framework.TORCH,
        # device_kind=target_lib.DeviceKind.CUDA,
        # run_modes=frozenset({common.RunMode.COMPILED}),
        # reason="LLVM cannot select v2i16 [su]mul_lohi in a Triton kernel",
    # ),
    # KnownFailure(
        # benchmark="gepard_1_0_forward_gen",
        # framework=mode_lib.Framework.TORCH,
        # device_kind=target_lib.DeviceKind.CUDA,
        # run_modes=frozenset({common.RunMode.COMPILED}),
        # reason="LLVM cannot select v2i16 [su]mul_lohi in a Triton kernel",
    # ),
    # KnownFailure(
        # benchmark="gepard_1_0_train_gen",
        # framework=mode_lib.Framework.TORCH,
        # device_kind=target_lib.DeviceKind.CUDA,
        # run_modes=frozenset({common.RunMode.COMPILED}),
        # reason="LLVM cannot select v2i16 [su]mul_lohi in a Triton kernel",
    # ),
    # KnownFailure(
        # benchmark="glm_ocr_forward_gen",
        # framework=mode_lib.Framework.TORCH,
        # device_kind=target_lib.DeviceKind.CUDA,
        # run_modes=frozenset({common.RunMode.COMPILED}),
        # reason="LLVM cannot select v2i16 [su]mul_lohi in a Triton kernel",
    # ),
    # KnownFailure(
        # benchmark="glucose_base_ja_v2_train_gen",
        # framework=mode_lib.Framework.TORCH,
        # device_kind=target_lib.DeviceKind.CUDA,
        # run_modes=frozenset({common.RunMode.COMPILED}),
        # reason="LLVM cannot select v2i16 [su]mul_lohi in a Triton kernel",
    # ),
    # KnownFailure(
        # benchmark="gpt2_train_gen",
        # framework=mode_lib.Framework.TORCH,
        # device_kind=target_lib.DeviceKind.CUDA,
        # run_modes=frozenset({common.RunMode.COMPILED}),
        # reason="LLVM cannot select v2i16 [su]mul_lohi in a Triton kernel",
    # ),
    # KnownFailure(
        # benchmark="lahja_sa_ahmad_v1_forward_gen",
        # framework=mode_lib.Framework.TORCH,
        # device_kind=target_lib.DeviceKind.CUDA,
        # run_modes=frozenset({common.RunMode.COMPILED}),
        # reason="LLVM cannot select v2i16 [su]mul_lohi in a Triton kernel",
    # ),
    # KnownFailure(
        # benchmark="qwen3_8_27b_4_layers_simple_fsdp_training",
        # framework=mode_lib.Framework.TORCH,
        # device_kind=target_lib.DeviceKind.CUDA,
        # run_modes=frozenset({common.RunMode.COMPILED}),
        # reason="LLVM cannot select v2i16 [su]mul_lohi in a Triton kernel",
    # ),
    # KnownFailure(
        # benchmark="sam3_litetext_s0_forward_gen",
        # framework=mode_lib.Framework.TORCH,
        # device_kind=target_lib.DeviceKind.CUDA,
        # run_modes=frozenset({common.RunMode.COMPILED}),
        # reason="LLVM cannot select v2i16 [su]mul_lohi in a Triton kernel",
    # ),
    # KnownFailure(
        # benchmark="siglip_base_forward",
        # framework=mode_lib.Framework.TORCH,
        # device_kind=target_lib.DeviceKind.CUDA,
        # run_modes=frozenset({common.RunMode.COMPILED}),
        # reason="LLVM cannot select v2i16 [su]mul_lohi in a Triton kernel",
    # ),
    # KnownFailure(
        # benchmark="timm_resnet_50_training",
        # framework=mode_lib.Framework.TORCH,
        # device_kind=target_lib.DeviceKind.CUDA,
        # run_modes=frozenset({common.RunMode.COMPILED}),
        # reason="LLVM cannot select v2i16 [su]mul_lohi in a Triton kernel",
    # ),
    # KnownFailure(
        # benchmark="timm_vit_small_dino_forward",
        # framework=mode_lib.Framework.TORCH,
        # device_kind=target_lib.DeviceKind.CUDA,
        # run_modes=frozenset({common.RunMode.COMPILED}),
        # reason="LLVM cannot select v2i16 [su]mul_lohi in a Triton kernel",
    # ),
    # KnownFailure(
        # benchmark="timm_vit_small_dino_training",
        # framework=mode_lib.Framework.TORCH,
        # device_kind=target_lib.DeviceKind.CUDA,
        # run_modes=frozenset({common.RunMode.COMPILED}),
        # reason="LLVM cannot select v2i16 [su]mul_lohi in a Triton kernel",
    # ),
    # KnownFailure(
        # benchmark="tiny_doc_qa_vision_encoder_decoder_forward_gen",
        # framework=mode_lib.Framework.TORCH,
        # device_kind=target_lib.DeviceKind.CUDA,
        # run_modes=frozenset({common.RunMode.COMPILED}),
        # reason="LLVM cannot select v2i16 [su]mul_lohi in a Triton kernel",
    # ),
    # KnownFailure(
        # benchmark="wan_2_2_ti2v_5b_training",
        # framework=mode_lib.Framework.TORCH,
        # device_kind=target_lib.DeviceKind.CUDA,
        # run_modes=frozenset({common.RunMode.COMPILED}),
        # reason="LLVM cannot select v2i16 [su]mul_lohi in a Triton kernel",
    # ),
    # KnownFailure(
        # benchmark="yolos_small_forward_gen",
        # framework=mode_lib.Framework.TORCH,
        # device_kind=target_lib.DeviceKind.CUDA,
        # run_modes=frozenset({common.RunMode.COMPILED}),
        # reason="LLVM cannot select v2i16 [su]mul_lohi in a Triton kernel",
    # ),
    # copybara:uncomment_end
)


def get_known_failure_reason(
    benchmark: str,
    *,
    framework: mode_lib.Framework,
    device_kind: target_lib.DeviceKind,
    run_mode: common.RunMode,
) -> str | None:
  """Returns the failure reason if the benchmark is known to fail, else None."""
  for failure in KNOWN_FAILURES:
    if failure.matches(
        benchmark,
        framework=framework,
        device_kind=device_kind,
        run_mode=run_mode,
    ):
      return failure.reason
  return None
