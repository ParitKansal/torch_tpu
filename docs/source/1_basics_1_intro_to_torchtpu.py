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

# pylint: skip-file
import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
  import marimo as mo

  return (mo,)


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    # Introduction to TorchTPU

    TorchTPU is an open-source, native PyTorch backend for [Google Cloud TPUs](https://cloud.google.com/tpu), developed in collaboration between Google and the PyTorch team at Meta. With TorchTPU, you can run existing PyTorch models on TPU hardware with minimal code changes and scale training and inference workloads from a single chip to multi-slice TPU pods.

    ## Why use TorchTPU

    Google TPUs are application-specific integrated circuits (ASICs) purpose-built for the matrix and vector operations that power deep learning, offering high throughput, energy efficiency, and performance per dollar at scale. While TPUs have historically been paired with the TensorFlow and JAX ecosystems, TorchTPU brings native TPU support directly to the [PyTorch ecosystem](https://github.com/google-pytorch/torch_tpu) around three design principles:

    * **Native PyTorch citizenship:** TorchTPU integrates directly with PyTorch with no JAX runtime dependencies. It preserves standard PyTorch tensor view and in-place mutation semantics, adopts PyTorch's one-process-per-device execution model, and supports `torch.compile` through Dynamo.
    * **Out-of-the-box usability:** Porting a workload from a GPU to a TPU starts with changing `device="cuda"` to `device="tpu"` and `backend="nccl"` to `backend="tpu_dist"`. Your unmodified PyTorch model runs immediately in eager mode, with clear error messages when diagnosing runtime or compiler issues.
    * **Production-scale performance:** Beyond op-by-op eager execution, TorchTPU introduces a deferred-fusion eager mode, a Dynamo-to-XLA compiler pipeline, bounded dynamism to avoid shape-driven recompilations, and custom kernel APIs to help you reach peak hardware utilization.

    Out of the box usability is a core tenet of TorchTPU. In the following cell you can see a simple example of a PyTorch program running on TPU, and if you run this in colab with the TPU runtime it should print that it is using the TPU:
    """)
  return


@app.cell
def _():
  import torch

  device = torch.device("tpu")
  x = torch.randn((4, 4), device=device)
  y = torch.ones((4, 4), device=device)

  result = torch.matmul(x, y).cpu()
  print(f"Executed on device: {x.device}")
  print(f"Result shape on CPU: {result.shape}, sum: {result.sum():.4f}")
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""

    ## Key features of the current TorchTPU release

    * **Comprehensive operator and model coverage:** Supports approximately 450 core ATen operators (both forward and backward passes) and is continuously validated across nearly 5,000 popular Hugging Face models spanning large language models (Llama, Qwen, Gemma, DeepSeek, Mistral), vision (ViT, ResNet, ConvNeXt, DINOv2), audio (Whisper, Wav2Vec2), and diffusion (FLUX, Stable Diffusion).
    * **Three eager execution modes:**
      * **Strict (default):** Dispatches one ATen operator at a time asynchronously to the TPU, matching standard PyTorch eager behavior on GPUs.
      * **Debug (`TORCH_TPU_LAUNCH_BLOCKING=1`):** Runs each operator synchronously with the CPU and enables bounds checking to simplify debugging.
      * **DeferAndFuse (`TORCH_TPU_DEFER_AND_FUSE=1`):** Defers and fuses sequences of ATen operators up to a materialization point (such as `.cpu()` or `.item()`), reducing High Bandwidth Memory (HBM) traffic and accelerating eager execution.
    * **Graph compilation with `torch.compile`:** Registers the `"tpu"` Dynamo backend (`torch.compile(model)`) to lower entire PyTorch FX graphs into [StableHLO](https://openxla.org/stablehlo/spec) and compile them with the [OpenXLA compiler](https://openxla.org/xla). Active support for dynamic shapes (`torch._dynamo.mark_dynamic`) lets compiled graphs handle varying batch sizes or sequence lengths up to a specified bound without recompiling.
    * **Multi-tier compilation cache:** Eliminates redundant XLA compilations across three cache tiers:
      * **Tier 1:** Process-local in-memory cache (always enabled).
      * **Tier 2:** Host-local shared memory for cross-process reuse in multi-device jobs.
      * **Tier 3:** Persistent remote cache (such as Cloud Storage) for reuse across hosts and job restarts.
    * **Native PyTorch distributed support:** Implements TPU collectives at the `torch.distributed` Process Group (`c10d::Backend`) level via `backend="tpu_dist"`, supporting `DistributedDataParallel` (DDP), `FullyShardedDataParallel` (FSDP2 and SimpleFSDP), and `DTensor`. You can also use `torch.tpu.distributed.topology_aware_mesh` to align parallelism dimensions with physical TPU interconnects.
    * **Custom TPU kernels:** Author custom kernels in [Pallas](https://docs.jax.dev/en/latest/pallas/index.html) (exposed to PyTorch via `torch.tpu.pallas.jax_op`, including direct inter-chip RDMA collectives), write tile-based kernels using the experimental [Helion](https://pytorch.org/blog/helion/) DSL, or import optimized TPU kernels from Tokamax.
    * **Ecosystem integrations:** Works with standard PyTorch tooling and serving/training frameworks, including `torch.profiler` (with TensorBoard and XProf), vLLM, SGLang, TorchTitan, Ray, and TorchRec.

    ## Known limitations of the current TorchTPU release

    As an early product, TorchTPU does currently have known limitations that will be addressed over time. TorchTPU is a collaborative, community-driven effort, and we invite you to join us on this journey. Please review our [feature set and roadmap](./6_reference_1_supported_feature_set.md) to see what is on the horizon, or jump right in by [submitting PRs](https://github.com/google-pytorch/torch_tpu) on GitHub to help us resolve these limitations. This is a community project, and we'd love to hear from you and invite you to collaborate with us via [GitHub](https://github.com/google-pytorch/torch_tpu), [https://discuss.pytorch.org/c/tpu](https://discuss.pytorch.org/c/tpu), [https://dev-discuss.pytorch.org/c/tpu/](https://dev-discuss.pytorch.org/c/tpu/) and [torchtpu-feedback@google.com](mailto:torchtpu-feedback@google.com).

    * Dynamic Shapes cause recompilations
      * `torch.compile` sets `dynamic=False` by default on TorchTPU. `dynamic=None` or `dynamic=True` are not yet supported for production. [See discussion on bounded dynamism](./2_in_depth_6_dynamic_shapes.md)
      * As example, running decode/generate step on Hugging Face LLMs will involve recompilations at each step unless bounded dynamism is enabled.
    * View ops may be expensive on TPU. See [TorchTPU memory model](./2_in_depth_2_torchtpu_and_tpu_memory_model.md)
    * MPMD with collectives is supported for functionality, but it will not reach peak performance yet. Collective overlap with compute has [some constraints](./2_in_depth_5_distributed.md):
      * Full stream support with offload will come in the near future.
      * Use SimpleFSDP with `torch.compile()` as opposed to FSDPv2 with eager or `torch.compile` mode [FSDP best practices](./3_practical_examples_3_fsdp_best_practices.md)
      * Use the `spmd_safe` decorator on regions **users guarantees** to be exactly the same is the recommended way to get peak performance from XLA optimizations on communicating modules
    * TorchTPU does not currently have native FlexAttention support
    * Current generation TPUs do not support fast accumulation in lower precision than fp32.

    **Other recommendations:**

    | Limitation | Recommendation |
    | :--- | :--- |
    | Eager modes have a non-zero warmup time | Enable persistent compilation cache |
    | Torch.compile takes a long time | Enable persistent compilation cache |
    | Only 1 process can access a TPU device | co-location RL is not yet supported |
    | CNN in PyTorch default using channel-first layout is very inefficient on TPU | Use this [PyTorch option](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html) to set a channel last layout for better performance. |
    | Device mesh mapping may be suboptimal by default due to differences in topology between TPUs and other architectures | Use `topology_aware_mesh` for optimal device mesh mapping, and ability to customize device mesh mapping |

    ## Getting started

    To get started, check out the following tutorials:
    * Try TorchTPU [on a VM](./1_basics_2_get_started_on_vm.md) or [through GKE](./1_basics_3_get_started_on_gke.md).
    * Learn in depth about [how TorchTPU works](./2_in_depth_1_how_torchtpu_works.md).
    * Or investigate examples on:
      * [optimizing Flux](./5_case_studies_1_flux_optimization.md)
      * [training Muse Glimmer](./3_practical_examples_2_llm_training_end_to_end.md)
      * the effective [application of FSDP](./3_practical_examples_3_fsdp_best_practices.md)
    """)
  return


if __name__ == "__main__":
  app.run()
