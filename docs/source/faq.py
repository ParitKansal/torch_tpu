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
    # TorchTPU FAQ

    ## What is the difference between PyTorch/XLA and TorchTPU?

    Pytorch/XLA historically served as a bridging framework, heavily dependent
    on JAX runtime components and specialized APIs. TorchTPU offers a native
    Pytorch backend for [Google Cloud TPUs](https://cloud.google.com/tpu) and
    zero JAX runtime dependencies, and existing PyTorch workloads run out of the
    box with trivial code changes (e.g., `.to("cuda")` → `.to("tpu")`)

    ## What is happening to PyTorch/XLA?

    PyTorch/XLA is deprecated and replaced by
    [TorchTPU](https://github.com/google-pytorch/torch_tpu), which is now
    public.

    ## Is TorchTPU replacing JAX?

    No. TorchTPU is designed to complement JAX by providing a native PyTorch
    entry point to Cloud TPUs. It allows developers with existing PyTorch
    codebases to run directly on TPUs with zero or minimal rewrites, eliminating
    the need to rewrite models in JAX or manage separate codebases for GPU and
    TPU infrastructure.

    * **When to think of TorchTPU?** If you have existing PyTorch codebase, want
      multi-hardware portability (GPU/TPU) without refactoring, using
      PyTorch-native frameworks (e.g., TorchTitan, vLLM, SGLang, Hugging Face),
      or wants rapid GPU-to-TPU migration.

    ## What is the best way for me to connect with TorchTPU developers?

    You can share feedback by emailing
    [torchtpu-feedback@google.com](mailto:torchtpu-feedback@google.com) and we
    welcome contributions directly to the repo:
    [https://github.com/google-pytorch/torch_tpu](https://github.com/google-pytorch/torch_tpu).
    You can also use the tag "TPU" at [discuss.pytorch.org](https://discuss.pytorch.org/).

    ## Will there be any community engagement forums?

    Yes, community engagement is extremely important for us, and we'll be
    sharing details for such forums. Additionally, we plan to host monthly
    office hours.

    ## What performance can I expect when onboarding standard models?

    At the time of writing, we have validated out-of-box execution on 5,000 most
    downloaded Hugging Face models, including:

    * **LLMs:** LLama, DeepSeek, Mistral, Gemma, Qwen, GPT, OPT, BLOOM, Falcon,
      StableML
    * **Audio/Speech:** Whisper, Wav2Vec2, Hubert & MMS
    * **Vision:** DinoV2, SigLIP, CLIP, Swin Transformer & ConvNeXt, ViT, ResNet
      & DETR
    * **Text Encoders & Search:** BERT, ALBERT & Electra, RoBERTa, DeBERTa, T5,
      MT5 & Flan-T5

    For more details, please refer to our published
    [Case Studies](./index.md#case-studies).

    ## What support does TorchTPU provide for kernel development?

    Refer to [TorchTPU Custom Kernel support](./2_in_depth_4_custom_kernels.md).

    ## Which TPU generations are actively supported? What's the plan for future hardware support?

    TorchTPU actively supports v5e, v5p, Trillium (TPU v6e) and Ironwood (TPU
    v7x). Validation and optimization for TPU v8 and future chips will continue
    as part of our roadmap.

    ## Where can we find TorchTPU's supported features & future roadmap?

    See [Supported features](./6_reference_1_supported_feature_set.md) for the current feature set and planned upcoming feature set.

    ## Is there an agent to make my onboarding easier?

    Check out the TPU agent suite (Experimental): [https://github.com/google-pytorch/tpu-agent-suite](https://github.com/google-pytorch/tpu-agent-suite)

    The TPU Agent Suite is an open-source plugin delivering specialized
    sub-agents to automate onboarding, diagnostics, and inference performance
    tuning for PyTorch models on Cloud TPU v6e and v7x.

    ## Can I use TorchTPU with our custom serving solution?

    Yes. While TorchTPU offers out-of-the-box integration with industry-standard serving engines like vLLM and SGLang, custom or in-house serving solutions can integrate with TorchTPU via native PyTorch execution APIs and lower-level compiler backends. For advanced orchestration, customers can also deploy custom serving setups on GKE with llm-d.
    """)
  return


if __name__ == "__main__":
  app.run()
