---
torch_tpu_version: 0.1.1.dev20260922
---

# TorchTPU supported feature set

## Current supported features

|  | **Validated** | **Experimental** |
| ----- | ----- | ----- |
| **PyTorch Community Engagement** | <ul><li>OSS release for TorchTPU, vLLM, SGLang, TorchTitan, TorchRec</li><li>Accepting external PRs</li><li>Documentation / Tutorials</li><li>CI/CD</li><li>Colab</li></ul> | <ul><li>TPU Agent for ATEN migration</li></ul> |
| **TPU Support** | <ul><li>v5e, v5p, v6e, v7</li></ul> |  |
| **Basic functionality** | <ul><li>Ease of migration</li><li>Mixed eager \+ compiled</li><li>FW \+ BW</li><li>Torch.profile</li><li>Dynamic ops (recompilation)</li></ul> | <ul><li>SymInts</li><li>Jagged tensors</li></ul> |
| **Model Architectures** | <ul><li>Transformer (Dense \+ MoE): Qwen, Llama, Muse Glimmer, Gemma</li><li>Diffusion: WAN, FLUX</li><li>CNN</li></ul> | <ul><li>GLM, Kimi, DeepSeek, GPT OSS, many more (hill climb)</li><li>Mamba</li><li>LSTM/GRU</li><li>DLRM</li></ul> |
| **Kernel Support** | <ul><li>Pallas</li><li>Tokamax</li></ul> | <ul><li>Helion</li></ul> |
| **Quantization** | <ul><li>Full tensor FP8/FP4</li><li>Per-channel FP8</li></ul> |  |
| **Distributed** | <ul><li>Optimal mesh</li><li>SimpleFSDP</li><li>Checkpointing</li><li>Activation offloading</li><li>Multi-slice up to 8K chip</li></ul> | <ul><li>FSDPv2 (low perf)</li><li>MPMD (low perf)</li><li>TorchComm</li></ul> |
| **Ecosystem Integrations** | <ul><li>Hugging Face</li><li>Ray</li><li>vLLM / SGLang</li><li>TorchTitan</li><li>TorchRec</li></ul> |  |
| **Production-ready Solutions** | <ul><li>Inference (multi-host, P/D disagg)</li><li>Fine-tuning (up to \~8K chips)</li></ul> |  |

## Future roadmap

|  | **Soon-ish** | **Later** |
| ----- | ----- | ----- |
| **PyTorch Community Engagement** | <ul><li>More docs / tutorial / blogs</li></ul> | <ul><li>TPU Agent for Kernel migration</li></ul> |
| **TPU Support** | <ul><li>v8i/t/r</li></ul> |  |
| **Basic functionality** | <ul><li>Streams</li><li>Graph capture</li><li>Eager mode with fast startup time</li><li>Dynamic ops (no recompilation)</li></ul> |  |
| **Model Architectures** | <ul><li>Day 1 model support</li></ul> |  |
| **Kernel Support** |  | <ul><li>?</li></ul> |
| **Quantization** | <ul><li>Block-wise FP8/FP4 w/ torchao (peak perf on TPUv8t)</li></ul> |  |
| **Distributed** | <ul><li>FSDPv2 (w/ perf)</li><li>MPMD (w/ perf)</li><li>Training \> 8K chips</li><li>Multi-tier checkpointing</li><li>Goodput</li></ul> | Training \>\> 8K chips  |
| **Ecosystem Integrations** | <ul><li>VeRL</li></ul> |  |
| **Production-ready Solutions** | <ul><li>RL</li></ul> | <ul><li>Pre-training</li></ul> |
