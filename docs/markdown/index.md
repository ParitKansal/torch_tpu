---
torch_tpu_version: 0.1.1.dev20260922
---

<p align="center">
  <!-- This image will ONLY show up in GitHub's dark mode -->
 <img src="../images/logo.png#gh-dark-mode-only" alt="TorchTPU" style="width: 10%;">
   <!-- This image will ONLY show up in GitHub's light mode (and on other platforms) -->
 <img src="../images/logo.png#gh-light-mode-only" alt="TorchTPU" style="width: 10%;">
</p>
<p align="center">
<a href="https://github.com/google-project/torch_tpu"><b>GitHub</b></a> | <a href="https://discuss.pytorch.org/c/tpu/45"><b>Discussion Forum</b></a> | <a href="https://docs.cloud.google.com/tpu/docs/pytorch-stack"><b>PyTorch on TPU</b></a>
</p>

## About

TorchTPU is an open-source, native PyTorch backend for Google Cloud TPUs, developed in collaboration between Google and Meta. With TorchTPU, you can run existing PyTorch models on TPU hardware with minimal code changes and scale training and inference workloads from a single chip to multi-slice TPU pods.

## What are you trying to do today?

<div class="grid cards two-columns" markdown>

-   :material-rocket-launch:{ .lg .middle } **__Get started with TorchTPU__**

    ---

    Provision environments, understand fundamental architectures, and run your first workloads.

    [:octicons-arrow-right-24: Introduction to TorchTPU](./1_basics_1_intro_to_torchtpu.md)

    [:octicons-arrow-right-24: Get started on a TPU VM](./1_basics_2_get_started_on_vm.md)

    [:octicons-arrow-right-24: Get started on GKE](./1_basics_3_get_started_on_gke.md)

    [:octicons-arrow-right-24: Features and roadmap](./6_reference_1_supported_feature_set.md)

    [:octicons-arrow-right-24: API reference](./6_reference_2_torchtpu_api.md)

    [:octicons-arrow-right-24: FAQ](./faq.md)

-   :material-sine-wave:{ .lg .middle } **__Deep dives__**

    ---

    Deep-dive into the mechanics of memory layouts, execution engines, and distributed compilation.

    [:octicons-arrow-right-24: How TorchTPU works](./2_in_depth_1_how_torchtpu_works.md)

    [:octicons-arrow-right-24: The TorchTPU execution model](./2_in_depth_3_torchtpu_execution_modes.md)

    [:octicons-arrow-right-24: Distributed support](./2_in_depth_5_distributed.md)

    [:octicons-arrow-right-24: Custom kernel support (Concept)](./2_in_depth_4_custom_kernels.md)

    [:octicons-arrow-right-24: The TorchTPU memory model](./2_in_depth_2_torchtpu_and_tpu_memory_model.md)

    [:octicons-arrow-right-24: Dynamic shapes support](./2_in_depth_6_dynamic_shapes.md)   

-   :material-book-open-blank-variant:{ .lg .middle } **__Case studies__**

    ---

    Run end-to-end models, leverage advanced scaling APIs, and develop high-performance kernels.

    [:octicons-arrow-right-24: FLUX Optimization](./3_practical_examples_1_flux_optimization.md)

    [:octicons-arrow-right-24: Muse Glimmer 30B](./5_case_studies_2_muse_glimmer_30b.md)

    [:octicons-arrow-right-24: LLM inference end-to-end](./3_practical_examples_1_llm_inference_end_to_end.md)

    [:octicons-arrow-right-24: Using FSDP](./3_practical_examples_3_fsdp_best_practices.md)

-   :material-lightning-bolt:{ .lg .middle } **__Analyze your workload__**

    ---

    Understand performance bottlenecks and optimize your models for TPU execution.

    [:octicons-arrow-right-24: Profiling and benchmarking](./4_debugging_and_performance_analysis_1_numerical_precision.md)

    [:octicons-arrow-right-24: Profile performance with XProf](./4_debugging_and_performance_analysis_2_xprof_for_torchtpu.md)
</div>