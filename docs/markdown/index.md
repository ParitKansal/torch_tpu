---
torch_tpu_version: 0.1.1.dev20260922
---

# TorchTPU Tutorials & Guides
Explore tutorials, how-to guides, reference specifications, and performance
optimization guides for TorchTPU on Google Cloud TPUs.

## Get started with TorchTPU
*   [Introduction to TorchTPU](./1_basics_1_intro_to_torchtpu.md): Why TorchTPU
    was created, what it provides, and how it can help you.
*   [Get started with TorchTPU on a VM](./1_basics_2_get_started_on_vm.md):
    Setup TorchTPU on a TPU VM.
*   [Get started with TorchTPU on GKE](./1_basics_3_get_started_on_gke.md):
    Use TorchTPU through GKE.
*   [How TorchTPU works](./2_in_depth_1_how_torchtpu_works.md): The basic design top to bottom.
*   [The TorchTPU memory model](./2_in_depth_2_torchtpu_and_tpu_memory_model.md): Using views effectively on TPU.
*   [The TorchTPU execution model](./2_in_depth_3_torchtpu_execution_modes.md): In depth detail of the execution modes TorchTPU supports.
*   [Custom kernel support](./2_in_depth_4_custom_kernels.md): How to write custom kernels for TorchTPU.
*   [Distributed support](./2_in_depth_5_distributed.md): In depth detail of distributed functionality for TorchTPU.
*   [Dynamic shapes support](./2_in_depth_6_dynamic_shapes.md): In-depth concepts and mechanics of how TorchTPU compiles and executes dynamic shapes using bounded dynamism.
*   [Supported feature set](./6_reference_1_supported_feature_set.md):
    Supported feature set table.
*   [torch.tpu API](./6_reference_2_torchtpu_api.md):
    In-depth API and environment variable reference.

## Practical Examples
*   [LLM inference end-to-end](./3_practical_examples_1_llm_inference_end_to_end.md): Simple example of doing an LLM and getting some tokens back.
*   [LLM training end-to-end](./3_practical_examples_2_llm_training_end_to_end.md): Simple example of training an LLM.
*   [Using FSDP](./3_practical_examples_3_fsdp_best_practices.md): Description of recommended approach to FSDP for TorchTPU.
*   [Write custom kernels with Pallas](./3_practical_examples_4_adding_custom_kernels_using_pallas.md): Integration of a pallas kernel.
*   [Write custom communicating kernels with Pallas](./3_practical_examples_5_distributed_pallas_kernels.md): As above, but adding distributed functionality and showing how to map between Pytorch and Pallas.
*   [Reduce recompilation from dynamic shapes](./3_practical_examples_6_bounded_dynamism.md): Hands-on walkthrough applying bounded dynamism to an end-to-end LLM inference decode loop with a dynamic KV cache.
*   [Case study: FLUX Optimization](./5_case_studies_1_flux_optimization.md): Example demonstrating how to identify and apply multiple optimizations on FLUX.
*   [Case study: Muse Glimmer 30B](./5_case_studies_2_muse_glimmer_30b.md): Example of a simple training loop for pure-PyTorch implementation of Muse Glimmer 30B model using FSDP, torch.compile, and activation checkpointing.

## Troubleshoot your workload
*   [Troubleshoot error messages](./4_debugging_and_performance_analysis_3_getting_better_error_information.md):
    How to get better error information.
*   [View compilation artifacts](./4_debugging_and_performance_analysis_4_viewing_compilation_artifacts.md):
    How to dump, view and understand compilation artifacts.


## Analyze performance
*   [Numerical precision and parity](./4_debugging_and_performance_analysis_1_numerical_precision.md):
    Controlling precision in TorchTPU.
*   [Profile performance with XProf](./4_debugging_and_performance_analysis_2_xprof_for_torchtpu.md):
    How to use XProf as a TorchTPU profiler.
