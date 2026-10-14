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
<p align="center">
  <!-- This image will ONLY show up in GitHub's dark mode -->
 <img src="images/logo.png#gh-dark-mode-only" alt="TorchTPU" style="width: 10%;">
   <!-- This image will ONLY show up in GitHub's light mode (and on other platforms) -->
 <img src="images/logo.png#gh-light-mode-only" alt="TorchTPU" style="width: 10%;">
</p>
<p align="center">
<a href="https://github.com/google-project/torch_tpu"><b>GitHub</b></a> | <a href="https://discuss.pytorch.org/c/tpu/45"><b>Discussion Forum</b></a> | <a href="https://docs.cloud.google.com/tpu/docs/pytorch-stack"><b>PyTorch on TPU</b></a>
</p>

## About

TorchTPU is an open-source, native PyTorch backend for Google Cloud TPUs, developed in collaboration between Google and Meta. With TorchTPU, you can run existing PyTorch models on TPU hardware with minimal code changes and scale training and inference workloads from a single chip to multi-slice TPU pods.

## What are you trying to do today?

<div class="grid cards two-columns" markdown>
- :material-rocket-launch:{ .lg .middle } **__Get Started__**
   Provision environments, understand fundamental architectures, and run your first workloads.
   [:octicons-arrow-right-24: Introduction to TorchTPU](markdown/1_basics_1_intro_to_torchtpu.md)
   [:octicons-arrow-right-24: Get started on a VM](markdown/1_basics_2_get_started_on_vm.md)
   [:octicons-arrow-right-24: Get started on GKE](markdown/1_basics_3_get_started_on_gke.md)
   [:octicons-arrow-right-24: How TorchTPU works](markdown/2_in_depth_1_how_torchtpu_works.md)
- :material-sine-wave:{ .lg .middle } **__In-Depth Architecture__**
   Deep-dive into the mechanics of memory layouts, execution engines, and distributed compilation.
   [:octicons-arrow-right-24: The TorchTPU memory model](markdown/2_in_depth_2_torchtpu_and_tpu_memory_model.md)
   [:octicons-arrow-right-24: The TorchTPU execution model](markdown/2_in_depth_3_torchtpu_execution_modes.md)
   [:octicons-arrow-right-24: Distributed support](markdown/2_in_depth_5_distributed.md)
   [:octicons-arrow-right-24: Dynamic shapes support](markdown/2_in_depth_6_dynamic_shapes.md)
- :material-book-open-blank-variant:{ .lg .middle } **__Specifications & API References__**
   Explore the formal API surface mappings, environment flags, and features.
   [:octicons-arrow-right-24: Supported feature set](markdown/6_reference_1_supported_feature_set.md)
   [:octicons-arrow-right-24: torch.tpu API specifications](markdown/6_reference_2_torchtpu_api.md)
- :material-code-brackets:{ .lg .middle } **__Practical Examples__**
   Run end-to-end models, leverage advanced scaling APIs, and develop high-performance kernels.
   [:octicons-arrow-right-24: LLM inference end-to-end](markdown/3_practical_examples_1_llm_inference_end_to_end.md)
   [:octicons-arrow-right-24: LLM training end-to-end](markdown/3_practical_examples_2_llm_training_end_to_end.md)
   [:octicons-arrow-right-24: Scaling with FSDP](markdown/3_practical_examples_3_fsdp_best_practices.md)
   [:octicons-arrow-right-24: Reduce recompilation / Dynamism](markdown/3_practical_examples_6_bounded_dynamism.md)
- :octicons-cpu-16:{ .lg .middle } **__Custom Pallas Kernels__**
   Write hardware-specific TPU kernels directly in Python using Triton-like Pallas abstractions.
   [:octicons-arrow-right-24: Custom kernel support (Concept)](markdown/2_in_depth_4_custom_kernels.md)
   [:octicons-arrow-right-24: Write custom kernels with Pallas](markdown/3_practical_examples_4_adding_custom_kernels_using_pallas.md)
   [:octicons-arrow-right-24: Write communicating distributed kernels](markdown/3_practical_examples_5_distributed_pallas_kernels.md)
- :material-tools:{ .lg .middle } **__Troubleshooting & Debugging__**
   Diagnose compilation crashes, locate execution bottlenecks, and inspect compiled graph artifacts.
   [:octicons-arrow-right-24: Troubleshoot error messages](markdown/4_debugging_and_performance_analysis_3_getting_better_error_information.md)
   [:octicons-arrow-right-24: View compilation artifacts](markdown/4_debugging_and_performance_analysis_4_viewing_compilation_artifacts.md)
- :material-chart-timeline-variant:{ .lg .middle } **__Performance & Optimization__**
   Profile trace timelines using XProf, optimize numerical parity, and review advanced case studies.
   [:octicons-arrow-right-24: Numerical precision and parity](markdown/4_debugging_and_performance_analysis_1_numerical_precision.md)
   [:octicons-arrow-right-24: Profile performance with XProf](markdown/4_debugging_and_performance_analysis_2_xprof_for_torchtpu.md)
   [:octicons-arrow-right-24: Case study: FLUX Optimization](markdown/5_case_studies_1_flux_optimization.md)

</div>

    """)
  return


if __name__ == "__main__":
  app.run()
