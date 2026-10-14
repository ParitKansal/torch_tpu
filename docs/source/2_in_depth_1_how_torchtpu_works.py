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
    # How TorchTPU works

    ## TPU primer

    [Google TPUs](https://cloud.google.com/tpu) are specialized ASICs designed
    specifically to perform the massive tensor operations that are the foundation of
    deep learning. TPUs
    [vary in architecture](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm)
    between generations, but broadly defined, a TPU Chip contains:

    * One or more TensorCores, each of which is a single-threaded processor
      comprising matrix-multiply units (MXUs), a vector unit (VPU), and a scalar
      unit.
    * One or more SparseCores, which are specialized for sparse operations and to
      which the TensorCore can offload operations.
    * A large High Bandwidth Memory (HBM) – 288 GB at 8.6 TB/s in
      [TPU 8i](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive)
      which can be used to store ML data and share that with the CPU.
    * Bidirectional Inter-Chip Interconnect (ICI) – 19.2 TBps in
      [TPU 8t and 8i](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/),
      allowing fast communication between TPU chips within user-configured
      [TPU Slices](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm#slices),
      which are logical subgroups of a
      [TPU Pod](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#tpu-pod),
      comprising of a preconfigured number of TPU chips connected in a 3D torus (for
      TPU 7, TPUv 8t) or Boardfly (TPU 8i and TPU v8tx) configuration.
    * Bidirectional Data-Center-Network (DCN) - 400 GBps in
      [TPU 8t](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/),
      allowing communication between TPU chips across Slices.

    TPUs are programmed with the [XLA compiler](https://github.com/openxla/xla)
    stack through two different entry points:
    [StableHLO](https://openxla.org/stablehlo/spec) and
    [Mosaic](https://docs.jax.dev/en/latest/pallas/design/design.html#lowering-pallas-to-mosaic-for-tpu).
    The actual TPU ISA is proprietary, subject to change, and not publicly
    available. Effectively, one can think of StableHLO and (even more so) of Mosaic
    as equivalent to PTX. See picture below for a high-level sketch:

    ![How TorchTPU Works](../images/torch_tpu_inputs_to_tpu.svg)

    A C++ program can access a local TPU by instantiating a
    [PjRT Client](https://openxla.org/xla/pjrt/cpp_api_overview), whose API provides
    methods to compile MLIR (StableHLO or Mosaic) into a TPU executable, to launch
    that executable, and to copy data to-from the TPU. Note that TPU execution is
    asynchronous and the CPU can wait for a TPU result to be ready by waiting on the
    associated PjRTBuffer. StableHLO follows the
    [SSA Paradigm](https://en.wikipedia.org/wiki/Static_single-assignment_form) and,
    hence, doesn't allow for in-place mutations, i.e., a write operation always
    produces a new value (at least conceptually). This is particularly significant
    for TorchTPU, which provides familiar Pytorch "view" semantics, including
    mutations, through view decomposition and inversion. The XLA compiler assumes
    that tensor shapes are known at compile time. This constraint is relaxed in
    TorchTPU by [bounded dynamism](./3_practical_examples_6_bounded_dynamism.md).
    Recognizing this as a limitation for PyTorch workloads, we have been adding
    support for bounded dynamism, i.e., when the maximum bounds of a shape's
    dimensions are known at compile time, but the actual dimensions are known only
    at runtime.

    ## TorchTPU: An out of tree backend with autoload support

    Following
    [current best practice](https://docs.pytorch.org/tutorials/advanced/extend_dispatcher.html)
    where additional backends are constructed out of tree and registered at runtime,
    we extend the PyTorch dispatcher with a new C++ backend, registered as
    `PrivateUse1`. Additionally, we have used the standard `at::Tensor` mechanism to
    define a device-specific implementation of tensor storage (by using the
    `c10::UniqueVoidPtr` inside `c10::StorageImpl`).

    We designed TorchTPU with usability in mind. When the user imports PyTorch on a
    TPU machine where the TorchTPU wheel is installed, Python will autoload the
    TorchTPU wheel. Autoloading the wheel sets up the backend and replaces the
    `PrivateUse1` name with `tpu`. This makes a TorchTPU program no different in
    simple example from the in-tree backends, and a PyTorch program can run on the
    TPU with little-to-no change:

    ```python
    import torch
    import torch.distributed as dist

    def main():
        device = torch.device("tpu")
        dist.init_process_group(backend="tpu_dist")
        rank = dist.get_rank()
        word_size = dist.get_world_size()
        ...
        model = model.to(device)
        ...
    ```

    Because the TPU becomes the current accelerator and `tpu_dist` is the default
    backend for TorchTPU, they don't need to be specified explicitly:

    ```python
    import torch
    import torch.distributed as dist

    def main():
        device = torch.accelerator.current_accelerator()
        dist.init_process_group()
        rank = dist.get_rank()
        word_size = dist.get_world_size()
        ...
        model = model.to(device)
        ...
    ```

    ## Capturing the graph

    TorchTPU traces eager ops at the ATen level, relying on the PyTorch op dispatch
    hooks that are part of the backend infrastructure. TorchTPU ops are constructed
    as StableHLO programs, and passed to XLA singly, or in groups, allowing the
    compiler to perform significant fusion of operations to more effectively utilize
    the hardware. When TorchTPU receives a callback invocation from the PyTorch
    dispatcher the runtime:

    * Creates a Deferred Op instance that contains:
      * A StableHLO builder object, which when invoked returns a StableHLO function
        implementing the semantics of `aten::mul`, as shown below (note that any
        implicit broadcasting and dtype promotion are handled elsewhere)

    ```
    func.func @main(%arg0: tensor<4x16xbfloat16>, %arg1: tensor<4x16xbfloat16>) -> tensor<4x16xbfloat16> {
      %0 = stablehlo.multiply %arg0, %arg1 : tensor<4x16xbfloat16>
      return %0 : tensor<4x16xbfloat16>
    }
    ```

    * An optional map of static op parameters (e.g., training and momentum for
      `aten::native_batch_norm`) to be used for the computation of a cache key;
    * Return a new `at::Tensor` with `c10::StorageImpl` pointing (through
      indirection) to that Deferred Op instance.

    Upon reaching an appropriate synchronization point, dependent on the execution
    mode, TorchTPU will:

    * Analyze the region of ops;
    * Determine which ops to materialize into executable code;
    * Create a compilation cache key for the graph;
    * If it already exists in the cache, fetch and dispatch the executable on the
      TPU;
    * If it does not exist in the cache, asynchronously compile the object, add it
      to the cache and dispatch for execution;
    * Potentially wait for the value to be ready and copy it back, if that was
      appropriate for the synchronization point reached.

    We discuss mode details on the various modes of execution in
    [TorchTPU execution model](./2_in_depth_3_torchtpu_execution_modes.md).

    ## The TorchTPU compilation cache

    To avoid unnecessary compilations, TorchTPU employs a multi-tiered compilation
    cache:

    * **Tier 1:** Process-local in-memory cache for sharing compilation within the
      same process. This is the fastest of the three tiers. It is always enabled.
    * **Tier 2:** Machine-local shared memory that enables cross-process sharing on
      the same host. This is useful for multi-device training/inference. Tier-2 is
      enabled by default for distributed programs, but for single-device programs it
      offers no benefit so is disabled.
    * **Tier 3:** Persistent cache backed by a remote file system (e.g., Google
      Cloud Storage), allowing reuse across hosts, programs, and job restarts. This
      is useful when running TPU programs in execution environments like Colab and
      data centers. This is disabled by default but can be enabled by specifying a
      storage location.

    Each entry is a compiled TPU executable representing a StableHLO computation
    graph generated from an ATen op or a sequence of ATen ops. The StableHLO graphs
    are represented by fingerprints combining the structure of the graph, the tensor
    shapes of the inputs/outputs, and the compiler options including the versions of
    the TPU hardware, libtpu, and the PjRt runtime.

    ## `torch.compile` support

    To maintain a user experience consistent with other hardware supporting
    `torch.compile()` we register a new TorchDynamo
    [backend](https://docs.pytorch.org/docs/stable/torch.compiler_custom_backends.html),
    which can be used as follows: `compiled_model = torch.compile(model, backend="tpu")`.

    Dynamo invokes the TorchTPU backend by passing `torch.fx.GraphModule` plus a
    list of example input tensors. At that point the runtime traces the provided
    graph as if we were in eager mode, but with all ops deferred and no execution on
    the TPU. The runtime generates and compiles a single StableHLO program for the
    entire FX graph and returns a Python callable that wraps the invocation. Note
    that TorchTPU does not use `TorchInductor`, instead reusing the mature XLA
    compiler stack that has developed alongside the TPU architecture.

    ## More detail

    In the following documents we'll go into more detail about the:

    * [Memory model](./2_in_depth_2_torchtpu_and_tpu_memory_model.md) supported by
      the TPU and what its implications are on PyTorch;
    * The [Execution Modes](./2_in_depth_3_torchtpu_execution_modes.md) TorchTPU
      supports;
    * Support for [custom kernels](./2_in_depth_4_custom_kernels.md);
    * [Distributed execution](./2_in_depth_5_distributed.md) on clusters of TPUs.

    """)
  return


if __name__ == "__main__":
  app.run()
