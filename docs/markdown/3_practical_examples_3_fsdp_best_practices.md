---
torch_tpu_version: 0.1.1.dev20260922
---

# FSDP best practices

Fully Sharded Data Parallel (FSDP) shards model parameters, gradients, and
optimizer state across data-parallel ranks, so that each TPU core stores only a
`1 / world_size` slice of the model. Before a layer runs, its full weights are
reconstructed with an **all-gather**; during backward, gradients are combined
and re-sharded with a **reduce-scatter**.

This guide explains the recommended way to do FSDP on TorchTPU and points to
runnable examples in
[`examples/fsdp`](https://github.com/google-pytorch/torch_tpu/tree/main/examples/fsdp).

In this guide, you will learn:

1.  **Which FSDP approach to choose** on TorchTPU.
2.  **SimpleFSDP (recommended)**: compile-first FSDP with a single wrapping
    call.
3.  **Manual FSDP**: putting collectives directly in the model definition.
4.  **Best practices** that apply to every approach.
5.  **Running the examples**: launch commands and what each example checks.
<!---->
## 1. Choosing an FSDP approach

On TPU, performance comes from compiling the whole training step into a single
XLA program with `torch.compile(backend="tpu")`. When the XLA compiler sees all
forward, backward, and collective operations at once, it can schedule
collectives to overlap with computation and combine small collectives when
profitable.

Choose an FSDP approach based on how well it composes with whole-graph
compilation:

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Approach | When to use it | Example |
| :------- | :------------- | :------ |
| **SimpleFSDP** (`simple_fsdp.data_parallel`) | **Recommended default.** You want FSDP or HSDP without changing the model definition.| [`fsdp_to_simple_fsdp/`](https://github.com/google-pytorch/torch_tpu/tree/main/examples/fsdp/fsdp_to_simple_fsdp) |
| **Manual FSDP** (manual collectives) | You own the model code and want the communication code next to the computation for performance tuning. | [`manual/`](https://github.com/google-pytorch/torch_tpu/tree/main/examples/fsdp/manual) |
| **PyTorch FSDPv2** (`fully_shard`) | Existing code that has not been migrated yet. It works, but its runtime hooks cause graph breaks.| [Migration guide](https://github.com/google-pytorch/torch_tpu/blob/main/examples/fsdp/fsdp_to_simple_fsdp/fsdp_to_simple_fsdp.md) |
<!-- mdformat on -->

SimpleFSDP and manual FSDP both express the collectives as part of the traced
graph, so XLA can optimize them together with the rest of the step. PyTorch
FSDPv2 (`fully_shard`) relies on Python pre-forward and post-backward hooks.
These hooks split the step into many smaller graphs, even when the model is
compiled.

For a side-by-side comparison of compiled FSDPv2 and compiled SimpleFSDP, see
the
[FSDPv2 to SimpleFSDP tutorial](https://github.com/google-pytorch/torch_tpu/blob/main/examples/fsdp/fsdp_to_simple_fsdp/fsdp_to_simple_fsdp.md).
<!---->
## 2. Recommended: SimpleFSDP

[`simple_fsdp`](https://github.com/google-pytorch/torch_tpu/blob/main/examples/fsdp/simple_fsdp.py)
is a compile-first FSDP implementation. It was introduced in TorchTitan, and a
self-contained copy that depends only on PyTorch is provided in
`examples/fsdp/simple_fsdp.py`.

SimpleFSDP stores each parameter as a sharded `DTensor`. When a parameter is
accessed, a parametrization calls `DTensor.redistribute` to all-gather it. The
autograd backward of `redistribute` performs the gradient reduce-scatter. There
are no runtime hooks, so the whole step compiles without graph breaks. The whole
model is sharded with a single `simple_fsdp.data_parallel(...)` call, with no
per-layer wrapping or prefetch tuning, and HSDP is supported with
`mode="hybrid_shard"` on a 2D `DeviceMesh`.

To learn how to use SimpleFSDP, see:

*   [SimpleFSDP README](https://github.com/google-pytorch/torch_tpu/blob/main/examples/fsdp/fsdp_to_simple_fsdp/README.md):
    quick start and available options.
*   [FSDPv2 to SimpleFSDP tutorial](https://github.com/google-pytorch/torch_tpu/blob/main/examples/fsdp/fsdp_to_simple_fsdp/fsdp_to_simple_fsdp.md):
    how to migrate existing `fully_shard` code.
<!---->
## 3. Alternative: Manual FSDP

If you control the model definition, you can write FSDP directly with manual
collectives (`torch.distributed._functional_collectives`). Each layer stores
only its weight shard and all-gathers the full weight in forward. Functional
collectives have autograd formulas, so the backward of the all-gather is a
reduce-scatter, and you don't need to write a custom `torch.autograd.Function`.

This keeps the communication code next to the computation code, which can make
performance tuning easier. The tradeoff is that you must change the model
definition.

To learn how to implement manual FSDP, see the
[manual FSDP guide](https://github.com/google-pytorch/torch_tpu/blob/main/examples/fsdp/manual/manual_fsdp.md).
<!---->
## 4. Best practices

These recommendations apply to SimpleFSDP and manual FSDP.

1.  **Compile the entire training step with `torch.compile(backend="tpu")`.** If
    you can't compile the entire step, compile at least the forward, loss, and
    backward together. Compiling only the model module doesn't produce a single
    graph, which limits how well XLA can overlap the collectives with
    computation.
2.  **Use activation checkpointing on each sharded layer.** Wrap each layer's
    forward in `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`.
    This frees the unsharded weights after each layer's forward pass instead of
    keeping them until backward, which gives the ZeRO-3 memory behavior expected
    from FSDP.
3.  **Make sharded dimensions divisible by the mesh size.** Manual FSDP requires
    it. SimpleFSDP replicates parameters that are too small to shard. Use
    `shard_dim` to pick a dimension that divides evenly.
4.  **Seed each rank differently when initializing sharded weights.** If every
    rank initializes its own shard with the same seed, the full weight is the
    same random block repeated across ranks, which can hurt convergence. For
    reproducible, distinct shards, seed each rank with a rank-dependent value,
    such as `torch.manual_seed(base_seed + dist.get_rank())`.
5.  **Keep all ranks running the same program.** Don't put operations that force
    materialization (such as `.item()`) inside rank-specific branches like `if
    rank == 0`.
6.  **Use HSDP when fully sharding over the whole mesh doesn't scale.** Use
    `mode="hybrid_shard"` with a 2D mesh when your ICI slice is too large to
    shard the model effectively, or when sharding across it creates too much
    network traffic. When running across multiple slices, it is almost always
    best to replicate (data-parallel) across the slower DCN and shard within
    each slice.
7.  **Check memory and numerics before scaling up.** Compare against an
    unsharded or FSDPv2 reference with `assert_close`, and measure memory
    retained across the forward-backward boundary. The examples include a parity
    check (`fsdp_to_simple_fsdp_worker.py`) and a memory test
    (`manual_fsdp_memory_test.py`).
<!---->
## 5. Running the examples

Each example comes with a launcher script that runs on 8 TPU cores (for example,
`v6e-8`):

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Example | Launch command | What it checks |
:-------- | :------------- | :------------- |
| [`fsdp_to_simple_fsdp/`](https://github.com/google-pytorch/torch_tpu/tree/main/examples/fsdp/fsdp_to_simple_fsdp) | `./run_fsdp_to_simple_fsdp.sh` | Compiled FSDPv2 and compiled <br> SimpleFSDP have matching losses <br> and predictions, and their step <br> times are reported. |
| [`manual/`](https://github.com/google-pytorch/torch_tpu/tree/main/examples/fsdp/manual) | `./launch.sh` or `python3 launch.py` | Manual FSDP gradients and predictions <br> match a single-device CPU reference. |
| [`manual/`](https://github.com/google-pytorch/torch_tpu/tree/main/examples/fsdp/manual) | `python3 manual_fsdp_memory_test.py` | Unsharded weights are freed after the <br> forward pass. |
<!-- mdformat on -->
