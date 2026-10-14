---
torch_tpu_version: 0.1.1.dev20260922
---

# TorchTPU custom kernels

While TorchTPU natively accelerates standard PyTorch framework operators
(such as [`torch.nn.functional.scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)),
you may need to integrate custom kernels to accelerate model-specific
operations, fuse operations to reduce memory bandwidth overhead, or
implement novel architectures not yet covered by standard PyTorch operators.

## Choosing an integration approach

TorchTPU provides three ways to use custom kernels on TPU, depending on your
use case:

| Approach | Status | Best For |
| :---- | :---- | :---- |
| **Tokamax** | Production Ready | Drop-in acceleration for common LLM/MoE building blocks without writing kernel code. |
| **Pallas (`torch.tpu.pallas.jax_op`)** | Production Ready | Writing bespoke, maximum-performance TPU kernels in JAX/Pallas and calling them from PyTorch. |
| **Helion** | Experimental | Writing kernels that are portable across accelerators (e.g. GPU, TPU) using a PyTorch-native DSL. |

## 1. Using pre-built kernels from Tokamax

Before writing a custom kernel from scratch, check if the operator is
available in [Tokamax](http://openxla.org/tokamax)—a library of
high-performance TPU kernels with native PyTorch interoperability.

### When to use

Use Tokamax when your model requires standard high-performance building
blocks, including:

* **Mixture-of-Experts (MoE):** Grouped matrix multiplication, token gathers, and scatters.
* **Attention Variants:** Paged attention, Gated DeltaNet, Kimi Delta Attention, and Multi-head Latent Attention (MLA).
* **Selection:** Optimized `TopK` computation.

### How it works

Tokamax kernels are authored in JAX/Pallas and backed by integrated
autotuning (such as automated tile-size search), low-overhead XProf
benchmarking, and regression testing across TPU generations. Under the hood,
Tokamax uses `torch.tpu.pallas.jax_op` to handle framework interoperability
transparently and exposes native PyTorch APIs directly to your model.

## 2. Writing Custom Pallas Kernels

If you need a bespoke operation not covered by PyTorch or Tokamax, you can
write a TPU kernel in [Pallas](https://docs.jax.dev/en/latest/pallas/index.html)
and register it as a PyTorch operator using `torch.tpu.pallas.jax_op`.

### How to integrate a Pallas kernel

1. **Define the Pallas kernel** using input and output memory references (`Ref`s).
2. **Wrap the kernel in a JAX function** using `pl.pallas_call` to specify output shapes and dtypes.
3. **Register the JAX function with TorchTPU** via `torch.tpu.pallas.jax_op`.

```python {.marimo}
import jax
from jax.experimental import pallas as pl
import torch

# 1. Define the Pallas kernel
def add_vectors_pallas_kernel(x_ref, y_ref, o_ref):
  x, y = x_ref[...], y_ref[...]
  o_ref[...] = x + y

# 2. Wrap the Pallas kernel into a JAX function
def add_vectors_jax(x: jax.Array, y: jax.Array) -> jax.Array:
  return pl.pallas_call(
      add_vectors_pallas_kernel,
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  )(x, y)

# 3. Wrap it for PyTorch using the functional interface
add_vectors_fn = torch.tpu.pallas.jax_op(
    "pallas::add_vectors",
    add_vectors_jax,
)

# 4. Call it like a standard PyTorch function
x = torch.tensor([1.0, 2.0, 3.0], device="tpu")
y = torch.tensor([4.0, 5.0, 6.0], device="tpu")
z = add_vectors_fn(x, y)
```

## 3. Writing portable kernels with Helion (experimental)

If you want your kernels to be portable across accelerators (e.g. TPU, GPU),
you can use [Helion](https://helionlang.com/), a PyTorch-native DSL for
kernel authoring.

!!! note
    Helion support on TorchTPU is currently **experimental**. For production
    workloads, use Tokamax or `torch.tpu.pallas.jax_op`.

### How to write a Helion kernel

Decorate your function with `@helion.kernel`. Use standard PyTorch
operations for host-side setup, and `hl.tile` to define the device
execution loop over partitioned tiles:

```python {.marimo}
import helion
import helion.language as hl
import torch

# 1. Define the kernel in Helion
@helion.kernel
def add_vectors(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  # Standard PyTorch host-side setup
  x, y = torch.broadcast_tensors(x, y)
  out = torch.empty_like(x)

  # Device execution loop over automatically partitioned tiles
  for tile in hl.tile(out.size()):
    out[tile] = x[tile] + y[tile]

  return out

# 2. Call it like a standard PyTorch function
x = torch.tensor([1.0, 2.0, 3.0], device="tpu")
y = torch.tensor([4.0, 5.0, 6.0], device="tpu")
z = add_vectors(x, y)
```

For more information about Helion, see the
[Helion documentation](https://helionlang.com/), ask questions in the
[PyTorch developer discussion forum](https://dev-discuss.pytorch.org/c/helion/31), or
[report issues on GitHub](https://github.com/pytorch/helion/issues).