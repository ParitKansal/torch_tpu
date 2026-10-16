---
torch_tpu_version: 0.1.1.dev20260922
---

# Muse Glimmer 30B

In this case study, we will go through our standalone Muse Glimmer 30B example,
which demonstrates a simple training loop for the full model, using FSDP
on 4x Ironwood TPUs.

For information on how to run this example on TPUs in your GKE (Google
Kubernetes Engine) cluster, please refer to example
[README](../../examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop/README.md).
Here we instead explain some of the implementation choices.

The focus of this example is simplicity and readability. It runs on just a
single host with 4 Ironwood TPUs. Clearly, this isn't a realistic pre-training
scale, but our goal is to have a working example you can run, iterate on, and
study at low cost.

This pure PyTorch (no custom kernels) code achieves training speed of about
2400 to 2800 TPS per rank (so 19k to 22k TPS total), at sequence length 5120.
In future updates, we plan to describe more optimizations and performance
best practices when running on TPUs.

## Model definition
[Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) follows
a fairly standard transformer architecture.
It consists of 52 layers of causal self-attention and feed-forward (FFN) blocks.
Every 4-th layer of the model uses full attention, and all the other layers use
sliding-window attention (SWA).

The model code is fully self-contained in [model_fsdp.py](../../examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop/model_fsdp.py)
(and [rope.py](../../examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop/rope.py)
for rotary positional encoding).

Configuration is specified in [model_fsdp.py](../../examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop/model_fsdp.py):

```python
class Config:
  # General model config
  vocab_size: int = 202048
  num_layers: int = 52
  dmodel: int = 6656
  dff: int = 19968
  dhead: int = 128
  num_q_heads: int = 32
  num_kv_heads: int = 2

  # Details
  swa_size: int = 2048
  # ... omitted ...

  # Perf evaluation config
  batch_size: int = 8  # Local (per rank) batch size
  seq_len: int = 5120
```

The definition of the model backbone boils down to:

```python
self.embed_tokens = nn.Embedding(config.vocab_size, config.dmodel)
self.embed_norm = nn.RMSNorm(
    (config.dmodel,), eps=config.rmsnorm_eps, elementwise_affine=False
)
self.layers = nn.ModuleList(
    DecoderBlock(config, layer_idx=i) for i in range(config.num_layers)
)
self.norm = nn.RMSNorm((config.dmodel,), eps=config.rmsnorm_eps)
```

Notice that model head (aka “LM head”) is not present here – in case of
training, it is usually better placed with the loss function, see the section
on chunkwise fused loss below.

## Per-layer torch.compile

TorchTPU performs best with `torch.compile`, thanks to XLA compiler's ability to
perform various optimizations, like op fusions, memory optimization, and
automatically overlap computations with communications.

This example shows how you can apply `torch.compile` on each model layer
separately (as opposed to the whole model). Both approaches work fine, but
whole-model compilation can take quite a bit longer, and performance-wise, at
least for this model, there is no benefit to compiling beyond the scope of a
single layer.

```python
class MuseGlimmer30B(nn.Module):

  def __init__(self, config: Config):
    # ... omitted ...
    self.layers = nn.ModuleList(
        DecoderBlock(config, layer_idx=i) for i in range(config.num_layers)
    ) 

  def apply_torch_compile(self, backend):
    self.embed_norm = torch.compile(self.embed_norm, dynamic=False, backend=backend)
    self.layers = nn.ModuleList(
        torch.compile(layer, dynamic=False, backend=backend)
        for layer in self.layers
    )
    self.norm = torch.compile(self.norm, dynamic=False, backend=backend)
```

Here, `dynamic=False` means no dynamic shapes (all shapes are static during our
training), and the `backend` variable here is just to make switching between
`"tpu"` and `"inductor"` (for CUDA) backends easier.

Internally, TorchTPU will cache compiled layers, and because there are just two
kinds of layers (full and SWA) in this model, this greatly speeds up the
compilation process.

## Activation checkpointing

The model weights take up about 60 GiB (in bf16) of HBM, and would easily fit on
a single Ironwood TPU TensorCore (which has 96 GiB).

However, during training we also need memory for gradients, optimizer states
(e.g. AdamW momentum and variance), but even more importantly – memory usage can
be easily dominated by intermediate activations needed for the autograd
functionality. Activation checkpointing (aka rematerialization) is a very
common technique for addressing the latter.

Because we compile the model per-layer, it is very natural to apply activation
checkpointing at the same granularity. We do this using standard
[`torch.utils.checkpoint`](https://docs.pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint)
functionality:

```python
class DecoderBlock(nn.Module):
  # ... omitted ...

  def forward(self, h: torch.Tensor, rope_sin_cos: tuple) -> torch.Tensor:
    return torch.utils.checkpoint.checkpoint(
        self.forward_impl,
        h,
        rope_sin_cos,
        use_reentrant=False,
    )

  def forward_impl(self, h: torch.Tensor, rope_sin_cos: tuple) -> torch.Tensor:
    h = h + self.post_attention_layernorm(
        self.self_attn(self.input_layernorm(h), rope_sin_cos)
    )
    h = h + self.post_feedforward_layernorm(
        self.mlp(self.pre_feedforward_layernorm(h))
    )
    return h
```

> [!NOTE]
> Activation checkpointing (AC) is applied inside the model layer
(`DecoderBlock` module) `forward` method, and `torch.compile` is applied
outside / on the layer itself – this ordering is important.
You could keep AC inside the layer, and apply `torch.compile` on the whole model
– this would work fine.
But you cannot swap them, e.g. if you were to apply AC the same way,
but apply `torch.compile` inside on attention and FFN sub-layers separately,
it would not work as expected.

`torch.compile` works particularly well with AC in TorchTPU: during the (global)
backward pass, each layer performs a small joint forward+backward pass (due to
rematerialization), and because the TPU backend compiler (XLA) sees the whole
joint forward+backward computation graph, it can do additional optimizations and
even reduce memory usage for intermediate tensors.

## "Naive" FSDP

In this example, we wanted to demonstrate the simplest possible way of doing
FSDP (fully-sharded data-parallel) training, using native PyTorch collective ops
directly.

> [!TIP]
> For an even simpler description of this approach, take a look at our
[manual FSDP example](../../examples/fsdp/manual).

We do FSDP by manually sharding all attention and FFN projection weights, e.g.:

```python
# QKV projections are fully sharded:
self.q_proj = nn.Linear(dm, num_q_heads * dh // world_size, bias=False)
```

> [!NOTE]
> PyTorch `nn.Linear` layer stores its weight matrix transposed, which is why
we shard on the 2nd dimension here. The actual shape of the weight will be
`[num_q_heads * dh / world_size, dm]`. This is important because PyTorch
all-gather / reduce-scatter collectives work on the 1st dimension.

Then, during the forward pass, we use PyTorch's functional collectives to
gather the weights before using them:

```python
# Functional collectives
import torch.distributed._functional_collectives as func_col

# ... all-gather:
q_proj_w = func_col.all_gather_single(self.q_proj.weight, gather_dim=0, group=self.pg)

# ... use full weight directly:
q = F.linear(h, q_proj_w).view(local_bsz, seq_len, num_q_heads, dh)
```

The same idea is applied on all other projections (attention Q/K/V/out/gate,
and FFN up/gate/down).

Critically, functional collectives like `func_col.all_gather_single`, are
registered with PyTorch autograd engine, so during `.backward()` pass we
automatically get corresponding reduce-scatter collectives (i.e. we don't need
to write them ourselves).

You will notice that this implementation is relatively naive, in that it doesn't
attempt to pre-fetch the next layer's weights, while running the current layer.
Would this not cause stalled/idle TPU TensorCores waiting for collectives to
complete?

Because we apply `torch.compile` on the whole layer, TorchTPU's compiler (XLA)
is able to reorder and offload collectives to run them asynchronously on the
SparseCore, both in forward and in joint forward+backward passes – which in this
case turns out to be enough to achieve nearly perfect communications /
computation overlap.

> TODO: add detailed xprof trace of how XLA ends up overlapping
collectives with computations here.

## Chunkwise fused LM head + cross-entropy loss

The example by default uses local batch size 8 and sequence length 5120. If we
were to naively compute the cross-entropy training loss, we would need to
materialize an intermediate logits tensor of shape `[8, 5120, 202048]`, where
the last number is the model's vocabulary size. In bf16 this alone would already
require about 16 GiB of HBM memory, which is too wasteful.

A common solution to this problem is to perform loss computation chunkwise,
fusing the vocabulary matrix weight matmul with the cross-entropy loss. One
approach is to perform chunking along the vocab dimension (which requires
online-softmax and is difficult to express in pure PyTorch), and another
approach is to chunk along the sequence dimension – we do the latter here:

```python
class MuseGlimmerChunkwiseFusedLMHeadLoss(nn.Module):
  # ... omitted ... 

  for i in range(0, h.shape[0], chunk_size):
    h_chunk.data.copy_(h[i : (i + chunk_size)])
    tgt_chunk.data.copy_(tgt[i : (i + chunk_size)])

    self.fused_loss_fn_fwd_bwd(
        h_chunk, tgt_chunk, self.lm_head.weight, alpha, cap, loss_scaler
    )

    h_chunk_grads.append(h_chunk.grad)
    h_chunk.grad = None

  return torch.cat(h_chunk_grads, dim=0).view(bsz, seq_len, dm)
```

The `fused_loss_fn_fwd_bwd` refers to the `torch.compile`-d version of the
static function that computes the fused vocab matmul + cross-entropy loss +
backward pass on a single chunk of the tokens:

```python
@staticmethod
def _chunk_fused_lmhead_loss_fwd_bwd(
    h, tgt, lm_head_w, alpha, cap, loss_scaler
):
  logits = F.tanh(F.linear(h, lm_head_w) * (alpha / cap)) * cap
  loss = F.cross_entropy(input=logits, target=tgt, reduction="sum")
  loss *= loss_scaler
  # Note: in recent versions of PyTorch, .backward is no longer a graph break!
  loss.backward()
  return
```

To take the most benefit from this approach, we even perform the backward pass
on the chunk. In more recent versions of PyTorch, invoking `loss.backward()` no
longer causes a graph-break, which permits TorchTPU's compiler (XLA) to perform
more optimizations and fusions.

## What’s missing in this example

This example illustrates only the basic training loop, and, for clarity, omits
various details important to any practical training setup:

 - data loading, tokenization,
 - model checkpoint loading/saving,
 - job health monitoring, convergence diagnostics,
 - fault-tolerance.

For a realistic production-grade training setup with TorchTPU, we recommend
taking a look at [TorchTitan](https://github.com/google-pytorch/torchtitan/).
