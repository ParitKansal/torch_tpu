# LLM inference for Muse Glimmer 30B

This directory provides a self-contained implementation and benchmarking harness
for single-device LLM inference with the
[Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) model on
Google Cloud TPUs (TPU v7x / Ironwood).

## Overview

This example demonstrates LLM prefill and decode using `torch.compile()` with
TorchTPU on a single TPU device:

*   **Dynamic KV cache**: Demonstrates `DynamicCache` paired with
    `torch._dynamo.mark_dynamic()`. By marking the cache sequence dimension
    dynamic before decode steps, `torch.compile` generates a single dynamic FX
    graph on the very first decode step. This completely eliminates
    recompilations across subsequent decode steps.
*   **Static KV cache**: Demonstrates `StaticCache` with fixed buffer sizing and
    tensor padding for static execution profiles.
*   **Sliding window attention**: Supports hybrid full-context and
    sliding-window attention layers within both `DynamicCache` and
    `StaticCache`.
*   **Checkpoint loading**: Supports both Hugging Face Hub weights and random
    weight initialization for hermetic benchmarking.

## Run the example

### Run from the command line

You can execute the benchmarks directly from a TPU VM or inside a TorchTPU
Docker container using `launch.sh`:

```bash
# Run both dynamic and static benchmarks:
bash ./launch.sh
```

Run only the dynamic inference benchmark:

<!-- doc-test: single_tpu_inference_dynamic_doc_test -->

```bash
bash ./launch.sh dynamic
```

Run only the static inference benchmark:

<!-- doc-test: single_tpu_inference_static_doc_test -->

```bash
bash ./launch.sh static
```

Other options:

```bash
# Run benchmarks using random weights (skips checkpoint download):
bash ./launch.sh --random-weights

# Run a specific test method:
bash ./launch.sh static \
  --test_filter=SingleTpuInferenceStaticTest.test_muse_glimmer_static_swa_prefill
```

`launch.sh` automatically configures performance flags in `LIBTPU_INIT_ARGS`
(`--xla_tpu_dvfs_p_state=7`) and selects the appropriate test filter.

The KV cache unit tests need neither a TPU nor the checkpoint. Run the tests for
the KV caches and `mark_dynamic`:

<!-- doc-test: kv_cache_doc_test -->

```bash
python3 ./kv_cache_test.py
```

Run the sliding window attention cache tests:

<!-- doc-test: swa_cache_doc_test -->

```bash
python3 ./swa_cache_test.py
```

### Run on GKE

The included [batchv1_job.yaml](./batchv1_job.yaml) provides a reference
Kubernetes `Job` manifest for running inference on Google Kubernetes Engine
(GKE).

1.  Update the nodepool selector in `batchv1_job.yaml` to match your cluster's
    TPU nodepool:

    ```yaml
    nodeSelector:
      cloud.google.com/gke-nodepool: torchtpu-v7x-4chip-pool
    ```

2.  Submit the job to your cluster:

    ```bash
    kubectl apply -f batchv1_job.yaml
    ```

3.  Inspect the job logs:

    ```bash
    kubectl logs -f job/muse-glimmer-inference
    ```
