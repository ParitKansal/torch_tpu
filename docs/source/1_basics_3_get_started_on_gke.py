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

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
  import marimo as mo

  return (mo,)


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    # Get started with TorchTPU on GKE

    This tutorial walks through running PyTorch workloads on a Google
    Kubernetes Engine (GKE) TPU cluster in three parts:

    1.  **Run a simple single-device example** (`examples/simple_pytorch/simple_pytorch.py`) as a Kubernetes `Job`.
    2.  **Run a multi-device distributed example** (`examples/tensor_parallel/manual/tp_worker.py`) on a single 4-chip host (`Job`) and across a multi-host 64-chip slice (`JobSet`).
    3.  **Run an upstream PyTorch example** from [`pytorch/examples`](https://github.com/pytorch/examples) (such as MNIST).

    We assume you already have a GKE Standard cluster with a TPU node pool; see
    [Deploy TPU workloads in GKE Standard](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/tpus)
    if you need to create one.

    ## Before you begin

    ### Prerequisites

    *   A GKE cluster with a `tpu7x` node pool. This tutorial uses a 4-chip
        (`2x2x1`) node pool.
    *   The [gcloud CLI](https://docs.cloud.google.com/sdk/docs/install-sdk).

    ### Set environment variables

    All steps in this tutorial rely on the following variables; set them
    accordingly. `LOCATION` is the zone or region of your cluster.

    ```bash
    export PROJECT=<YOUR_PROJECT_ID>
    export CLUSTER=<YOUR_CLUSTER_NAME>
    export LOCATION=<YOUR_CLUSTER_ZONE_OR_REGION>
    export NODEPOOL_NAME=<YOUR_4CHIP_NODEPOOL_NAME>
    export DOCKER_IMAGE=us-docker.pkg.dev/ml-oss-artifacts-published/torch-tpu-public-release-container/torch-tpu:latest
    ```

    ### Authenticate and get cluster credentials

    ```bash
    gcloud auth login
    gcloud auth application-default login
    gcloud components install kubectl gke-gcloud-auth-plugin
    gcloud container clusters get-credentials ${CLUSTER} --location=${LOCATION} --project=${PROJECT}
    ```

    See
    [Install kubectl and configure cluster access](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl)
    for details.

    ## Submit workloads with kubectl

    Workloads are submitted as standard Kubernetes `Job` or `JobSet` manifests.
    The Python example scripts are already pre-installed inside the public
    container image at `/workspace/examples/`, while the reference Kubernetes
    `.yaml` manifests
    (`examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml`,
    `examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml`, and
    `examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml`) live in the
    `examples/` directory of the TorchTPU repository. Run the `sed` and
    `kubectl apply` commands below from the root of a local `torch_tpu`
    repository checkout.

    ## Run a single-device workload (one TensorCore of a 4-chip host)

    The manifest `examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml`
    reserves one host with 4 TPU v7x chips. Each v7x chip has two TensorCores,
    and TorchTPU exposes each TensorCore as one device, so the host has 8 TPU
    devices and its slice topology is `2x2x1`:

    ```yaml
        cloud.google.com/gke-tpu-slice-topology: 2x2x1
        ...
        cloud.google.com/gke-tpu-topology: 2x2x1
    ```

    The example runs a single process on a single TPU device (one TensorCore)
    and leaves the other 7 devices on the host idle. This is expressed as a
    `1x1x1x1` TorchTPU topology, where the fourth number is the number of
    TensorCores per chip that the job uses:

    ```yaml
        - name: TORCH_TPU_TOPOLOGY
          value: "1,1,1,1"
    ```

    The container runs `simple_pytorch.py`, which ships in the image under
    `/workspace/examples/`:

    ```yaml
      command:
        - /bin/bash
        - -c
        - |
          set -ex
          export PYTHONUNBUFFERED=1
          export TPU_LAUNCH_BLOCKING=1
          export ABSL_LOGGING_MIN_LOG_LEVEL=0
          echo "=== Starting Single-Device 1-Chip, 1-Device TPU v7x Workload ==="
          cd /workspace/examples/simple_pytorch/
          python3 ./simple_pytorch.py
          echo "=== Single-Host single-device Workload Completed Successfully ==="
    ```

    This is the whole program:
    """)
  return


@app.cell(hide_code=True)
def _():
  import torch

  device = torch.device("tpu")  # Change from cuda to tpu
  print(f"TPU Initialization complete. Primary device: {device}")

  # 1. Create a tensor on the CPU
  cpu_tensor = torch.ones((5, 5))

  # 2. Move it to the TPU
  tpu_tensor = cpu_tensor.to(device)  # Using the device object
  # OR
  tpu_tensor_alt = cpu_tensor.to("tpu")  # Using the "tpu" string

  # 3. Create a tensor directly on the TPU
  direct_tpu_tensor = torch.randn((5, 5), device="tpu")

  print(f"cpu_tensor is on: {cpu_tensor.device}")
  print(f"tpu_tensor is on: {tpu_tensor.device}")
  print(f"direct_tpu_tensor is on: {direct_tpu_tensor.device}")
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    Patch the image and node pool into the manifest and apply it:

    ```bash
    sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
    sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
    kubectl apply -f examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
    ```

    Check job and pod status, then read the logs:

    ```bash
    kubectl get job,pods -n default
    kubectl logs -n default -l job-name=torchtpu-singledevice-1chip-job -f
    ```

    The job runs to completion and prints the four lines shown above.

    ## Run distributed workloads on GKE

    ### Single host, multiple chips (4 chips / 8 devices on one host)

    The first step up from the single-device example is to use every TPU
    device on one host. Submit a single-host Kubernetes `Job` by first updating
    the manifest with your Docker image and node pool name and then launching
    the job using `kubectl`.

    The manifest `examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml`
    applies to a single host of 4 v7x TPUs (8 TensorCores), so the slice
    topology is `2x2x1`:

    ```yaml
        cloud.google.com/gke-tpu-slice-topology: 2x2x1
    ```

    TorchTPU interprets this as a `2x2x1x2` topology because it launches one
    process per TPU TensorCore:

    ```yaml
        - name: TORCH_TPU_TOPOLOGY
          value: "2,2,1,2"
    ```

    It runs the manual tensor-parallel example from the Docker image:

    ```yaml
      command:
        - /bin/bash
        - -c
        - |
          set -ex
          export PYTHONUNBUFFERED=1
          export TPU_LAUNCH_BLOCKING=1
          export ABSL_LOGGING_MIN_LOG_LEVEL=0
          echo "=== Starting Single-Host 4-Chip (8-Device) TPU v7x Workload ==="
          cd /workspace/examples/tensor_parallel/manual/
          bash ./launch_singlehost.sh
          echo "=== Single-Host 4-Chip Workload Completed Successfully ==="
    ```

    `examples/tensor_parallel/manual/launch_singlehost.sh` detects the TPU
    topology, exports it, and starts one process per TPU device with
    `torch.distributed.run` (`torchrun`) running
    `examples/tensor_parallel/manual/tp_worker.py`:

    ```bash
    # Detect and export TPU configuration dynamically.
    eval "$(python3 -m torch.tpu.distributed)"
    export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES

    python3 -m torch.distributed.run --nnodes=1 --node_rank=0 \
        --master_addr=localhost --master_port=29500 \
        --nproc_per_node=8 ./tp_worker.py
    ```

    The worker and launch scripts are in `examples/tensor_parallel/manual/`.

    Patch the image and node pool into the manifest and apply it:

    ```bash
    sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml
    sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml
    kubectl apply -f examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml
    ```

    Check job and pod status:

    ```bash
    kubectl get job,pods -n default
    ```

    ### Multiple hosts (64 chips / 128 devices across 16 hosts)

    To scale past one host, the same example runs as a Kubernetes `JobSet`:
    one pod per host, 16 hosts for a `4x4x4` slice, with
    `examples/tensor_parallel/manual/launch_multihost.sh` deriving each host's
    rank from its position in the slice. Update the manifest with your Docker
    image and 64-chip node pool name and apply it with `kubectl`:

    ```bash
    sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml
    sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml
    kubectl apply -f examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml
    ```

    Check `JobSet` and pod status:

    ```bash
    kubectl get jobset,pods -n default
    ```

    ## Run an upstream PyTorch example

    Because TorchTPU is a native PyTorch backend (`device="tpu"`,
    `torch.accelerator`, distributed `backend="tpu_dist"`), you can pull and
    run standard scripts from the upstream
    [`pytorch/examples`](https://github.com/pytorch/examples) repository
    directly on GKE.

    ### Example: Running upstream MNIST on GKE

    The upstream [`pytorch/examples/mnist`](https://github.com/pytorch/examples/tree/main/mnist)
    training script (`mnist/main.py`) uses PyTorch's device-agnostic
    `torch.accelerator` API:

    ```python
    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")
    ```

    When `torch_tpu` is installed in the container,
    `torch.accelerator.current_accelerator()` automatically selects the `"tpu"`
    device, requiring **zero code changes** to `mnist/main.py`.

    To run `mnist/main.py` on a single TPU device on GKE starting from
    `examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml`:

    1.  **Update the container `command:` block** in
        `examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml` to install
        `git`, clone
        [`https://github.com/pytorch/examples.git`](https://github.com/pytorch/examples),
        keep `DataLoader` `num_workers` at its default (`0`) so it does not fork
        a worker subprocess that attempts to re-acquire the `libtpu` device
        lockfile, and run `mnist/main.py`:

        ```yaml
          command:
            - /bin/bash
            - -c
            - |
              set -ex
              export PYTHONUNBUFFERED=1
              apt-get update && apt-get install -y git
              git clone --depth 1 https://github.com/pytorch/examples.git /tmp/pytorch_examples
              cd /tmp/pytorch_examples/mnist
              sed -i "/'num_workers': 1,/d; /'persistent_workers': True,/d" main.py
              python3 main.py --epochs=1 --dry-run
        ```

    2.  **Delete any previous run of the job** (since Kubernetes `Job` specs are
        immutable once created), patch the placeholders if needed, apply the
        updated manifest, and stream the MNIST training logs:

        ```bash
        kubectl delete -f examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml --ignore-not-found
        sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
        sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
        kubectl apply -f examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
        kubectl logs -n default -l job-name=torchtpu-singledevice-1chip-job -f
        ```

    ### Choosing a manifest template for other upstream examples

    To run other single-device or distributed upstream examples, pick the
    reference manifest template matching your target topology:

    | Workload Type | Manifest Path | TPU Topology (`gke-tpu-topology`) | `TORCH_TPU_TOPOLOGY` | Description |
    | :--- | :--- | :--- | :--- | :--- |
    | **Single-Device** (1 TensorCore on a 4-chip host) | `examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml` | `2x2x1` | `"1,1,1,1"` | Runs a single Python process on 1 TPU device. |
    | **Single-Host Distributed** (4 chips / 8 cores) | `examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml` | `2x2x1` | `"2,2,1,2"` | Runs 8 distributed ranks (`WORLD_SIZE=8`) on a single 4-chip host via `torchrun`. |
    | **Multi-Host Distributed** (64 chips / 128 cores) | `examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml` | `4x4x4` | `"4,4,4,2"` | Kubernetes `JobSet` coordinating 16 worker pods across a 64-chip slice. |

    To target other TPU slice topologies, update
    `cloud.google.com/gke-tpu-slice-topology`,
    `cloud.google.com/gke-tpu-topology`, and `TORCH_TPU_TOPOLOGY` in the
    manifest (see
    [TPU VM system architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)
    and
    [TPU machine types and topologies](https://docs.cloud.google.com/compute/docs/tpus/tpu-machines)).

    ### Porting an upstream PyTorch example to run on GKE

    Because TorchTPU is a native PyTorch backend, getting an upstream PyTorch
    example or test running on GKE is a minimal process:

    *   **No code changes needed for non-distributed examples**: Non-distributed
        examples written for standard PyTorch devices (via `torch.accelerator`
        or `torch.device`) work with TorchTPU without code changes.
    *   **Non-deprecated examples are expected to work**: Actively maintained,
        non-deprecated upstream PyTorch examples and modern APIs work out of
        the box (see
        [Supported feature set](6_reference_1_supported_feature_set.py) for
        details).
    *   **Distributed examples may need minor modifications**: Some upstream
        distributed examples are overtuned to other devices (such as GPU- or
        NCCL-specific backends, flags, or sharding assumptions) and may require
        minor modifications to run on TPU (see the distributed sections above
        and [FSDP best practices](3_practical_examples_3_fsdp_best_practices.py)).
    """)
  return


if __name__ == "__main__":
  app.run()
