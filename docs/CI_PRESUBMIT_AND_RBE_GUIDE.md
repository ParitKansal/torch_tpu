# Developer Guide: CI Presubmits, TPU v5 Bypass & RBE Modes

This document explains how to control CI presubmit test executions in `torch_tpu`, including running Remote Build Execution (RBE) as a shadow run or replacing self-hosted Cloud TPU v5 presubmits with the RBE setup.

---

## 1. Quick Reference: Which Label Should I Use?

There are two labels. Both require write access to the repo, which is the point:
skipping hardware tests should take more than editing a PR description.

| Goal | How to Trigger | What Happens |
| :--- | :--- | :--- |
| **Normal presubmits** | Default | Runs CPU, TPU v5 (`linux-x86-ct5lp-224-8tpu`), and TPU v7. |
| **Shadow run** | Label `run-rbe` | Normal presubmits still run. The RBE suite runs alongside as advisory (`continue-on-error: true`) and never blocks the PR. |
| **Replacement** | Label `ci:replace-tpu-v5` | Drops TPU v5 from the presubmit matrix and promotes RBE to a gating check (`continue-on-error: false`). CPU and TPU v7 still run. |
| **Standalone bypass** | Label `ci:bypass-tpu-v5` | Drops TPU v5 and runs nothing in its place. For runner outages, or PRs that don't touch v5 code. |
| **Manual dispatch** | Actions UI or `gh workflow run` | `bypass-tpu-v5: true` on `presubmit.yml`, or `mode: replacement` on `test_rbe_opt_in.yml`. |

---

## 2. Execution Modes Explained

### Mode 1: Shadow run (`run-rbe`)
- **Use case**: Try the RBE suite on your PR while the normal presubmits stay the
  source of truth.
- **What happens**:
  - `presubmit.yml` runs CPU, TPU v5, and TPU v7 as usual.
  - `test_rbe_opt_in.yml` runs the CPU and TPU v5e suites on RBE workers.
  - RBE failures are advisory and don't block the merge.

### Mode 2: Replacement (`ci:replace-tpu-v5`)
- **Use case**: The self-hosted TPU v5 runners are backed up, or you're
  validating RBE as the gating path for v5.
- **What happens**:
  - `presubmit_job_matrix.sh` leaves `linux-x86-ct5lp-224-8tpu` out of the matrix.
  - A stub job named `"Presubmit on linux-x86-ct5lp-224-8tpu"` reports the
    required check immediately on `ubuntu-latest`, so branch protection stays
    satisfied without burning runner hours.
  - `test_rbe_opt_in.yml` runs with `continue-on-error: false`, so an RBE failure
    blocks the PR.

> [!WARNING]
> If RBE credentials aren't configured, a run in this mode fails instead of
> reporting green. A gating check that can't reach RBE has verified nothing.

### Mode 3: Standalone bypass (`ci:bypass-tpu-v5`)
- **Use case**: The PR only touches CPU, TPU v7, or docs, or the v5 runners are
  down.
- **What happens**: TPU v5 is skipped, RBE is not invoked, CPU and TPU v7 run as
  usual.

---

## 3. Manual Workflow Dispatch via GitHub CLI

You can trigger these workflows from the command line:

```bash
# Run presubmits with TPU v5 bypassed
gh workflow run presubmit.yml -f bypass-tpu-v5=true

# Run RBE suite in replacement mode (strict gating)
gh workflow run test_rbe_opt_in.yml -f mode=replacement

# Run only the TPU v5e RBE test suite
gh workflow run test_rbe_opt_in.yml -f mode=replacement -f test_suite=tpu_v5e_only
```

---

## 4. RBE credentials (one-time repo setup)

The RBE workflow authenticates with Workload Identity Federation. It does not
use a service account key, and can't: `rbe-tpu-oss` carries
`constraints/iam.disableServiceAccountKeyCreation`, so no exportable key exists
to put in a secret.

Set two **repository variables** (not secrets — neither value is sensitive):

| Variable | Value |
| :--- | :--- |
| `GCP_WIF_PROVIDER` | Full provider resource name, `projects/<num>/locations/global/workloadIdentityPools/<pool>/providers/<provider>` |
| `GCP_RBE_SERVICE_ACCOUNT` | Service account email the provider is allowed to impersonate |

Until these are set:

- **Shadow runs** log a warning and skip. They were never going to block anything.
- **Replacement runs fail.** They gate the PR on RBE, so reporting green without
  reaching RBE would be a lie.

---

## 5. Running the suite on Spot TPUs locally

`presubmit.yml` and `test_rbe_opt_in.yml` are the CI paths. To run the same test
targets against real Spot TPU v5e VMs from a workstation, use the relay:

```bash
# Single VM, one test at a time (a v5litepod-1 has one chip).
scripts/run_presubmit_v5_relay.sh

# See what would run without provisioning anything.
scripts/run_presubmit_v5_relay.sh --dry-run

# Spread the suite over a fleet. Each test leases one VM for its lifetime.
scripts/spot_tpu_fleet.sh up --size 8
scripts/run_presubmit_v5_relay.sh --session-pool=/tmp/torch_tpu_relay/pool
scripts/spot_tpu_fleet.sh down
```

> [!CAUTION]
> `spot_tpu_fleet.sh down` is not optional. Spot VMs bill until deleted. Check
> for strays with `scripts/spot_tpu_manager.sh reap --dry-run`.

> [!NOTE]
> Spot capacity is the real limit, not quota. As of this writing only
> `europe-west4-b` has a v5e reservation with room; `us-central1-a` regularly
> returns "no more capacity" and `us-east5-a` has no reservation at all. Asking
> for 12 VMs can easily get you 4, so `up` is resumable — re-run it to fill the
> gaps.

---

## 6. SSH Relay Architecture

We use a Bazel `--run_under` wrapper (`ci/tools/relay_test_runner.sh`) that intercepts test actions on the host, leases a Cloud TPU v5e VM from a pool via `flock`, and runs the test in a sandbox on the remote VM (`ci/tools/remote_tpu_executor.sh`).

### Performance and the Base Cache

A real 57-target run expands to 330 test actions holding 7,895s of test body time. With 8 VMs, a perfect floor is 987s. The original wall time was 3,015s — a 32.7% efficiency caused by 49.2s of relay overhead per action. The pool averaged 2.58 active VMs; adding more VMs does not speed up the run if they stay locked in I/O.

The overhead was mostly the `libpywrap_torch_tpu_common.so` extension module. At 493 MB out of a 499 MB payload, it was identical for every test but got packaged and shipped repeatedly.

`ci/tools/stage_relay_base.sh` now pushes `_main/csrc` alongside the shared C++ solibs into the base cache on the VM. `ci/tools/remote_tpu_executor.sh` symlinks `${BASE_CACHE}/csrc` directly into the sandbox workspace root on a cache miss, and `relay_test_runner.sh` excludes it from the per-action payload. The payload packing step dropped from 14.63s (120 MB gzipped) to 0.066s (200 KB). The base cache layer stamp already hashes file size and mtime, so a rebuilt extension module re-stages automatically.

### Bazel Execution Strategy

`.bazelrc` now uses `--strategy=TestRunner=local` to pin only test actions to the relay runner. The previous flag, `--spawn_strategy=standalone,local`, inadvertently forced compilation and linking jobs onto the host instead of sending them to RBE.

### Watchdogs and Quarantine

We removed the in-guest shutdown watchdogs from `spot_tpu_manager.sh`. Guest-side `shutdown -h` tells the TPU service to reboot the VM, not stop billing. Reboots wipe `/tmp` (deleting the base cache) and kill the SSH control master.

Before, a broken VM failed tests in milliseconds, released its `flock` instantly, and won the race for the next lease. We saw one rebooted VM instantly fail 64 shards while seven healthy VMs sat idle.

Now `relay_test_runner.sh` writes a `.quarantine` marker next to the session file of any VM answering without a base cache or SSH connection. The fleet scripts skip it from then on.

To catch orphaned billing, `scripts/spot_tpu_fleet.sh up` arms a detached host-side deadline (default 180 min, disable with `--deadline-minutes 0`). The timeout runs `down`, which sweeps both zones explicitly for unknown VMs named `spot-tpu-v5e-*`.

### Unrunnable Multi-Chip Targets

The relay leases one `v5litepod-1` VM per action. Multi-chip targets will hang until they hit their timeout, then fail. The relay excludes `requires-tpu-v5lite:8` by default. 20 of the 77 v5 targets need 8 chips, leaving 57 targets holding the 330 shards.
