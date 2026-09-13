# Developer Guide: CI Presubmits, TPU v5 Bypass & RBE Modes

This document explains how to control CI presubmit test executions in `torch_tpu`, including running Remote Build Execution (RBE) as a shadow run or replacing self-hosted Cloud TPU v5 presubmits with the RBE setup.

---

## 1. Quick Reference: Which Label Should I Use?

Every label needs write access to the repo, which is the point: skipping
hardware tests should take more than editing a PR description.

| Goal | How to Trigger | What Happens |
| :--- | :--- | :--- |
| **Normal presubmits** | Default | Runs CPU, TPU v5 (`linux-x86-ct5lp-224-8tpu`), and TPU v7. |
| **Relay shadow run** | Label `ci:relay-tpu-v5` | Normal presubmits still run. The relay runs the same v5 targets on Cloud TPU v5e VMs as advisory (`continue-on-error: true`) and never blocks the PR. |
| **Relay replacement** | Label `ci:replace-tpu-v5` | Drops TPU v5 from the presubmit matrix and gates the PR on the relay instead. The relay job publishes under the ct5lp check name, so branch protection sees its real verdict. CPU and TPU v7 still run. |
| **Full-RBE shadow run** | Label `run-rbe` | Advisory run of the CPU and TPU v5e suites through RBE workers. Needs TPU worker pools that do not exist yet, so today this reports a credentials warning and skips. |
| **Standalone bypass** | Label `ci:bypass-tpu-v5` | Drops TPU v5 and runs nothing in its place. For runner outages, or PRs that don't touch v5 code. |
| **Manual dispatch** | Actions UI or `gh workflow run` | `bypass-tpu-v5: true` on `presubmit.yml`, or `hardware_path` + `mode` on `test_rbe_opt_in.yml`. |

---

## 2. Execution Modes Explained

### Mode 1: Relay shadow run (`ci:relay-tpu-v5`)
- **Use case**: Try the relay on your PR while the ct5lp presubmit stays the
  source of truth. This is the mode to start with.
- **What happens**:
  - `presubmit.yml` runs CPU, TPU v5, and TPU v7 as usual.
  - `test_rbe_opt_in.yml`'s `relay_tpu_v5` job attaches to the standing v5e
    fleet in `rbe-tpu-oss`, runs the `presubmit-v5` single-chip targets, and
    uploads a markdown report as a run artifact.
  - Relay failures are advisory and don't block the merge.
- **What it covers**: the same targets the ct5lp job runs, minus the ones tagged
  `requires-tpu-v5lite:8`. The relay leases one v5litepod-1 VM per test action,
  so a multi-chip test can never pass there. That is 57 of the 77
  `presubmit-v5` targets.

### Mode 2: Relay replacement (`ci:replace-tpu-v5`)
- **Use case**: The self-hosted TPU v5 runners are backed up, or you're
  validating the relay as the gating path for v5.
- **What happens**:
  - `presubmit_job_matrix.sh` leaves `linux-x86-ct5lp-224-8tpu` out of the matrix.
  - `presubmit.yml`'s bypass notice job stands down. It publishes under
    `TPU v5 Bypass Notice` instead of claiming the ct5lp check name,
    because an always-green check next to a real one hides a red relay run.
  - `relay_tpu_v5` claims `Presubmit on linux-x86-ct5lp-224-8tpu` and runs with
    `continue-on-error: false`, so a relay failure blocks the PR.

> [!WARNING]
> If the relay can't reach `rbe-tpu-oss`, a run in this mode fails instead of
> reporting green. A gating check that never touched hardware has verified
> nothing.

> [!NOTE]
> Multi-chip coverage does not move with the label. The 20 targets tagged
> `requires-tpu-v5lite:8` run on neither path while `ci:replace-tpu-v5` is on.
> Don't leave the label on a PR that touches collective ops.

### Mode 3: Full-RBE shadow run (`run-rbe`)
- **Use case**: Exercising the RBE path once TPU-attached worker pools exist.
- **What happens**: `run_tests` builds and tests through RBE with
  `--config=ci_tpu_v5_full_rbe`. Advisory only. This label deliberately cannot
  gate a PR, because `//bazel/platforms:rbe_tpu_v5e` asks for a worker pool
  nobody has provisioned.

### Mode 4: Standalone bypass (`ci:bypass-tpu-v5`)
- **Use case**: The PR only touches CPU, TPU v7, or docs, or the v5 runners are
  down.
- **What happens**: TPU v5 is skipped, nothing runs in its place, CPU and TPU v7
  run as usual. The bypass notice job claims the ct5lp check name so branch
  protection stays satisfied.

---

## 3. Manual Workflow Dispatch via GitHub CLI

```bash
# Run presubmits with TPU v5 bypassed
gh workflow run presubmit.yml -f bypass-tpu-v5=true

# Relay, advisory
gh workflow run test_rbe_opt_in.yml -f hardware_path=relay -f mode=shadow

# Relay, gating
gh workflow run test_rbe_opt_in.yml -f hardware_path=relay -f mode=replacement

# Full RBE instead of the relay
gh workflow run test_rbe_opt_in.yml -f hardware_path=full-rbe -f test_suite=tpu_v5e_only
```

Adding a label from the command line, since `gh pr edit --add-label` does not
work on this repo:

```bash
gh api repos/google-pytorch/torch_tpu/issues/<PR>/labels \
  -f 'labels[]=ci:relay-tpu-v5'
```

---

## 4. RBE credentials (one-time repo setup)

Both paths authenticate with Workload Identity Federation. Neither uses a
service account key, and neither can: `rbe-tpu-oss` carries
`constraints/iam.disableServiceAccountKeyCreation`, so no exportable key exists
to put in a secret.

Set these **repository variables** (not secrets — none of the values are
sensitive):

| Variable | Value | Needed by |
| :--- | :--- | :--- |
| `GCP_WIF_PROVIDER` | Full provider resource name, `projects/<num>/locations/global/workloadIdentityPools/<pool>/providers/<provider>` | both paths |
| `GCP_RBE_SERVICE_ACCOUNT` | Service account email the provider is allowed to impersonate | both paths |
| `RELAY_TPU_ZONES` | Space-separated zones to look for fleet VMs in. Defaults to `europe-west4-b` | relay only |

Until these are set:

- **Shadow runs** log a warning and skip. They were never going to block anything.
- **Replacement runs fail.** They gate the PR on hardware, so reporting green
  without reaching it would be a lie.

---

## 5. Turning the relay on for the repo (one-time setup)

Everything below is infrastructure work, done once. After that, contributors
only need to add a label.

**1. A standing v5e fleet.** The relay attaches to VMs that already exist; it
never creates them. Bring the fleet up from a workstation or a long-lived job:

```bash
scripts/spot_tpu_fleet.sh up --size 8 --on-demand \
  --zone europe-west4-b --pool /tmp/tpu_pool \
  --deadline-minutes 0     # 0 disables the auto-teardown deadline
```

Check what is running with `scripts/spot_tpu_fleet.sh status --pool /tmp/tpu_pool`.

> [!CAUTION]
> These VMs bill until deleted. A standing fleet is a standing cost. Tear it
> down with `scripts/spot_tpu_fleet.sh down --pool /tmp/tpu_pool --zone <zone>`
> when the experiment ends, and check for strays with
> `scripts/spot_tpu_manager.sh reap --dry-run`.

**2. A GitHub OIDC provider in `rbe-tpu-oss`.** Create a workload identity pool
and a provider for `https://token.actions.githubusercontent.com`, restricted to
the `google-pytorch/torch_tpu` repository. Put its resource name in
`GCP_WIF_PROVIDER`.

**3. IAM on the service account the provider impersonates.** The relay job needs
to list TPU VMs, read their addresses, and push its public key:

| Role | Why |
| :--- | :--- |
| `roles/tpu.viewer` | `attach` lists READY VMs and reads their external IPs |
| `roles/tpu.admin` | `gcloud compute tpus tpu-vm ssh` uploads the run's public key |
| `roles/remotebuildexecution.actionCacheWriter` | the build phase writes to the RBE cache |

**4. Network path.** The runner SSHes to the VMs' external IPs on port 22. The
default network's `default-allow-ssh` rule already permits this. If that rule is
tightened, the relay stops working from GitHub-hosted runners and needs a
self-hosted runner inside the VPC instead.

**5. Branch protection, only if you want replacement to gate.** Nothing to
change: the relay job publishes under the existing
`Presubmit on linux-x86-ct5lp-224-8tpu` check name when `ci:replace-tpu-v5` is
on, so the required check you already have keeps working and now carries the
relay's verdict.

**6. The labels.** Create `ci:relay-tpu-v5` if it does not exist:

```bash
gh label create ci:relay-tpu-v5 --repo google-pytorch/torch_tpu \
  --description "Run the TPU v5e SSH relay as an advisory shadow check"
```

### How the CI job borrows the fleet

`relay_tpu_v5` mints an ed25519 key for the run, calls
`spot_tpu_fleet.sh attach`, and always calls `detach` at the end.

- `attach` lists `state:READY` VMs whose names start with `spot-tpu-v5e-`,
  uploads the run's public key to each, and writes one session file per VM. It
  issues no `create` and no `delete`.
- `detach` deletes the session files and leaves the hardware alone. CI must
  never call `down`: that would delete VMs belonging to whoever brought the
  fleet up.

A job-level concurrency group (`relay-tpu-v5-fleet`, `cancel-in-progress:
false`) keeps two relay runs from each trying to lease the whole fleet.

---

## 6. Running the suite on Spot TPUs locally

`presubmit.yml` and `test_rbe_opt_in.yml` are the CI paths. To run the same test
targets against real Spot TPU v5e VMs from a workstation, use the relay:

```bash
# Single VM, one test at a time (a v5litepod-1 has one chip).
scripts/run_presubmit_v5_relay.sh

# See what would run without provisioning anything.
scripts/run_presubmit_v5_relay.sh --dry-run

# Spread the suite over a fleet. Each test leases one VM for its lifetime.
scripts/spot_tpu_fleet.sh up --size 8 --pool /tmp/tpu_pool
scripts/run_presubmit_v5_relay.sh --session-pool=/tmp/tpu_pool --jobs 8
scripts/spot_tpu_fleet.sh down --pool /tmp/tpu_pool --zone europe-west4-b

# Borrow a fleet somebody else brought up, then hand it back.
scripts/spot_tpu_fleet.sh attach --pool /tmp/my_pool --zone europe-west4-b
scripts/run_presubmit_v5_relay.sh --session-pool=/tmp/my_pool --jobs 8
scripts/spot_tpu_fleet.sh detach --pool /tmp/my_pool
```

On a machine with a cold Bazel output base, add `--bazel-config ci_tpu_v5_relay`
so the compile actions go out to RBE. Plain `--config=ci` will not do: it sets
`--remote_download_minimal`, and `stage_relay_base.sh` builds the base cache by
reading the runfiles trees off `bazel-bin`, which minimal downloads never
materialise.

> [!CAUTION]
> `spot_tpu_fleet.sh down` is not optional if you brought the fleet up yourself.
> Spot VMs bill until deleted. Check for strays with
> `scripts/spot_tpu_manager.sh reap --dry-run`.

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
