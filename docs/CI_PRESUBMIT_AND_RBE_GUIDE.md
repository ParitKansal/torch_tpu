# Developer Guide: CI Presubmits, TPU v5 Bypass & RBE Modes

This document explains how to control CI presubmit test executions in `torch_tpu`, including running Remote Build Execution (RBE) as a shadow run or replacing self-hosted Cloud TPU v5 presubmits with the RBE setup.

---

## 1. Quick Reference: Which Label or Tag Should I Use?

| Goal | How to Trigger | What Happens |
| :--- | :--- | :--- |
| **Normal Presubmits** | Default (open PR or push commits) | Executes CPU, TPU v5 (`linux-x86-ct5lp-224-8tpu`), and TPU v7 runners. |
| **Shadow Run Mode** | Add label `run-rbe` | Executes all normal presubmits **AND** runs the RBE presubmit suite concurrently. RBE runs in advisory mode (`continue-on-error: true`). |
| **Replacement Mode** | Add label `ci:replace-tpu-v5` (or `replace-tpu-v5`) | **Bypasses** self-hosted TPU v5 presubmit (`linux-x86-ct5lp-224-8tpu`) and **runs RBE** as an authoritative gating presubmit (`continue-on-error: false`). CPU and TPU v7 presubmits continue to run. |
| **Dual-Label Replace** | Add labels `run-rbe` + `ci:bypass-tpu-v5` | Same as replacement mode: TPU v5 is bypassed, RBE runs. |
| **Standalone Bypass** | Add label `ci:bypass-tpu-v5` (or include `[skip-tpu-v5]` in PR title / description) | Bypasses self-hosted TPU v5 presubmit. RBE is **not** triggered. Ideal during GKE TPU runner stockouts or when only modifying CPU/v7 code. |
| **Manual Dispatch** | Use GitHub Actions UI or `gh workflow run` | Specify `bypass-tpu-v5: true` on `presubmit.yml`, or `mode: replacement` on `test_rbe_opt_in.yml`. |

---

## 2. Execution Modes Explained

### Mode 1: Shadow Run Mode (`run-rbe`)
- **Use case**: You want to test the RBE TPU presubmit test suite on your PR while preserving the standard upstream presubmit checks as the source of truth.
- **Workflow behavior**:
  - `.github/workflows/presubmit.yml` runs normally across CPU, TPU v5, and TPU v7.
  - `.github/workflows/test_rbe_opt_in.yml` executes CPU Presubmit Tests and TPU v5e Tests (Full RBE) on Google Cloud RBE workers.
  - Failures in RBE are advisory and do not block PR merging.

### Mode 2: Replacement Mode (`ci:replace-tpu-v5` or `replace-tpu-v5`)
- **Use case**: Self-hosted TPU v5 runners on GKE are backlogged or undergoing maintenance, or you want to actively validate RBE as the gating CI mechanism for TPU v5.
- **Workflow behavior**:
  - In `.github/workflows/presubmit.yml`, the `setup` step detects the replacement label and excludes `linux-x86-ct5lp-224-8tpu` from the matrix.
  - A 1-second stub job named `"Presubmit on linux-x86-ct5lp-224-8tpu"` completes immediately on `ubuntu-latest`, satisfying status check requirements without consuming GKE runner hours.
  - `.github/workflows/test_rbe_opt_in.yml` runs the RBE presubmit suite with `continue-on-error: false`. Test failures in RBE will block the PR from merging.

### Mode 3: Standalone Bypass Mode (`ci:bypass-tpu-v5` or `[skip-tpu-v5]`)
- **Use case**: Your PR only touches CPU, TPU v7, or documentation code and does not require testing on physical TPU v5 hardware.
- **Workflow behavior**:
  - Self-hosted TPU v5 runner is bypassed.
  - RBE is not invoked.
  - CPU and TPU v7 presubmits execute as usual.

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
