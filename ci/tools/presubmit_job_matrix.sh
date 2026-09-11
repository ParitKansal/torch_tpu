#!/usr/bin/env bash
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

# Prints the CI presubmit job matrix as a single line of JSON for
# .github/workflows/presubmit.yml to feed into `strategy.matrix`.
#
# Set BYPASS_TPU_V5=true to drop the self-hosted TPU v5 runner from the matrix.
# That runner sits behind a shared queue that regularly takes over an hour, so
# PRs that cover TPU v5 another way (see docs/CI_PRESUBMIT_AND_RBE_GUIDE.md) can
# leave it out and let the bypass job report the required check instead.
#
# Usage:
#   $ ci/tools/presubmit_job_matrix.sh
#   $ BYPASS_TPU_V5=true ci/tools/presubmit_job_matrix.sh

set -euo pipefail

# Runner label of the self-hosted TPU v5 machine, and the only entry the bypass
# drops. presubmit.yml names it again in the bypass job so the required check
# keeps reporting under the same name.
readonly TPU_V5_RUNNER="linux-x86-ct5lp-224-8tpu"

# Accelerator runners hand each test to a wrapper that leases it a single chip
# and reaps leftover workers, so concurrent tests don't fight over one device.
# Single quotes keep $(pwd) intact for the runner to expand.
readonly ACCELERATOR_RUN_UNDER='--run_under="$(pwd)/ci/tools/parallel_accelerator_execute.sh"'

# TODO(gunhyun): Re-enable the v6 runner once we have more quota.
#   {runner: "linux-x86-ct6e-180-8tpu", config: "ci_tpu_v6_presubmit",
#    extra_flags: $accelerator_run_under}
matrix=$(jq -cn \
  --arg accelerator_run_under "$ACCELERATOR_RUN_UNDER" \
  --arg tpu_v5_runner "$TPU_V5_RUNNER" '[
  {
    name: "CPU",
    runner: "linux-x86-n4-16",
    config: "ci_cpu_presubmit",
    extra_flags: ""
  },
  {
    runner: $tpu_v5_runner,
    config: "ci_tpu_v5_presubmit",
    extra_flags: $accelerator_run_under
  },
  {
    runner: "linux-x86-tpu7x-224-4tpu",
    config: "ci_tpu_v7_presubmit",
    extra_flags: $accelerator_run_under
  }
]')

if [[ "${BYPASS_TPU_V5:-false}" == "true" ]]; then
  matrix=$(jq -c --arg runner "$TPU_V5_RUNNER" \
    'map(select(.runner != $runner))' <<<"$matrix")
fi

printf '%s\n' "$matrix"
