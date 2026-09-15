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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_BUCKET="torch-tpu-mailbox-eval"
DEFAULT_REPO="google-pytorch/torch_tpu"
DEFAULT_FILTER="presubmit-v5,-fails-on-tpu-v5,-nopresubmit,-notest,-nobuild,-requires-tpu-v5lite:8"
DEFAULT_SCOPE="//tests/..."

CLI_PR=""
CLI_REPO="$DEFAULT_REPO"
CLI_MODE="shadow"
CLI_SHA=""
CLI_BUCKET="$DEFAULT_BUCKET"
CLI_TARGETS=()
CLI_JOBS=28
CLI_BAZEL_CONFIG="ci_tpu_v5_mailbox"
CLI_BAZEL_FLAGS=()
CLI_OUTPUT_DIR=""
CLI_NO_REPORT=false
CLI_DRY_RUN=false
CLI_TEST_TIMEOUT="900"
CLI_SCOPE="$DEFAULT_SCOPE"
CLI_FILTER="$DEFAULT_FILTER"

_STATUS_POSTED=false

show_help() {
  cat <<'HELP'
Usage: scripts/mailbox_presubmit_pr.sh [options]

Executes the torch_tpu single-chip presubmit suite on Cloud TPU v5e hardware
via the GCS mailbox runner.

Modes:
  shadow       Advisory run. Reports under "TPU v5e mailbox (shadow)" and
               does not block PR merge (default).
  replacement  Gating run. Reports under "Presubmit on linux-x86-ct5lp-224-8tpu",
               resolving the required status check.

Options:
  --pr N                Pull request number to test.
  --repo OWNER/NAME     GitHub repository (default: google-pytorch/torch_tpu).
  --mode MODE           shadow (default) or replacement.
  --sha SHA             Commit SHA to report against (default: PR head SHA).
  --bucket NAME         GCS mailbox bucket (default: torch-tpu-mailbox-eval).
  --target TARGET       Specific test target to run. Can be repeated.
  --jobs N              Concurrent test jobs (default: 28).
  --bazel-config NAME   Bazel config to inherit (default: ci_tpu_v5_mailbox).
  --bazel-flag FLAG     Extra flag passed to bazel test. Can be repeated.
  --output-dir DIR      Directory to write execution reports and logs.
  --no-report           Do not post commit statuses or comments to GitHub.
  --dry-run             Resolve targets and show execution plan without running.
  -h, --help            Show this help.
HELP
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pr)
      CLI_PR="$2"
      shift 2
      ;;
    --repo)
      CLI_REPO="$2"
      shift 2
      ;;
    --mode)
      CLI_MODE="$2"
      shift 2
      ;;
    --sha)
      CLI_SHA="$2"
      shift 2
      ;;
    --bucket)
      CLI_BUCKET="$2"
      shift 2
      ;;
    --target)
      CLI_TARGETS+=("$2")
      shift 2
      ;;
    --jobs)
      CLI_JOBS="$2"
      shift 2
      ;;
    --bazel-config)
      CLI_BAZEL_CONFIG="$2"
      shift 2
      ;;
    --bazel-flag)
      CLI_BAZEL_FLAGS+=("$2")
      shift 2
      ;;
    --output-dir)
      CLI_OUTPUT_DIR="$2"
      shift 2
      ;;
    --no-report)
      CLI_NO_REPORT=true
      shift
      ;;
    --dry-run)
      CLI_DRY_RUN=true
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      show_help >&2
      exit 2
      ;;
  esac
done

if [[ "$CLI_MODE" != "shadow" && "$CLI_MODE" != "replacement" ]]; then
  echo "ERROR: --mode must be 'shadow' or 'replacement', got '${CLI_MODE}'" >&2
  exit 2
fi

GATING_CONTEXT="Presubmit on linux-x86-ct5lp-224-8tpu"
SHADOW_CONTEXT="TPU v5e mailbox (shadow)"
TARGET_CONTEXT="$SHADOW_CONTEXT"
[[ "$CLI_MODE" == "replacement" ]] && TARGET_CONTEXT="$GATING_CONTEXT"

post_status() {
  local sha="$1" state="$2" description="$3" context="$4"
  [[ "$CLI_NO_REPORT" == "true" ]] && return 0
  command -v gh >/dev/null 2>&1 || return 0
  description="${description:0:140}"
  gh api -X POST "repos/${CLI_REPO}/statuses/${sha}" \
    -f "state=${state}" \
    -f "context=${context}" \
    -f "description=${description}" >/dev/null 2>&1 || {
      echo "WARNING: Failed to post commit status to GitHub." >&2
    }
}

NOW_TAG="$(date -u +%Y%m%d_%H%M%SZ)"
if [[ -z "$CLI_OUTPUT_DIR" ]]; then
  if [[ -n "$CLI_PR" ]]; then
    CLI_OUTPUT_DIR="${REPO_ROOT}/mailbox_reports/pr${CLI_PR}_${NOW_TAG}"
  else
    CLI_OUTPUT_DIR="${REPO_ROOT}/mailbox_reports/run_${NOW_TAG}"
  fi
fi
mkdir -p "$CLI_OUTPUT_DIR"

if [[ -n "$CLI_PR" && -z "$CLI_SHA" ]]; then
  if command -v gh >/dev/null 2>&1; then
    CLI_SHA="$(gh api "repos/${CLI_REPO}/pulls/${CLI_PR}" --jq '.head.sha' 2>/dev/null || true)"
  fi
fi

echo "========================================================================"
echo "TPU Mailbox Presubmit Pipeline"
echo "========================================================================"
echo "Bucket:          gs://${CLI_BUCKET}"
echo "Output Directory:${CLI_OUTPUT_DIR}"
echo "Mode:            ${CLI_MODE}"
[[ -n "$CLI_PR" ]]  && echo "Pull Request:    #${CLI_PR} (${CLI_REPO})"
[[ -n "$CLI_SHA" ]] && echo "Commit SHA:      ${CLI_SHA}"
echo "========================================================================"

echo "Checking fleet status in gs://${CLI_BUCKET}/fleet/..."
FLEET_COUNT=$(gcloud storage ls "gs://${CLI_BUCKET}/fleet/" 2>/dev/null | grep -c '\.json$' || true)
if [[ "$FLEET_COUNT" -eq 0 ]]; then
  echo "WARNING: No healthy TPU VMs currently registered in gs://${CLI_BUCKET}/fleet/." >&2
else
  echo "Found ${FLEET_COUNT} registered TPU VM(s) in fleet."
fi

RESOLVED_TARGETS=()
if [[ ${#CLI_TARGETS[@]} -gt 0 ]]; then
  RESOLVED_TARGETS=("${CLI_TARGETS[@]}")
else
  echo "Querying test targets in ${CLI_SCOPE} matching ${CLI_FILTER}..."
  pos_tags=()
  neg_tags=()
  IFS=',' read -r -a filter_items <<< "$CLI_FILTER"
  for item in "${filter_items[@]}"; do
    item=$(echo "$item" | xargs)
    [[ -z "$item" ]] && continue
    if [[ "$item" == -* ]]; then
      neg_tags+=("${item#-}")
    else
      pos_tags+=("$item")
    fi
  done

  query_expr="tests(${CLI_SCOPE})"
  if [[ ${#pos_tags[@]} -gt 0 ]]; then
    pos_pattern=$(IFS='|'; echo "${pos_tags[*]}")
    query_expr="${query_expr} intersect attr(tags, \"${pos_pattern}\", ${CLI_SCOPE})"
  fi
  if [[ ${#neg_tags[@]} -gt 0 ]]; then
    neg_pattern=$(IFS='|'; echo "${neg_tags[*]}")
    query_expr="${query_expr} except attr(tags, \"${neg_pattern}\", ${CLI_SCOPE})"
  fi

  while IFS= read -r line; do
    [[ -n "$line" && "$line" == //* ]] && RESOLVED_TARGETS+=("$line")
  done < <(bazel query "$query_expr" --output=label 2>/dev/null || true)
fi

TARGET_COUNT="${#RESOLVED_TARGETS[@]}"
echo "Resolved ${TARGET_COUNT} target(s) to execute."

if [[ "$TARGET_COUNT" -eq 0 ]]; then
  echo "ERROR: No test targets resolved." >&2
  exit 1
fi

if [[ "$CLI_DRY_RUN" == "true" ]]; then
  echo "[DRY RUN] Execution plan confirmed. Resolved targets ($TARGET_COUNT):"
  printf "  %s\n" "${RESOLVED_TARGETS[@]}"
  exit 0
fi

if [[ -n "$CLI_SHA" && "$CLI_NO_REPORT" == "false" ]]; then
  post_status "$CLI_SHA" "pending" "TPU mailbox presubmit running across ${FLEET_COUNT} chips" "$TARGET_CONTEXT"
  _STATUS_POSTED=true
fi

BAZEL_ARGS=(
  "test"
  "--test_env=TORCH_TPU_RELAY_BUCKET=${CLI_BUCKET}"
  "--run_under=//ci/tools/relay_mailbox:worker_main"
  "--test_timeout=${CLI_TEST_TIMEOUT}"
  "--jobs=${CLI_JOBS}"
  "--local_test_jobs=${CLI_JOBS}"
  "--test_output=errors"
  "--nocache_test_results"
)

if [[ -n "$CLI_BAZEL_CONFIG" ]]; then
  BAZEL_ARGS+=("--config=${CLI_BAZEL_CONFIG}")
fi

if [[ ${#CLI_BAZEL_FLAGS[@]} -gt 0 ]]; then
  BAZEL_ARGS+=("${CLI_BAZEL_FLAGS[@]}")
fi

BAZEL_ARGS+=("${RESOLVED_TARGETS[@]}")

LOG_FILE="${CLI_OUTPUT_DIR}/bazel_test.log"
REPORT_FILE="${CLI_OUTPUT_DIR}/report.md"

START_TIME=$(date +%s)
echo "Starting Bazel test execution (logs -> ${LOG_FILE})..."

set +e
bazel "${BAZEL_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
BAZEL_RC=$?
set -e

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

VERDICT="PASSED"
STATUS_STATE="success"
[[ "$BAZEL_RC" -ne 0 ]] && VERDICT="FAILED" && STATUS_STATE="failure"

echo "Bazel execution completed in ${ELAPSED}s with verdict: ${VERDICT} (rc=${BAZEL_RC})"

cat > "$REPORT_FILE" <<RPT
# TPU Mailbox Presubmit Report

**Verdict:** ${VERDICT}
**Mode:** ${CLI_MODE}
**Wall Clock Time:** ${ELAPSED}s
**Bucket:** \`gs://${CLI_BUCKET}\`
**Targets Evaluated:** ${TARGET_COUNT}
**Fleet Capacity:** ${FLEET_COUNT} chips
**Timestamp:** ${NOW_TAG}

## Summary

| Metric | Value |
| :--- | :--- |
| **Status** | **${VERDICT}** |
| **Duration** | ${ELAPSED}s |
| **Exit Code** | ${BAZEL_RC} |
| **Targets** | ${TARGET_COUNT} |
| **Fleet Capacity** | ${FLEET_COUNT} chips registered |

RPT

if [[ "$BAZEL_RC" -ne 0 ]]; then
  echo "### Failures" >> "$REPORT_FILE"
  echo '```' >> "$REPORT_FILE"
  grep -E 'FAIL:|TIMEOUT:|ERROR:' "$LOG_FILE" | head -n 30 >> "$REPORT_FILE" || true
  echo '```' >> "$REPORT_FILE"
fi

echo "Report generated at: ${REPORT_FILE}"

if [[ -n "$CLI_SHA" && "$CLI_NO_REPORT" == "false" ]]; then
  post_status "$CLI_SHA" "$STATUS_STATE" "TPU mailbox presubmit ${VERDICT} in ${ELAPSED}s (${TARGET_COUNT} targets)" "$TARGET_CONTEXT"
fi

if [[ -n "$CLI_PR" && "$CLI_NO_REPORT" == "false" ]]; then
  if command -v gh >/dev/null 2>&1; then
    echo "Posting report to PR #${CLI_PR} on ${CLI_REPO}..."
    gh pr comment "$CLI_PR" -R "$CLI_REPO" --body-file "$REPORT_FILE" || {
      echo "WARNING: Failed to post comment to PR #${CLI_PR}." >&2
    }
  fi
fi

exit "$BAZEL_RC"
