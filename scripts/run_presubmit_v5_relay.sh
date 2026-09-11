#!/usr/bin/env bash
# scripts/run_presubmit_v5_relay.sh
# Presubmit execution pipeline on physical Cloud TPU v5e via SSH relay.
set -euo pipefail
set -m

readonly ALLOWED_PROJECT="rbe-tpu-oss"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly SESSION_ENV_FILE="/tmp/tpu_active_session.env"
if [[ -n "${SPOT_TPU_MANAGER_BIN:-}" ]]; then
  SPOT_MANAGER="${SPOT_TPU_MANAGER_BIN}"
elif command -v spot_tpu_manager.sh >/dev/null 2>&1 && [[ "$(command -v spot_tpu_manager.sh)" != "${SCRIPT_DIR}/spot_tpu_manager.sh" ]]; then
  SPOT_MANAGER="$(command -v spot_tpu_manager.sh)"
else
  SPOT_MANAGER="${SCRIPT_DIR}/spot_tpu_manager.sh"
fi
readonly SPOT_MANAGER

readonly RELAY_RUNNER="%workspace%/ci/tools/relay_test_runner.sh"
readonly REPORTER="${GENERATE_PRESUBMIT_REPORT_BIN:-${SCRIPT_DIR}/generate_presubmit_report.py}"

# Default configuration values
CLI_ZONE="europe-west4-b"
CLI_FILTER="presubmit-v5,-fails-on-tpu-v5,-nopresubmit,-notest,-nobuild"
CLI_TARGETS=""
CLI_DRY_RUN=false
CLI_OUTPUT_DIR=""
CLI_TEST_TIMEOUT="900s"
CLI_KEEP_VM_ON_FAILURE=false
CLI_PROJECT="$ALLOWED_PROJECT"

# Process supervision state
_BAZEL_PID=""
_TEARDOWN_DONE=0
_PRESUBMIT_FAILED=0
_FINAL_EXIT_CODE=0

show_help() {
  cat <<'EOF'
Usage: scripts/run_presubmit_v5_relay.sh [options]

Executes the torch_tpu presubmit-v5 test suite on physical Cloud TPU silicon
via the SSH relay test runner and Spot TPU instances in rbe-tpu-oss.

Options:
  --zone=ZONE              GCP zone for Spot TPU VM (default: europe-west4-b)
  --filter=TAG_EXPR        Test tag filter expression
                           (default: presubmit-v5,-fails-on-tpu-v5,-nopresubmit,-notest,-nobuild)
  --targets=LABELS         Comma- or space-separated target labels or package pattern
                           (default: all targets in //tests/... matching filter tags)
  --dry-run                Resolve target list, print plan, and exit without provisioning VM
  --output-dir=DIR         Directory for test logs and reports
                           (default: presubmit_reports/run_<timestamp>)
  --test-timeout=SEC       Per-test timeout duration (default: 900s)
  --keep-vm-on-failure     Do not delete TPU VM on test failure or error (for debugging)
  --project=PROJECT        GCP project (strictly restricted to rbe-tpu-oss)
  -h, --help               Show this help message and exit
EOF
}

# 1. Project boundary enforcement
enforce_project_boundary() {
  local target_project="${1:-$ALLOWED_PROJECT}"
  if [[ "$target_project" != "$ALLOWED_PROJECT" ]]; then
    echo "ERROR [run_presubmit]: Project boundary violation: '$target_project' is not allowed." >&2
    echo "This presubmit runner is strictly restricted to project '$ALLOWED_PROJECT'." >&2
    exit 1
  fi

  local env_vars=("CLOUDSDK_CORE_PROJECT" "GOOGLE_CLOUD_PROJECT" "GCP_PROJECT" "GCLOUD_PROJECT" "TPU_PROJECT")
  for var in "${env_vars[@]}"; do
    if [[ -n "${!var:-}" && "${!var}" != "$ALLOWED_PROJECT" ]]; then
      echo "ERROR [run_presubmit]: Project boundary violation: Environment variable $var is set to '${!var}' (violates allowed project '$ALLOWED_PROJECT')." >&2
      exit 1
    fi
  done
}

# 2. CLI option parsing
parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --zone)
        [[ $# -lt 2 ]] && { echo "ERROR [run_presubmit]: --zone requires an argument." >&2; exit 1; }
        CLI_ZONE="$2"; shift 2 ;;
      --zone=*)
        CLI_ZONE="${1#*=}"; shift ;;
      --filter)
        [[ $# -lt 2 ]] && { echo "ERROR [run_presubmit]: --filter requires an argument." >&2; exit 1; }
        CLI_FILTER="$2"; shift 2 ;;
      --filter=*)
        CLI_FILTER="${1#*=}"; shift ;;
      --targets)
        [[ $# -lt 2 ]] && { echo "ERROR [run_presubmit]: --targets requires an argument." >&2; exit 1; }
        CLI_TARGETS="$2"; shift 2 ;;
      --targets=*)
        CLI_TARGETS="${1#*=}"; shift ;;
      --dry-run)
        CLI_DRY_RUN=true; shift ;;
      --output-dir)
        [[ $# -lt 2 ]] && { echo "ERROR [run_presubmit]: --output-dir requires an argument." >&2; exit 1; }
        CLI_OUTPUT_DIR="$2"; shift 2 ;;
      --output-dir=*)
        CLI_OUTPUT_DIR="${1#*=}"; shift ;;
      --test-timeout)
        [[ $# -lt 2 ]] && { echo "ERROR [run_presubmit]: --test-timeout requires an argument." >&2; exit 1; }
        CLI_TEST_TIMEOUT="$2"; shift 2 ;;
      --test-timeout=*)
        CLI_TEST_TIMEOUT="${1#*=}"; shift ;;
      --keep-vm-on-failure)
        CLI_KEEP_VM_ON_FAILURE=true; shift ;;
      --project)
        [[ $# -lt 2 ]] && { echo "ERROR [run_presubmit]: --project requires an argument." >&2; exit 1; }
        CLI_PROJECT="$2"; shift 2 ;;
      --project=*)
        CLI_PROJECT="${1#*=}"; shift ;;
      -h|--help)
        show_help; exit 0 ;;
      *)
        echo "ERROR [run_presubmit]: Unknown option: $1" >&2
        show_help >&2
        exit 1 ;;
    esac
  done

  enforce_project_boundary "$CLI_PROJECT"
}

# 3. Session health inspection
is_tpu_session_healthy() {
  if [[ ! -f "$SESSION_ENV_FILE" ]]; then
    return 1
  fi

  local status_output=""
  if ! status_output=$("$SPOT_MANAGER" status --project="$ALLOWED_PROJECT" 2>&1); then
    return 1
  fi

  if [[ "$status_output" == *"NO_ACTIVE_SESSION"* ]]; then
    return 1
  fi

  if [[ "$status_output" == *"INACTIVE"* || "$status_output" == *"CLOSED"* ]]; then
    return 1
  fi

  return 0
}

# 4. Signal traps and teardown supervisor
cleanup_orchestrator() {
  local origin_rc=$?
  local sig="${1:-EXIT}"

  if [[ "$_TEARDOWN_DONE" -eq 1 ]]; then
    return
  fi
  _TEARDOWN_DONE=1
  trap '' INT TERM HUP

  local exit_code=0
  if [[ "$sig" != "EXIT" ]]; then
    case "$sig" in
      INT)  exit_code=130 ;;
      TERM) exit_code=143 ;;
      HUP)  exit_code=129 ;;
      *)    exit_code=1 ;;
    esac
    _PRESUBMIT_FAILED=1
    echo -e "\nCaught signal $sig. Initiating clean termination..." >&2
  else
    exit_code="$origin_rc"
    if [[ "$exit_code" -ne 0 ]]; then
      _PRESUBMIT_FAILED=1
    fi
  fi

  # Terminate active Bazel child process tree
  local b_pid="${_BAZEL_PID:-}"
  if [[ -n "$b_pid" ]] && (kill -0 "$b_pid" 2>/dev/null || kill -0 -"$b_pid" 2>/dev/null); then
    echo "Terminating Bazel process group (PID: $b_pid)..." >&2
    kill -TERM -"$b_pid" 2>/dev/null || kill -TERM "$b_pid" 2>/dev/null || true
    local elapsed=0
    while (kill -0 "$b_pid" 2>/dev/null || kill -0 -"$b_pid" 2>/dev/null) && [[ $elapsed -lt 20 ]]; do
      sleep 0.5
      elapsed=$(( elapsed + 1 ))
    done
    if kill -0 "$b_pid" 2>/dev/null || kill -0 -"$b_pid" 2>/dev/null; then
      echo "Bazel process did not shut down gracefully; sending SIGKILL..." >&2
      kill -KILL -"$b_pid" 2>/dev/null || kill -KILL "$b_pid" 2>/dev/null || true
    fi
    wait "$b_pid" 2>/dev/null || true
    _BAZEL_PID=""
  fi

  # Conditional VM preservation or deletion
  if [[ "$CLI_DRY_RUN" == "false" ]]; then
    if [[ "$sig" == "EXIT" && "$CLI_KEEP_VM_ON_FAILURE" == "true" && "$_PRESUBMIT_FAILED" -eq 1 ]]; then
      local tpu_vm="unknown" tpu_z="unknown" tpu_ip="unknown"
      if [[ -f "$SESSION_ENV_FILE" ]]; then
        eval "$(grep -E '^(export )?(TPU_NAME|TPU_ZONE|TPU_IP)=' "$SESSION_ENV_FILE" 2>/dev/null || true)"
        tpu_vm="${TPU_NAME:-unknown}"
        tpu_z="${TPU_ZONE:-unknown}"
        tpu_ip="${TPU_IP:-unknown}"
      fi
      echo "========================================================================"
      echo "[PRESERVED] Spot TPU VM preserved for debugging (--keep-vm-on-failure):"
      echo "  VM Name:    $tpu_vm"
      echo "  Zone:       $tpu_z"
      echo "  Project:    $ALLOWED_PROJECT"
      echo "  IP:         $tpu_ip"
      echo "To connect:   gcloud compute tpus tpu-vm ssh '$tpu_vm' --zone='$tpu_z' --project='$ALLOWED_PROJECT'"
      echo "To teardown:  $SPOT_MANAGER down --project='$ALLOWED_PROJECT'"
      echo "========================================================================"
    else
      echo "Executing Spot TPU teardown..."
      "$SPOT_MANAGER" down --project="$ALLOWED_PROJECT" || true
    fi
  fi

  trap - EXIT INT TERM HUP
  exit "$exit_code"
}

# 5. Target enumeration and query
resolve_targets() {
  local target_arg="${1:-}"
  local filter_arg="${2:-}"
  local targets=()

  if [[ -n "$target_arg" ]]; then
    # Parse explicit targets or scope wildcard
    local clean_arg
    clean_arg=$(echo "$target_arg" | tr ',' ' ')
    read -r -a raw_targets <<< "$clean_arg"

    local has_wildcard=false
    for item in "${raw_targets[@]}"; do
      if [[ "$item" == *"..."* || "$item" == *":all"* ]]; then
        has_wildcard=true
        break
      fi
    done

    if [[ "$has_wildcard" == "true" ]]; then
      for pattern in "${raw_targets[@]}"; do
        if [[ "$pattern" == *"..."* || "$pattern" == *":all"* ]]; then
          local q_out
          q_out=$(bazel query "tests(${pattern})" --output=label 2>/dev/null || true)
          while IFS= read -r line; do
            [[ -n "$line" && "$line" == //* ]] && targets+=("$line")
          done <<< "$q_out"
        else
          targets+=("$pattern")
        fi
      done
    else
      targets=("${raw_targets[@]}")
    fi
  else
    # Build query expression from filter tags
    local pos_tags=()
    local neg_tags=()

    IFS=',' read -r -a filter_items <<< "$filter_arg"
    for item in "${filter_items[@]}"; do
      item=$(echo "$item" | xargs)
      [[ -z "$item" ]] && continue
      if [[ "$item" == -* ]]; then
        neg_tags+=("${item#-}")
      else
        pos_tags+=("$item")
      fi
    done

    local query_expr="tests(//tests/...)"
    if [[ ${#pos_tags[@]} -gt 0 ]]; then
      local pos_pattern
      pos_pattern=$(IFS='|'; echo "${pos_tags[*]}")
      query_expr="${query_expr} intersect attr(tags, \"${pos_pattern}\", //tests/...)"
    fi
    if [[ ${#neg_tags[@]} -gt 0 ]]; then
      local neg_pattern
      neg_pattern=$(IFS='|'; echo "${neg_tags[*]}")
      query_expr="${query_expr} except attr(tags, \"${neg_pattern}\", //tests/...)"
    fi

    local query_out
    query_out=$(bazel query "$query_expr" --output=label 2>/dev/null || true)
    while IFS= read -r line; do
      [[ -n "$line" && "$line" == //* ]] && targets+=("$line")
    done <<< "$query_out"
  fi

  if [[ ${#targets[@]} -eq 0 ]]; then
    echo "ERROR [run_presubmit]: No test targets found matching criteria." >&2
    exit 1
  fi

  printf "%s\n" "${targets[@]}"
}

# 6. Main execution flow
main() {
  parse_args "$@"

  # Set up output directory
  if [[ -z "$CLI_OUTPUT_DIR" ]]; then
    CLI_OUTPUT_DIR="${REPO_ROOT}/presubmit_reports/run_$(date +%Y%m%d_%H%M%S)"
  fi
  mkdir -p "$CLI_OUTPUT_DIR"

  # Register lifecycle supervisor traps
  trap 'cleanup_orchestrator EXIT' EXIT
  trap 'cleanup_orchestrator INT' INT
  trap 'cleanup_orchestrator TERM' TERM
  trap 'cleanup_orchestrator HUP' HUP

  # Resolve target list
  local resolved_targets_str
  resolved_targets_str=$(resolve_targets "$CLI_TARGETS" "$CLI_FILTER")
  local target_list=()
  while IFS= read -r line; do
    [[ -n "$line" ]] && target_list+=("$line")
  done <<< "$resolved_targets_str"
  local total_count="${#target_list[@]}"

  local targets_file="${CLI_OUTPUT_DIR}/targets.txt"
  printf "%s\n" "${target_list[@]}" > "$targets_file"

  echo "========================================================================"
  echo "torch_tpu Presubmit-v5 Relay Execution Pipeline"
  echo "========================================================================"
  echo "Target Count:        $total_count"
  echo "GCP Zone:            $CLI_ZONE"
  echo "GCP Project:         $ALLOWED_PROJECT"
  echo "Tag Filter:          $CLI_FILTER"
  echo "Output Directory:    $CLI_OUTPUT_DIR"
  echo "Test Timeout:        $CLI_TEST_TIMEOUT"
  echo "Keep VM on Failure:  $CLI_KEEP_VM_ON_FAILURE"
  echo "Sequential Mode:     --local_test_jobs=1 (zero PCIe collision)"
  echo "========================================================================"

  # Handle dry-run execution
  if [[ "$CLI_DRY_RUN" == "true" ]]; then
    echo -e "\n[DRY RUN] Execution plan confirmed. Resolved test targets ($total_count):"
    for t in "${target_list[@]}"; do
      echo "  - $t"
    done
    echo -e "\n[DRY RUN] Planned Bazel invocation:"
    echo "bazel test \\"
    echo "  --run_under=\"$RELAY_RUNNER\" \\"
    echo "  --modify_execution_info=TestRunner=+no-remote-exec \\"
    echo "  --strategy=TestRunner=local \\"
    echo "  --local_test_jobs=1 \\"
    echo "  --keep_going \\"
    echo "  --nocache_test_results \\"
    echo "  --test_output=errors \\"
    echo "  --test_summary=detailed \\"
    echo "  --test_tag_filters=\"$CLI_FILTER\" \\"
    echo "  --test_timeout=\"$CLI_TEST_TIMEOUT\" \\"
    echo "  --spawn_strategy=standalone,local \\"
    for t in "${target_list[@]}"; do
      echo "  $t \\"
    done

    # Generate dry-run reporting artifacts
    python3 "$REPORTER" \
      --output-dir="$CLI_OUTPUT_DIR" \
      --workspace-root="$REPO_ROOT" \
      --targets-file="$targets_file" \
      --dry-run >/dev/null 2>&1 || true

    echo -e "\nDry run complete. Zero resources provisioned."
    return 0
  fi

  # Session management: reuse active instance or provision a new one
  if is_tpu_session_healthy; then
    echo "Found active and healthy Spot TPU session. Reusing existing instance."
    "$SPOT_MANAGER" status --project="$ALLOWED_PROJECT"
  else
    echo "No healthy active TPU session found. Initializing new Spot TPU in '$CLI_ZONE'..."
    if [[ -f "$SESSION_ENV_FILE" ]]; then
      "$SPOT_MANAGER" down --project="$ALLOWED_PROJECT" 2>/dev/null || true
    fi
    "$SPOT_MANAGER" up --zone="$CLI_ZONE" --project="$ALLOWED_PROJECT"
  fi

  # Execute Bazel test suite sequentially with job control
  echo -e "\nStarting sequential test execution through relay runner..."
  local bazel_log="${CLI_OUTPUT_DIR}/bazel_presubmit.log"
  local start_time
  start_time=$(date +%s)

  set +e
  set -m
  bazel test \
    --run_under="$RELAY_RUNNER" \
    --modify_execution_info=TestRunner=+no-remote-exec \
    --strategy=TestRunner=local \
    --local_test_jobs=1 \
    --keep_going \
    --nocache_test_results \
    --test_output=errors \
    --test_summary=detailed \
    --test_tag_filters="$CLI_FILTER" \
    --test_timeout="$CLI_TEST_TIMEOUT" \
    --spawn_strategy=standalone,local \
    "${target_list[@]}" > >(tee "$bazel_log") 2>&1 &
  _BAZEL_PID=$!

  wait "$_BAZEL_PID"
  local bazel_rc=$?
  _BAZEL_PID=""
  wait
  set -e

  local end_time
  end_time=$(date +%s)
  local duration=$(( end_time - start_time ))

  echo "Bazel execution completed in ${duration}s with exit code $bazel_rc."

  if [[ $bazel_rc -ne 0 ]]; then
    _PRESUBMIT_FAILED=1
    _FINAL_EXIT_CODE="$bazel_rc"
  fi

  # Result aggregation and report synthesis
  echo "Synthesizing presubmit test reports in '$CLI_OUTPUT_DIR'..."
  set +e
  python3 "$REPORTER" \
    --output-dir="$CLI_OUTPUT_DIR" \
    --bazel-log="$bazel_log" \
    --workspace-root="$REPO_ROOT" \
    --targets-file="$targets_file" \
    --duration="$duration" \
    --bazel-exit-code="$bazel_rc"
  local report_rc=$?
  set -e

  if [[ $_FINAL_EXIT_CODE -eq 0 && $report_rc -ne 0 ]]; then
    _FINAL_EXIT_CODE="$report_rc"
  fi

  return "$_FINAL_EXIT_CODE"
}

main "$@"
