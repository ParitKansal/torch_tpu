#!/usr/bin/env bash
# ci/tools/relay_test_runner.sh
# Bazel --run_under test runner wrapper for physical Cloud TPU execution.
set -uo pipefail

readonly ALLOWED_PROJECT="rbe-tpu-oss"

readonly TEST_BIN="${1:-}"
if [[ -z "$TEST_BIN" ]]; then
  echo "ERROR [relay_test_runner]: No test binary specified." >&2
  echo "Usage: $0 <test_binary> [test_args...]" >&2
  exit 1
fi
shift || true
readonly TEST_ARGS=("$@")

readonly SESSION_ENV="${TPU_SESSION_ENV:-/tmp/tpu_active_session.env}"
readonly TARGET_LABEL="${TEST_TARGET:-$(basename "$TEST_BIN")}"
readonly CLEAN_TARGET="$(echo "$TARGET_LABEL" | tr '/:' '__')"
readonly SANDBOX_ID="${CLEAN_TARGET}_$$_${RANDOM}"
readonly REMOTE_SANDBOX="/tmp/torch_tpu_relay/sandboxes/${SANDBOX_ID}"
readonly TIMEOUT_RAW="${TEST_TIMEOUT:-900}"
_clean_timeout="${TIMEOUT_RAW%[sS]}"
if [[ "$_clean_timeout" =~ ^[0-9]+$ && "$_clean_timeout" -gt 0 ]]; then
  readonly TIMEOUT_SEC="$_clean_timeout"
else
  readonly TIMEOUT_SEC="900"
fi

# Helper: write fallback JUnit XML
write_fallback_xml() {
  local failure_msg="${1:-Test execution failed}"
  local failure_detail="${2:-No additional details}"
  local xml_path="${XML_OUTPUT_FILE:-}"

  if [[ -n "$xml_path" && ! -s "$xml_path" ]]; then
    mkdir -p "$(dirname "$xml_path")"
    cat <<XML_EOF > "$xml_path"
<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="${CLEAN_TARGET}" tests="1" failures="1" errors="0" time="0.0">
    <testcase classname="${CLEAN_TARGET}" name="execution" time="0.0">
      <failure message="${failure_msg}">
<![CDATA[${failure_detail}]]>
      </failure>
    </testcase>
  </testsuite>
</testsuites>
XML_EOF
  fi
}

write_success_xml() {
  local xml_path="${XML_OUTPUT_FILE:-}"
  if [[ -n "$xml_path" && ! -s "$xml_path" ]]; then
    mkdir -p "$(dirname "$xml_path")"
    cat <<XML_EOF > "$xml_path"
<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="${CLEAN_TARGET}" tests="1" failures="0" errors="0" time="0.0">
    <testcase classname="${CLEAN_TARGET}" name="execution" time="0.0"/>
  </testsuite>
</testsuites>
XML_EOF
  fi
}

# Host-level trap for cleanup on premature termination
cleanup_host() {
  local origin_rc=$?
  local trap_sig="${1:-EXIT}"
  trap - EXIT INT TERM HUP
  local exit_code=1
  case "$trap_sig" in
    INT)  exit_code=130 ;;
    TERM) exit_code=143 ;;
    HUP)  exit_code=129 ;;
    EXIT) exit_code="$origin_rc" ;;
    *)    exit_code="$origin_rc" ;;
  esac

  if [[ -n "${REMOTE_SANDBOX:-}" && -n "${SSH_CONTROL_PATH:-}" && -S "${SSH_CONTROL_PATH:-}" && -n "${TPU_IP:-}" && -n "${SSH_USER:-}" ]]; then
    ssh -S "$SSH_CONTROL_PATH" -o BatchMode=yes \
      -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o ConnectTimeout=2 \
      "${SSH_USER}@${TPU_IP}" \
      "rm -rf '${REMOTE_SANDBOX}'" 2>/dev/null || true
  fi

  if [[ "$exit_code" -ne 0 ]]; then
    write_fallback_xml "Test interrupted by signal ${trap_sig}" "Process received signal ${trap_sig} (exit code ${exit_code})."
  fi

  exit "$exit_code"
}
trap 'cleanup_host EXIT' EXIT
trap 'cleanup_host INT' INT
trap 'cleanup_host TERM' TERM
trap 'cleanup_host HUP' HUP

# Enforce GCP project boundary strictly to ALLOWED_PROJECT
enforce_project_boundary() {
  local target_project="${1:-$ALLOWED_PROJECT}"
  if [[ -n "$target_project" && "$target_project" != "$ALLOWED_PROJECT" ]]; then
    echo "ERROR [relay_test_runner]: Project boundary violation: '$target_project' is not allowed." >&2
    echo "This runner is strictly restricted to project '$ALLOWED_PROJECT'." >&2
    write_fallback_xml "Project Boundary Violation" "Project '$target_project' violates allowed project '$ALLOWED_PROJECT'."
    exit 1
  fi

  local env_vars=("CLOUDSDK_CORE_PROJECT" "GOOGLE_CLOUD_PROJECT" "GCP_PROJECT" "GCLOUD_PROJECT")
  for var in "${env_vars[@]}"; do
    if [[ -n "${!var:-}" && "${!var}" != "$ALLOWED_PROJECT" ]]; then
      echo "ERROR [relay_test_runner]: Project boundary violation: Environment variable $var is set to '${!var}' (violates allowed project '$ALLOWED_PROJECT')." >&2
      write_fallback_xml "Project Boundary Violation" "Environment variable $var violates allowed project '$ALLOWED_PROJECT'."
      exit 1
    fi
  done
}

# 1. Environment discovery and session validation
enforce_project_boundary "${TPU_PROJECT:-$ALLOWED_PROJECT}"

if [[ ! -f "$SESSION_ENV" ]]; then
  echo "ERROR [relay_test_runner]: Active TPU session file $SESSION_ENV not found." >&2
  echo "Provision a Spot TPU first using: scripts/spot_tpu_manager.sh up" >&2
  write_fallback_xml "No active TPU session" "Session file $SESSION_ENV not found on host."
  exit 1
fi

# shellcheck disable=SC1090
source "$SESSION_ENV"

TPU_PROJECT="$ALLOWED_PROJECT"

if [[ -z "${TPU_IP:-}" || -z "${SSH_USER:-}" || -z "${SSH_CONTROL_PATH:-}" ]]; then
  echo "ERROR [relay_test_runner]: Incomplete TPU session environment in $SESSION_ENV." >&2
  write_fallback_xml "Incomplete TPU session" "Required variables missing in $SESSION_ENV (TPU_IP, SSH_USER, or SSH_CONTROL_PATH)."
  exit 1
fi

# 2. ControlMaster socket health check, auto-reconnection, and preemption detection
check_ssh_socket() {
  [[ -n "${SSH_CONTROL_PATH:-}" && -S "${SSH_CONTROL_PATH}" ]] && \
  ssh -O check -S "$SSH_CONTROL_PATH" \
    -o BatchMode=yes \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    "${SSH_USER}@${TPU_IP}" 2>/dev/null
}

reconnect_ssh() {
  echo "WARNING [relay_test_runner]: SSH ControlMaster socket is not active. Attempting reconnection..." >&2
  rm -f "$SSH_CONTROL_PATH"

  local ssh_identity_args=()
  if [[ -f "${HOME}/.ssh/google_compute_engine" ]]; then
    ssh_identity_args+=("-i" "${HOME}/.ssh/google_compute_engine")
  fi

  ssh -M -N -f \
    -S "$SSH_CONTROL_PATH" \
    -o ControlPersist=1h \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o IdentitiesOnly=yes \
    ${ssh_identity_args[@]+"${ssh_identity_args[@]}"} \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=4 \
    -o BatchMode=yes \
    -o ConnectTimeout=10 \
    "${SSH_USER}@${TPU_IP}" 2>/dev/null || true

  check_ssh_socket
}

check_preemption() {
  local vm_state
  vm_state=$(gcloud compute tpus tpu-vm describe "${TPU_NAME:-}" \
    --project="$ALLOWED_PROJECT" \
    --zone="${TPU_ZONE:-europe-west4-b}" \
    --format="value(state)" 2>/dev/null || echo "NOT_FOUND")

  if [[ "$vm_state" == "PREEMPTED" || "$vm_state" == "TERMINATED" || "$vm_state" == "NOT_FOUND" ]]; then
    echo "FATAL [relay_test_runner]: Spot TPU VM '${TPU_NAME:-}' in zone '${TPU_ZONE:-europe-west4-b}' was PREEMPTED by Google Cloud (state: $vm_state)." >&2
    write_fallback_xml "Spot TPU Preempted" "VM ${TPU_NAME:-} entered state $vm_state during run."
    return 0
  else
    echo "ERROR [relay_test_runner]: Cannot connect to TPU VM at ${TPU_IP} via SSH (state: $vm_state)." >&2
    write_fallback_xml "SSH Connection Failed" "Host unreachable at ${TPU_IP} (state: $vm_state)."
    return 1
  fi
}

if ! check_ssh_socket; then
  if ! reconnect_ssh; then
    check_preemption || true
    exit 255
  fi
fi

# 3. Synchronize remote_tpu_executor.sh to remote TPU VM
ensure_remote_executor() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  local local_executor="${script_dir}/remote_tpu_executor.sh"
  if [[ ! -f "$local_executor" ]]; then
    local_executor="/usr/local/google/home/jonathanskim/torch_tpu/ci/tools/remote_tpu_executor.sh"
  fi

  if [[ -f "$local_executor" ]]; then
    local local_hash
    local_hash=$(sha256sum "$local_executor" 2>/dev/null | awk '{print $1}')
    ssh -S "$SSH_CONTROL_PATH" -o BatchMode=yes \
      -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      "${SSH_USER}@${TPU_IP}" \
      "if [ \"\$(sha256sum /tmp/torch_tpu_relay/remote_tpu_executor.sh 2>/dev/null | awk '{print \$1}')\" != \"$local_hash\" ]; then mkdir -p /tmp/torch_tpu_relay && cat > /tmp/torch_tpu_relay/remote_tpu_executor.sh && chmod +x /tmp/torch_tpu_relay/remote_tpu_executor.sh; fi" < "$local_executor" 2>/dev/null || true
  fi
}
ensure_remote_executor

# 4. Resolve runfiles root directory
RUNFILES_ROOT="${TEST_SRCDIR:-}"
if [[ -z "$RUNFILES_ROOT" || ! -d "$RUNFILES_ROOT" ]]; then
  if [[ -d "${TEST_BIN}.runfiles" ]]; then
    RUNFILES_ROOT="${TEST_BIN}.runfiles"
  elif [[ -d "../_main" ]]; then
    RUNFILES_ROOT="$(cd .. && pwd)"
  elif [[ -d "./_main" ]]; then
    RUNFILES_ROOT="$(pwd)"
  elif [[ -d "${WORKSPACE_ROOT:-}" ]]; then
    RUNFILES_ROOT="${WORKSPACE_ROOT}"
  else
    RUNFILES_ROOT="$(dirname "$TEST_BIN")"
  fi
fi

# Safely escape test arguments for remote shell execution
remote_args=""
if [[ ${#TEST_ARGS[@]} -gt 0 ]]; then
  printf -v remote_args "%q " "${TEST_ARGS[@]}"
fi

# 5. Delta runfiles streaming and remote execution
tar -czhf - \
  --ignore-failed-read \
  --exclude='*rules_python*' \
  --exclude='*._*.venv' \
  --exclude='*.venv' \
  --exclude='*/_solib_x86_64*' \
  --exclude='_solib_x86_64' \
  --exclude='*/torch_tpu/common*' \
  --exclude='torch_tpu/common' \
  --exclude='*.a' \
  --exclude='*.o' \
  --exclude='*.params' \
  --exclude='*.cppmap' \
  --exclude='*__pycache__*' \
  --exclude='*.pyc' \
  --exclude='MANIFEST' \
  --exclude='_repo_mapping' \
  -C "${RUNFILES_ROOT}" . 2>/dev/null | \
  timeout --foreground --signal=TERM --kill-after=10s "${TIMEOUT_SEC}s" \
  ssh -S "$SSH_CONTROL_PATH" \
    -o BatchMode=yes \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    "${SSH_USER}@${TPU_IP}" \
    "/tmp/torch_tpu_relay/remote_tpu_executor.sh '${REMOTE_SANDBOX}' '${TEST_BIN}' -- ${remote_args}"

pipestatus=("${PIPESTATUS[@]}")
ssh_rc="${pipestatus[1]:-0}"

# 6. Fetch remote test metadata (exitcode and test.xml) and clean remote sandbox
remote_meta=$(ssh -S "$SSH_CONTROL_PATH" -o BatchMode=yes \
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "${SSH_USER}@${TPU_IP}" \
  "cat '${REMOTE_SANDBOX}/test.exitcode' 2>/dev/null || true; echo '---TORCH_TPU_SPLIT---'; cat '${REMOTE_SANDBOX}/test.xml' 2>/dev/null || true; rm -rf '${REMOTE_SANDBOX}'" 2>/dev/null || echo "")

remote_exitcode=$(echo "$remote_meta" | awk '/---TORCH_TPU_SPLIT---/{exit} {print}' | tr -d ' \r\n')
remote_xml=$(echo "$remote_meta" | awk 'f{print} /---TORCH_TPU_SPLIT---/{f=1}')

# Disarm cleanup trap since remote sandbox was deleted above
trap - EXIT INT TERM HUP

# 7. Write JUnit XML artifact
xml_written=0
if [[ -n "$remote_xml" && "$remote_xml" =~ \<testsuite && -n "${XML_OUTPUT_FILE:-}" ]]; then
  mkdir -p "$(dirname "$XML_OUTPUT_FILE")"
  echo "$remote_xml" > "$XML_OUTPUT_FILE"
  xml_written=1
fi

# 8. Exit code resolution and SSH 255 disambiguation
if [[ "$ssh_rc" -eq 124 ]]; then
  echo "ERROR [relay_test_runner]: Test action timed out after ${TIMEOUT_SEC}s." >&2
  write_fallback_xml "Test Timeout" "Test execution timed out after ${TIMEOUT_SEC} seconds."
  exit 124
fi

if [[ "$ssh_rc" -eq 255 ]]; then
  if [[ "$remote_exitcode" == "255" ]]; then
    # Test legitimately exited with code 255
    if [[ "$xml_written" -eq 0 ]]; then
      write_fallback_xml "Test failed with exit code 255" "Process explicitly exited with code 255."
    fi
    exit 255
  fi

  # Transport error or VM preemption occurred
  check_preemption || true
  exit 255
fi

# For any non-zero exit code without valid XML, synthesize fallback
if [[ "$ssh_rc" -ne 0 && "$xml_written" -eq 0 ]]; then
  write_fallback_xml "Test failed with exit code $ssh_rc" "Process exited with code $ssh_rc without generating valid test.xml."
fi

# If successful but XML missing, synthesize success XML
if [[ "$ssh_rc" -eq 0 && "$xml_written" -eq 0 ]]; then
  write_success_xml
fi

exit "$ssh_rc"
