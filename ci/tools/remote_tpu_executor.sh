#!/usr/bin/env bash
# ci/tools/remote_tpu_executor.sh
# Remote execution engine running on physical Cloud TPU v5e VM silicon.
set -uo pipefail
set -m  # Enable job control for dedicated process group isolation

readonly ALLOWED_PROJECT="rbe-tpu-oss"

# Enforce GCP project boundary strictly to ALLOWED_PROJECT
enforce_project_boundary() {
  local target_project="${1:-$ALLOWED_PROJECT}"
  if [[ "$target_project" != "$ALLOWED_PROJECT" ]]; then
    echo "ERROR [remote_tpu_executor]: Project boundary violation: '$target_project' is not allowed." >&2
    echo "This executor is strictly restricted to project '$ALLOWED_PROJECT'." >&2
    exit 1
  fi

  local env_vars=("CLOUDSDK_CORE_PROJECT" "GOOGLE_CLOUD_PROJECT" "GCP_PROJECT" "GCLOUD_PROJECT" "TPU_PROJECT")
  for var in "${env_vars[@]}"; do
    if [[ -n "${!var:-}" && "${!var}" != "$ALLOWED_PROJECT" ]]; then
      echo "ERROR [remote_tpu_executor]: Project boundary violation: Environment variable $var is set to '${!var}' (violates allowed project '$ALLOWED_PROJECT')." >&2
      exit 1
    fi
  done
}

# Initial boundary check across environment variables
enforce_project_boundary "${TPU_PROJECT:-$ALLOWED_PROJECT}"

# Argument parsing: support both positional and named flag syntax
SANDBOX_DIR=""
TEST_BIN_REL=""
TARGET_LABEL=""
CLI_PROJECT=""
declare -a TEST_ARGS=()
declare -a EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sandbox-dir)
      [[ $# -lt 2 ]] && { echo "ERROR [remote_tpu_executor]: --sandbox-dir requires an argument." >&2; exit 1; }
      SANDBOX_DIR="$2"
      shift 2
      ;;
    --sandbox-dir=*)
      SANDBOX_DIR="${1#*=}"
      shift
      ;;
    --test-binary)
      [[ $# -lt 2 ]] && { echo "ERROR [remote_tpu_executor]: --test-binary requires an argument." >&2; exit 1; }
      TEST_BIN_REL="$2"
      shift 2
      ;;
    --test-binary=*)
      TEST_BIN_REL="${1#*=}"
      shift
      ;;
    --target-label)
      [[ $# -lt 2 ]] && { echo "ERROR [remote_tpu_executor]: --target-label requires an argument." >&2; exit 1; }
      TARGET_LABEL="$2"
      shift 2
      ;;
    --target-label=*)
      TARGET_LABEL="${1#*=}"
      shift
      ;;
    --project)
      [[ $# -lt 2 ]] && { echo "ERROR [remote_tpu_executor]: --project requires an argument." >&2; exit 1; }
      CLI_PROJECT="$2"
      enforce_project_boundary "$CLI_PROJECT"
      shift 2
      ;;
    --project=*)
      CLI_PROJECT="${1#*=}"
      enforce_project_boundary "$CLI_PROJECT"
      shift
      ;;
    --)
      shift
      TEST_ARGS+=("$@")
      EXTRA_ARGS+=("$@")
      break
      ;;
    -*)
      if [[ -n "$SANDBOX_DIR" && -n "$TEST_BIN_REL" ]]; then
        TEST_ARGS+=("$1")
        EXTRA_ARGS+=("$1")
      else
        echo "WARNING [remote_tpu_executor]: Unknown option $1" >&2
      fi
      shift
      ;;
    *)
      if [[ -z "$SANDBOX_DIR" ]]; then
        SANDBOX_DIR="$1"
      elif [[ -z "$TEST_BIN_REL" ]]; then
        TEST_BIN_REL="$1"
      else
        TEST_ARGS+=("$1")
        EXTRA_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

if [[ -z "$SANDBOX_DIR" || -z "$TEST_BIN_REL" ]]; then
  echo "ERROR [remote_tpu_executor]: Missing required arguments." >&2
  echo "Usage: $0 <sandbox_dir> <test_binary_rel> [args...]" >&2
  echo "   or: $0 --sandbox-dir=<dir> --test-binary=<bin> [--target-label=<label>] [--project=<project>] [-- args...]" >&2
  exit 1
fi
# 1. Unpack incoming runfiles tar stream from stdin into sandbox directory
mkdir -p "$SANDBOX_DIR"

if [[ ! -t 0 ]]; then
  # Stdin is a pipe or file; unpack stream
  tar -xzf - -C "$SANDBOX_DIR" 2>/dev/null || true
fi

# Locate workspace root inside sandbox
WORKSPACE_ROOT="${SANDBOX_DIR}/_main"
if [[ ! -d "$WORKSPACE_ROOT" ]]; then
  WORKSPACE_ROOT="$SANDBOX_DIR"
fi

# 2. Symlink base preheated C++ libraries and python venv
BASE_CACHE="${TORCH_TPU_BASE_CACHE:-/tmp/torch_tpu_relay/base}"
if [[ -d "$BASE_CACHE" ]]; then
  if [[ -d "${BASE_CACHE}/_solib_x86_64" ]]; then
    mkdir -p "$WORKSPACE_ROOT"
    ln -sfn "${BASE_CACHE}/_solib_x86_64" "${WORKSPACE_ROOT}/_solib_x86_64"
  fi
  if [[ -d "${BASE_CACHE}/torch_tpu/common" ]]; then
    mkdir -p "${WORKSPACE_ROOT}/torch_tpu"
    ln -sfn "${BASE_CACHE}/torch_tpu/common" "${WORKSPACE_ROOT}/torch_tpu/common"
  fi
fi

# Resolve Python interpreter (preheated venv or host fallback)
PYTHON_BIN="/tmp/tpu_venv/bin/python3"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi

# Symlink python venv stubs for rules_python compatibility
mkdir -p "${SANDBOX_DIR}/rules_python++python+python_3_12_x86_64-unknown-linux-gnu/bin"
if [[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]]; then
  ln -sfn "$PYTHON_BIN" "${SANDBOX_DIR}/rules_python++python+python_3_12_x86_64-unknown-linux-gnu/bin/python3"
  while IFS= read -r venv_dir; do
    mkdir -p "${venv_dir}/bin"
    ln -sfn "$PYTHON_BIN" "${venv_dir}/bin/python3"
  done < <(find "${SANDBOX_DIR}" -maxdepth 3 \( -name "*_test.venv" -o -name "*.venv" \) -type d 2>/dev/null || true)
fi

# 3. Configure hardware environment for Cloud TPU v5e (v5litepod-1)
export TPU_VISIBLE_DEVICES="${TPU_VISIBLE_DEVICES:-0}"
export TPU_VISIBLE_CHIPS="${TPU_VISIBLE_CHIPS:-0}"
export TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v5litepod-1}"
export TPU_CHIPS_PER_HOST_BOUNDS="${TPU_CHIPS_PER_HOST_BOUNDS:-1,1,1}"
export TPU_SKIP_MDS_QUERY="${TPU_SKIP_MDS_QUERY:-1}"
export ALLOW_MULTIPLE_LIBTPU_LOAD=true

# Resolve libtpu.so
if [[ -z "${TPU_LIBRARY_PATH:-}" ]]; then
  if [[ -f "/tmp/tpu_venv/lib/python3.12/site-packages/libtpu/libtpu.so" ]]; then
    export TPU_LIBRARY_PATH="/tmp/tpu_venv/lib/python3.12/site-packages/libtpu/libtpu.so"
  elif [[ -f "${BASE_CACHE}/libtpu.so" ]]; then
    export TPU_LIBRARY_PATH="${BASE_CACHE}/libtpu.so"
  elif [[ -f "/lib/libtpu.so" ]]; then
    export TPU_LIBRARY_PATH="/lib/libtpu.so"
  elif [[ -n "$PYTHON_BIN" ]]; then
    _dyn_libtpu=$("$PYTHON_BIN" -c "import libtpu; print(libtpu.get_library_path())" 2>/dev/null || true)
    if [[ -n "$_dyn_libtpu" && -f "$_dyn_libtpu" ]]; then
      export TPU_LIBRARY_PATH="$_dyn_libtpu"
    fi
  fi
fi

# 4. Set process and library search paths
export PYTHONUNBUFFERED=1
export PATH="/tmp/tpu_venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:${PATH:-}"
export PYTHONPATH="${WORKSPACE_ROOT}:${WORKSPACE_ROOT}/tests:${WORKSPACE_ROOT}/torch_tpu:${BASE_CACHE}/torch_tpu/common:/tmp/tpu_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${WORKSPACE_ROOT}/_solib_x86_64:${WORKSPACE_ROOT}/torch_tpu/common:${BASE_CACHE}/_solib_x86_64:${BASE_CACHE}/torch_tpu/common:/tmp/tpu_venv/lib/python3.12/site-packages/torch/lib:/tmp/tpu_venv/lib/python3.12/site-packages/libtpu:${LD_LIBRARY_PATH:-}"
export XML_OUTPUT_FILE="${SANDBOX_DIR}/test.xml"
export TEST_SRCDIR="${SANDBOX_DIR}"
export RUNFILES_DIR="${SANDBOX_DIR}"
export TEST_WORKSPACE="_main"
export TEST_TMPDIR="${SANDBOX_DIR}/tmp"
mkdir -p "${SANDBOX_DIR}/tmp"

# 5. Clear stale accelerator device locks before launch
if [[ -d "/dev/vfio" ]] && command -v fuser >/dev/null 2>&1; then
  fuser -k /dev/vfio/* 2>/dev/null || true
fi

# State tracking for supervision and cleanup
TEST_PID=""
TEST_PGID=""
TEST_EXIT_CODE=0
CLEANUP_DONE=0
TARGET_NAME="${TARGET_LABEL:-${TEST_TARGET:-$(basename "$TEST_BIN_REL")}}"
START_TIME_MS=$(date +%s%3N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1000))' 2>/dev/null || echo "0")

synthesize_fallback_xml() {
  local code="$1"
  local msg="${2:-Test exited with code $code}"
  local xml_path="${SANDBOX_DIR}/test.xml"

  local duration_s="0.000"
  if [[ "$START_TIME_MS" != "0" ]]; then
    local now_ms
    now_ms=$(date +%s%3N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1000))' 2>/dev/null || echo "0")
    if [[ "$now_ms" != "0" && "$now_ms" -ge "$START_TIME_MS" ]]; then
      duration_s=$(awk -v s="$START_TIME_MS" -v e="$now_ms" 'BEGIN { printf "%.3f", (e - s) / 1000 }' 2>/dev/null || echo "0.000")
    fi
  fi

  local failures=0
  local errors=0
  local failure_tag=""
  if [[ "$code" -ne 0 ]]; then
    failures=1
    failure_tag="<failure message=\"${msg}\"><![CDATA[Process failed with exit code ${code}. Check test.log for stderr details.]]></failure>"
  fi

  cat <<XML_EOF > "$xml_path"
<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="${TARGET_NAME}" tests="1" failures="${failures}" errors="${errors}" time="${duration_s}">
    <testcase name="execution" classname="${TARGET_NAME}" time="${duration_s}">
      ${failure_tag}
    </testcase>
  </testsuite>
</testsuites>
XML_EOF
}

cleanup() {
  local trap_sig="${1:-EXIT}"
  local origin_rc="${2:-0}"
  if [[ "$CLEANUP_DONE" -eq 1 ]]; then
    return
  fi
  CLEANUP_DONE=1
  trap - EXIT INT TERM HUP

  case "$trap_sig" in
    INT)  TEST_EXIT_CODE=130 ;;
    TERM) TEST_EXIT_CODE=143 ;;
    HUP)  TEST_EXIT_CODE=129 ;;
    EXIT)
      if [[ "$TEST_EXIT_CODE" -eq 0 && "$origin_rc" -ne 0 ]]; then
        TEST_EXIT_CODE="$origin_rc"
      fi
      ;;
    *)    TEST_EXIT_CODE=1 ;;
  esac

  # Signal traps (EXIT INT TERM HUP) executing process group reaping
  if [[ -n "${TEST_PGID:-}" && "$TEST_PGID" -gt 1 ]]; then
    if kill -0 -"$TEST_PGID" 2>/dev/null; then
      kill -TERM -"$TEST_PGID" 2>/dev/null || true
      sleep 0.2
      kill -KILL -"$TEST_PGID" 2>/dev/null || true
      sleep 0.3
    fi
  fi

  # Free any dangling hardware file descriptor locks on /dev/vfio/*
  if [[ -d "/dev/vfio" ]] && command -v fuser >/dev/null 2>&1; then
    fuser -k /dev/vfio/* 2>/dev/null || true
  fi

  # Record exit code for host runner retrieval and disambiguation
  echo "$TEST_EXIT_CODE" > "${SANDBOX_DIR}/test.exitcode"

  # Ensure JUnit XML exists; synthesize fallback if missing or empty
  if [[ ! -s "${SANDBOX_DIR}/test.xml" ]]; then
    synthesize_fallback_xml "$TEST_EXIT_CODE" "Process terminated with exit code ${TEST_EXIT_CODE} (${trap_sig})"
  fi

  # Update latest pointer symlinks
  mkdir -p /tmp/torch_tpu_relay/sandboxes 2>/dev/null || true
  ln -sfn "${SANDBOX_DIR}/test.xml" "/tmp/torch_tpu_relay/sandboxes/$(basename "$SANDBOX_DIR")_latest.xml" 2>/dev/null || true
  ln -sfn "${SANDBOX_DIR}/test.xml" "/tmp/torch_tpu_relay/latest_test.xml" 2>/dev/null || true

  exit "$TEST_EXIT_CODE"
}

trap 'cleanup EXIT $?' EXIT
trap 'cleanup INT 130' INT
trap 'cleanup TERM 143' TERM
trap 'cleanup HUP 129' HUP

# 6. Resolve test command to execute
CLEAN_BIN="${TEST_BIN_REL#./}"
STAGE2_BOOTSTRAP="$(find "${WORKSPACE_ROOT}" -name "_$(basename "$CLEAN_BIN")_stage2_bootstrap.py" 2>/dev/null | head -n 1 || true)"

declare -a EXEC_CMD=()
if [[ -f "${WORKSPACE_ROOT}/${CLEAN_BIN}.py" ]]; then
  EXEC_CMD=("$PYTHON_BIN" "${WORKSPACE_ROOT}/${CLEAN_BIN}.py" "${TEST_ARGS[@]}")
elif [[ -f "${WORKSPACE_ROOT}/${CLEAN_BIN}" && "${CLEAN_BIN}" == *.py ]]; then
  EXEC_CMD=("$PYTHON_BIN" "${WORKSPACE_ROOT}/${CLEAN_BIN}" "${TEST_ARGS[@]}")
elif [[ -n "$STAGE2_BOOTSTRAP" && -f "$STAGE2_BOOTSTRAP" ]]; then
  EXEC_CMD=("$PYTHON_BIN" "$STAGE2_BOOTSTRAP" "${TEST_ARGS[@]}")
elif [[ -x "${WORKSPACE_ROOT}/${CLEAN_BIN}" ]]; then
  EXEC_CMD=("${WORKSPACE_ROOT}/${CLEAN_BIN}" "${TEST_ARGS[@]}")
elif [[ -x "${SANDBOX_DIR}/${CLEAN_BIN}" ]]; then
  EXEC_CMD=("${SANDBOX_DIR}/${CLEAN_BIN}" "${TEST_ARGS[@]}")
elif [[ -f "${SANDBOX_DIR}/${CLEAN_BIN}" && "${CLEAN_BIN}" == *.py ]]; then
  EXEC_CMD=("$PYTHON_BIN" "${SANDBOX_DIR}/${CLEAN_BIN}" "${TEST_ARGS[@]}")
else
  # Flexible directory walk fallback
  MATCH=$(find "$WORKSPACE_ROOT" \( -name "$(basename "$CLEAN_BIN")" -o -name "$(basename "$CLEAN_BIN").py" \) 2>/dev/null | head -n 1 || true)
  if [[ -n "$MATCH" ]]; then
    if [[ "$MATCH" == *.py ]]; then
      EXEC_CMD=("$PYTHON_BIN" "$MATCH" "${TEST_ARGS[@]}")
    elif [[ -x "$MATCH" ]]; then
      EXEC_CMD=("$MATCH" "${TEST_ARGS[@]}")
    else
      EXEC_CMD=("$PYTHON_BIN" "$MATCH" "${TEST_ARGS[@]}")
    fi
  else
    echo "ERROR [remote_tpu_executor]: Test executable '${CLEAN_BIN}' not found in sandbox." >&2
    exit 127
  fi
fi

# 7. Execute test binary under job control (set -m) in dedicated process group ($PGID)
(
  cd "$WORKSPACE_ROOT"
  exec "${EXEC_CMD[@]}" < /dev/null
) &
TEST_PID=$!
TEST_PGID=$!

wait "$TEST_PID" 2>/dev/null
TEST_EXIT_CODE=$?
TEST_PID=""

exit "$TEST_EXIT_CODE"
