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

# Brings up a fleet of Spot TPU v5e VMs and writes one session file per VM into
# a pool directory. Point TPU_SESSION_POOL at that directory and
# relay_test_runner.sh will lease a VM per test, so a single Bazel invocation
# spreads its test actions across the whole fleet.
#
# Usage:
#   scripts/spot_tpu_fleet.sh up --size 10 [--pool DIR] [--zone ZONE] [--on-demand]
#                                [--deadline-minutes N]
#   scripts/spot_tpu_fleet.sh down [--pool DIR]
#   scripts/spot_tpu_fleet.sh status [--pool DIR]
#
# `up` arms a detached deadline that runs `down` after --deadline-minutes, so a
# fleet cannot outlive a crashed orchestrator. Pass 0 to turn it off. `down`
# cancels it.

set -uo pipefail

readonly ALLOWED_PROJECT="rbe-tpu-oss"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SPOT_MANAGER="${SCRIPT_DIR}/spot_tpu_manager.sh"
readonly DEFAULT_POOL="/tmp/torch_tpu_relay/pool"
# Matches the names spot_tpu_manager.sh generates. The orphan sweep filters on
# this so it can never touch a TPU VM someone else in the project created.
readonly VM_NAME_PREFIX="spot-tpu-v5e-"

# Spot capacity moves around between zones, so spread the fleet rather than
# betting the whole run on one zone having room. us-east5-a is deliberately
# absent: rbe-tpu-oss has no v5e reservation there and every create returns
# "Reservation not found".
readonly DEFAULT_ZONES=("europe-west4-b" "us-central1-a")

POOL_DIR="$DEFAULT_POOL"
FLEET_SIZE=10
ZONES=()
ON_DEMAND=false
# Long enough that no presubmit suite gets cut off (a 57-target run takes about
# 50 minutes), short enough that a forgotten fleet is hours of billing, not days.
DEADLINE_MINUTES=180
REAPER_PID_FILE=""

die() {
  echo "ERROR [spot_tpu_fleet]: $*" >&2
  exit 1
}

log() {
  echo "[spot_tpu_fleet] $*"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --size|--size=*)
        [[ "$1" == *=* ]] && FLEET_SIZE="${1#*=}" || { FLEET_SIZE="${2:-}"; shift; }
        shift ;;
      --on-demand)
        ON_DEMAND=true
        shift ;;
      --pool|--pool=*)
        [[ "$1" == *=* ]] && POOL_DIR="${1#*=}" || { POOL_DIR="${2:-}"; shift; }
        shift ;;
      --zone|--zone=*)
        local zone
        [[ "$1" == *=* ]] && zone="${1#*=}" || { zone="${2:-}"; shift; }
        ZONES+=("$zone")
        shift ;;
      --project|--project=*)
        local project
        [[ "$1" == *=* ]] && project="${1#*=}" || { project="${2:-}"; shift; }
        [[ "$project" == "$ALLOWED_PROJECT" ]] \
          || die "This fleet is restricted to project '$ALLOWED_PROJECT'."
        shift ;;
      --deadline-minutes|--deadline-minutes=*)
        [[ "$1" == *=* ]] && DEADLINE_MINUTES="${1#*=}" || { DEADLINE_MINUTES="${2:-}"; shift; }
        shift ;;
      -h|--help)
        sed -n '16,29p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
        exit 0 ;;
      *)
        die "Unknown option '$1'." ;;
    esac
  done

  [[ "$FLEET_SIZE" =~ ^[0-9]+$ && "$FLEET_SIZE" -gt 0 ]] \
    || die "--size must be a positive integer, got '$FLEET_SIZE'."
  [[ "$DEADLINE_MINUTES" =~ ^[0-9]+$ ]] \
    || die "--deadline-minutes must be a non-negative integer, got '$DEADLINE_MINUTES'."
  [[ ${#ZONES[@]} -gt 0 ]] || ZONES=("${DEFAULT_ZONES[@]}")
  REAPER_PID_FILE="${POOL_DIR}/reaper.pid"
}

# Brings up one VM and drops its session file in the pool. Runs as a background
# job, so it reports failure through its exit status and its own log file.
# Starts in the zone it was assigned, then works through the rest, because Spot
# capacity runs out one zone at a time.
provision_one() {
  local index="$1"
  local first_zone="$2"
  local session="${POOL_DIR}/vm_${index}.env"
  local log_file="${POOL_DIR}/vm_${index}.log"

  local ordered_zones=("$first_zone")
  local zone
  for zone in "${ZONES[@]}"; do
    [[ "$zone" == "$first_zone" ]] || ordered_zones+=("$zone")
  done

  for zone in "${ordered_zones[@]}"; do
    local up_args=(--zone="$zone" --project="$ALLOWED_PROJECT")
    [[ "$ON_DEMAND" == "true" ]] && up_args+=(--on-demand)
    if TPU_SESSION_ENV="$session" "$SPOT_MANAGER" up \
        "${up_args[@]}" >>"$log_file" 2>&1; then
      log "vm_${index} ready in ${zone}"
      return 0
    fi
    log "vm_${index} could not start in ${zone}"
    rm -f "$session"
  done

  log "vm_${index} failed in every zone (see ${log_file})"
  return 1
}

cmd_up() {
  [[ -x "$SPOT_MANAGER" ]] || die "$SPOT_MANAGER is missing."
  mkdir -p "$POOL_DIR"

  local pids=() existing=0
  local i
  for (( i = 0; i < FLEET_SIZE; i++ )); do
    # Re-running `up` after a partial capacity failure should fill the gaps,
    # not stand up a second VM for a slot that already has one.
    if [[ -s "${POOL_DIR}/vm_${i}.env" ]]; then
      existing=$(( existing + 1 ))
      continue
    fi
    local zone="${ZONES[$(( i % ${#ZONES[@]} ))]}"
    provision_one "$i" "$zone" &
    pids+=("$!")
  done

  local kind="Spot"
  [[ "$ON_DEMAND" == "true" ]] && kind="on-demand"
  log "Bringing up ${#pids[@]} ${kind} TPU v5e VMs across ${ZONES[*]} (${existing} already up)"

  local ready="$existing" pid
  for pid in "${pids[@]}"; do
    wait "$pid" && ready=$(( ready + 1 ))
  done

  log "Fleet ready: ${ready}/${FLEET_SIZE} VMs in ${POOL_DIR}"
  [[ "$ready" -gt 0 ]] || die "No VMs came up. Check ${POOL_DIR}/vm_*.log."
  arm_deadline
  return 0
}

# The orchestrator tears the fleet down on exit, but a `kill -9` or a dropped
# connection skips that and the VMs bill until someone notices. This arms a
# detached process that runs the same teardown at a fixed deadline.
arm_deadline() {
  [[ "$DEADLINE_MINUTES" -gt 0 ]] || return 0

  local args=(deadline --pool "$POOL_DIR" --deadline-minutes "$DEADLINE_MINUTES")
  local zone
  for zone in "${ZONES[@]}"; do
    args+=(--zone "$zone")
  done

  nohup "${BASH_SOURCE[0]}" "${args[@]}" >>"${POOL_DIR}/reaper.log" 2>&1 &
  local reaper_pid="$!"
  disown "$reaper_pid" 2>/dev/null || true
  echo "$reaper_pid" > "${REAPER_PID_FILE}"
  log "Deadline armed: fleet self-deletes in ${DEADLINE_MINUTES}m (pid ${reaper_pid})"
}

cmd_deadline() {
  log "Deadline armed for ${DEADLINE_MINUTES}m (pid $$)"
  sleep $(( DEADLINE_MINUTES * 60 ))
  log "Deadline reached, tearing the fleet down"
  cmd_down
}

# A reaper left running after a manual teardown would delete whatever fleet
# happens to be up when its deadline lands.
cancel_deadline() {
  [[ -f "$REAPER_PID_FILE" ]] || return 0
  local reaper_pid
  reaper_pid="$(cat "$REAPER_PID_FILE")"
  rm -f "$REAPER_PID_FILE"
  [[ "$reaper_pid" =~ ^[0-9]+$ && "$reaper_pid" != "$$" ]] || return 0
  kill "$reaper_pid" 2>/dev/null && log "Cancelled deadline (pid ${reaper_pid})"
  return 0
}

cmd_down() {
  cancel_deadline
  local pids=() session
  if [[ -d "$POOL_DIR" ]]; then
    for session in "$POOL_DIR"/*.env; do
      [[ -f "$session" ]] || continue
      TPU_SESSION_ENV="$session" "$SPOT_MANAGER" down --project="$ALLOWED_PROJECT" \
        >>"${session%.env}.log" 2>&1 &
      pids+=("$!")
    done

    local pid
    for pid in "${pids[@]}"; do
      wait "$pid" || true
    done

    rm -f "$POOL_DIR"/*.env "$POOL_DIR"/*.env.lock "$POOL_DIR"/*.quarantine
  fi

  sweep_orphans
  log "Fleet torn down. Verify with: $0 status"
}

# Deletes fleet VMs that no session file knows about. Provisioning can create
# the node and then fail before writing its session, and the pool has no record
# of it, so the loop above walks straight past it while it keeps billing.
sweep_orphans() {
  local pids=() zone leftover
  for zone in "${ZONES[@]}"; do
    while read -r leftover; do
      [[ -n "$leftover" ]] || continue
      log "deleting unrecorded ${leftover} in ${zone}"
      gcloud compute tpus tpu-vm delete "$leftover" \
        --zone="$zone" --project="$ALLOWED_PROJECT" --quiet >/dev/null 2>&1 &
      pids+=("$!")
    done < <(gcloud compute tpus tpu-vm list \
      --zone="$zone" --project="$ALLOWED_PROJECT" \
      --filter="name~${VM_NAME_PREFIX}" \
      --format="value(name.basename())" 2>/dev/null)
  done

  local pid
  for pid in "${pids[@]}"; do
    wait "$pid" || true
  done
}

cmd_status() {
  local zone
  for zone in "${DEFAULT_ZONES[@]}"; do
    echo "-- ${zone}"
    gcloud compute tpus tpu-vm list \
      --project="$ALLOWED_PROJECT" --zone="$zone" 2>&1 | sed 's/^/   /'
  done

  echo "-- pool ${POOL_DIR}"
  ls -1 "$POOL_DIR"/*.env 2>/dev/null | sed 's/^/   /' || echo "   (empty)"

  echo "-- deadline"
  local reaper_pid=""
  [[ -f "$REAPER_PID_FILE" ]] && reaper_pid="$(cat "$REAPER_PID_FILE")"
  if [[ -n "$reaper_pid" ]] && kill -0 "$reaper_pid" 2>/dev/null; then
    echo "   armed (pid ${reaper_pid})"
  else
    echo "   not armed"
  fi
}

main() {
  local subcommand="${1:-}"
  [[ $# -gt 0 ]] && shift
  parse_args "$@"

  case "$subcommand" in
    up)     cmd_up ;;
    down)   cmd_down ;;
    deadline) cmd_deadline ;;
    status) cmd_status ;;
    *)      die "Usage: $0 <up|down|status> [options]" ;;
  esac
}

main "$@"
