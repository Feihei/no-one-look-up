#!/usr/bin/env bash
# Headless driver for the bevy-elements-build-loop skill.
#
# The skill does one iteration per `omp -p` call and exits, so the loop is this script calling omp
# again. Each pass is a fresh, ephemeral session that reads the loop log for its starting point.
#
# The driver stops when:
#   - $STATE/STOP exists (checked before each pass; a running pass finishes first),   exit 0
#   - a pass exits non-zero or hits LOOP_TIMEOUT,                                     exit 1
#   - LOOP_NO_PROGRESS passes in a row add no "## Iteration" entry to the loop log,   exit 2
#   - LOOP_MAX passes have run.                                                       exit 0
#   - the repo is off main or has tracked changes before a pass starts,                exit 3
#
# Detach:          cd <repo> && setsid nohup scripts/bevy-build-loop.sh </dev/null >/dev/null 2>&1 &
# Stop after pass: touch ~/.local/state/bevy-elements-loop/STOP
# Kill now:        kill the omp process; the driver then sees a failed pass and exits 1.
# Logs:            $STATE/driver.log (the driver) and $STATE/runs/<utc>-pass<n>.log (each pass).
#
# Env: LOOP_MAX (3), LOOP_TIMEOUT (6h per pass), LOOP_NO_PROGRESS (1), LOOP_STATE,
#      LOOP_REPO (defaults to this script's repo), OMP_BIN (defaults to omp).
set -euo pipefail

REPO="${LOOP_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
STATE="${LOOP_STATE:-$HOME/.local/state/bevy-elements-loop}"
MAX="${LOOP_MAX:-3}"
PASS_TIMEOUT="${LOOP_TIMEOUT:-6h}"
NO_PROGRESS="${LOOP_NO_PROGRESS:-1}"
OMP="${OMP_BIN:-omp}"
PROMPT="Run one iteration of the bevy-elements-build-loop skill."
LOG="$REPO/docs/loop-log.md"

export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

mkdir -p "$STATE/runs"
exec 9>"$STATE/lock"
if ! flock -n 9; then
  echo "another bevy-build-loop holds $STATE/lock; not starting" >&2
  exit 1
fi
exec >>"$STATE/driver.log" 2>&1

log_entries() { grep -c '^## Iteration ' "$LOG" || true; }
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "$(now) driver start: repo=$REPO max=$MAX timeout=$PASS_TIMEOUT no_progress=$NO_PROGRESS"
cd "$REPO"
stalled=0

for ((pass = 1; pass <= MAX; pass++)); do
  if [[ -e "$STATE/STOP" ]]; then
    echo "$(now) STOP present; stopping before pass $pass"
    exit 0
  fi
  if [[ "$(git -C "$REPO" rev-parse --abbrev-ref HEAD)" != main ]] || ! git -C "$REPO" diff --quiet || ! git -C "$REPO" diff --cached --quiet; then
    echo "$(now) stopping before pass $pass: repo must be on main with no tracked changes (git status --short)"
    exit 3
  fi

  before=$(log_entries)
  run_log="$STATE/runs/$(date -u +%Y%m%dT%H%M%SZ)-pass$pass.log"
  echo "$(now) pass $pass start (loop-log entries: $before) -> $run_log"

  set +e
  timeout --kill-after=60s "$PASS_TIMEOUT" "$OMP" -p "$PROMPT" --no-session >"$run_log" 2>&1
  rc=$?
  set -e

  after=$(log_entries)
  echo "$(now) pass $pass exit=$rc (loop-log entries: $before -> $after)"

  if ((rc != 0)); then
    echo "$(now) stopping: pass $pass failed (exit $rc); see $run_log"
    exit 1
  fi

  if ((after > before)); then
    stalled=0
  else
    stalled=$((stalled + 1))
    if ((stalled >= NO_PROGRESS)); then
      echo "$(now) stopping: $stalled pass(es) in a row added no loop-log entry"
      exit 2
    fi
  fi
done

echo "$(now) stopping: reached LOOP_MAX=$MAX passes"
