#!/usr/bin/env bash
# Wrapper: runs train.py in its own session (setsid), logs any signal received.
set -uo pipefail

LOG=/tmp/train-monitor-$(date +%Y%m%d-%H%M%S).log
PIDFILE=/tmp/train-wrapper.pid

echo "=== wrapper started pid=$$ ppid=$PPID at $(date) ===" | tee "$LOG"
echo "=== log file: $LOG ==="

echo "$$" > "$PIDFILE"

_on_signal() {
    local sig=$1
    echo "=== received SIG${sig} at $(date) ===" >> "$LOG"
    echo "=== parent processes at time of signal: ===" >> "$LOG"
    ps -p $$ -o pid,ppid,comm= 2>/dev/null >> "$LOG" || true
    ps -A -o pid,ppid,comm= 2>/dev/null | awk -v pid=$$ '
        function parents(p,   q) { if (p in pp) { parents(pp[p]); print pp[p], p, cmd[p] } }
        { pp[$1]=$2; cmd[$1]=$3 }
        END { parents(pid) }
    ' >> "$LOG" 2>/dev/null || true
    echo "=== sending SIG${sig} to train.py ===" >> "$LOG"
    kill -${sig} "$CHILD_PID" 2>/dev/null || true
    exit 1
}

trap '_on_signal TERM' TERM
trap '_on_signal HUP'  HUP
trap '_on_signal INT'  INT

# Run in own process group so pkill/kill-tree doesn't catch us by group
set -m
HOME=/Users/czj .venv/bin/python train.py "$@" >> "$LOG" 2>&1 &
CHILD_PID=$!
echo "=== train.py pid=$CHILD_PID ===" | tee -a "$LOG"

wait "$CHILD_PID"
EXIT_CODE=$?
echo "=== train.py exited code=$EXIT_CODE at $(date) ===" | tee -a "$LOG"
rm -f "$PIDFILE"
exit "$EXIT_CODE"
