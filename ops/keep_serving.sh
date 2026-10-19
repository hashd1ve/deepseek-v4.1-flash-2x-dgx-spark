#!/bin/bash
# keep_serving.sh: on the head, starts the server again when it went down on its own (a rank crashed, or
# memory_watchdog.sh stopped it) instead of leaving clients hanging until someone notices. Every 30 s, when either rank
# is not running, ./stop.sh did not stop it (DATA_DIR/.stopped) and no start.sh is running, it waits for both Sparks'
# memory to come back and runs ./start.sh: at most 3 times in 2 hours, then it waits for a person. Stop the server with
# ./stop.sh, never docker stop, or this starts it again. Run it next to the recipe:
#   nohup ops/keep_serving.sh > /dev/null 2>&1 &
ROOT=$(cd "$(dirname "$0")/.." && pwd)
. "$ROOT/scripts/common.sh"
set +eu
LOG=$DATA_DIR/keep_serving.log
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
avail() { awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo; }
restarts=()
gave_up=0
log "watching $CONTAINER"
while true; do
  sleep 30
  [ -f "$DATA_DIR/.stopped" ] && { gave_up=0; continue; }
  pgrep -f 'start\.sh' > /dev/null && continue
  running_here "$CONTAINER" && running_there "$CONTAINER" && { gave_up=0; continue; }
  now=$(date +%s)
  recent=()
  for t in "${restarts[@]}"; do [ $((now - t)) -lt 7200 ] && recent+=("$t"); done
  restarts=("${recent[@]}")
  if [ ${#restarts[@]} -ge 3 ]; then
    [ $gave_up = 1 ] || log "down again after 3 restarts in 2 hours: not restarting until ./start.sh (see the logs in $DATA_DIR/logs)"
    gave_up=1
    continue
  fi
  log "down (rank 0 running: $(running_here "$CONTAINER" && echo yes || echo no), rank 1: $(running_there "$CONTAINER" && echo yes || echo no)): restarting"
  # the dead rank's memory comes back within seconds; a rank still serving alone holds its ~105 GiB until start.sh stops it
  for _ in $(seq 20); do [ "$(avail)" -ge 8 ] && break; sleep 15; done
  restarts+=("$now")
  if "$ROOT/start.sh" restart >> "$LOG" 2>&1 < /dev/null; then log "serving again"; else log "start.sh failed (its output above)"; fi
done
