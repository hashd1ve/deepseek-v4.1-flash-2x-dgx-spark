#!/bin/bash
# stop.sh: stop both ranks and free their GPU memory; each rank's log is saved first (gzipped, under DATA_DIR/logs)
. "$(dirname "$0")/scripts/common.sh"
step "Stopping $CONTAINER on both Sparks"
remove_both "$CONTAINER"
remove_both "$CONTAINER-inputs" >/dev/null
touch "$DATA_DIR/.stopped"  # stopped on purpose: ops/keep_serving.sh leaves it down until start.sh
note "stopped"
