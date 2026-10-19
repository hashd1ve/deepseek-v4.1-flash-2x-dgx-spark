#!/bin/bash
# memory_peak.sh CMD...: rank 0's RSS before and at its peak while CMD runs (run on the head, next to the server), e.g.
#   bash memory_peak.sh python3 chat_messages.py chat 40 2000
P=$(pgrep -f "^/opt/tf/native/bin/tensorfold-native" | head -1)
[ -n "$P" ] || { echo "no tensorfold-native process here"; exit 1; }
rss() { awk '/^VmRSS/{print int($2/1024)}' /proc/$P/status; }
( while true; do rss; sleep 0.05; done ) > /tmp/memory_peak.txt & S=$!
b=$(rss); "$@"; kill $S
echo "rank 0 RSS: ${b} MiB before, $(sort -n /tmp/memory_peak.txt | tail -1) MiB at the peak"
