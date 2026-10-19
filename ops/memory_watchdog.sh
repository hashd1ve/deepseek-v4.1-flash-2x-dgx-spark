#!/bin/bash
# memory_watchdog.sh [CONTAINER]: every 0.5 s, stops the engine's container when the host's memory is really gone, before
# a GB10 stalls: MemFree under 3 GiB with under 1 GiB of file cache left to reclaim, or memory PSI "full" avg10 over 30,
# two readings in a row. Page cache counts as reclaimable (MemFree alone stops a healthy server). Run it with nohup on
# each Spark; it logs each new low and every stop to ~/dsv41_watchdog.log.
CONTAINER=${1:-dsv41-1017-tf}
LOG=$HOME/dsv41_watchdog.log; low=999999999; bad=0
echo "$(date +%T) armed for $CONTAINER (MemFree<3G and cache<1G, or PSI full>30)" >> $LOG
while true; do
  read f c < <(awk '/^MemFree:/{f=$2} /^Active\(file\):/{a=$2} /^Inactive\(file\):/{i=$2} END{print f, a+i}' /proc/meminfo)
  p=$(awk '/^full/{split($2,x,"="); print int(x[2])}' /proc/pressure/memory)
  if [ "$f" -lt "$low" ]; then low=$f; echo "$(date +%T) lowest MemFree $((f/1024)) MiB (cache $((c/1024)) MiB, psi $p)" >> $LOG; fi
  if { [ "$f" -lt 3145728 ] && [ "$c" -lt 1048576 ]; } || [ "$p" -gt 30 ]; then bad=$((bad+1)); else bad=0; fi
  if [ $bad -ge 2 ]; then docker kill "$CONTAINER" >/dev/null 2>&1; echo "$(date +%T) STOPPED: MemFree $((f/1024)) MiB, cache $((c/1024)) MiB, psi $p" >> $LOG; bad=0; fi
  sleep 0.5
done
