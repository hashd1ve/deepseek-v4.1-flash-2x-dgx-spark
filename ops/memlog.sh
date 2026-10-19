#!/bin/bash
# memlog.sh [CONTAINER]: every 30 s, the host's memory and the engine's (cgroup, GPU, RSS), to see what grows while it serves
CONTAINER=${1:-dsv41-1017-tf}
out=$HOME/dsv41_memlog.log
while true; do
  m=$(awk '/^MemFree:/{f=$2} /^Active\(file\):/{a=$2} /^Inactive\(file\):/{i=$2} /^AnonPages:/{n=$2} /^Slab:/{s=$2} /^Shmem:/{h=$2} /^SwapFree:/{w=$2} /^SwapTotal:/{t=$2} END{printf "free=%d cache=%d anon=%d slab=%d shmem=%d swap=%d", f/1024, (a+i)/1024, n/1024, s/1024, h/1024, (t-w)/1024}' /proc/meminfo)
  id=$(docker inspect -f '{{.Id}}' "$CONTAINER" 2>/dev/null)
  cg=$(cat /sys/fs/cgroup/system.slice/docker-$id.scope/memory.current 2>/dev/null || echo 0)
  g=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | head -1)
  r=$(ps -C tensorfold-native,tf-dsv41-lanes -o rss= 2>/dev/null | head -1)
  echo "$(date +%T) $m cgroup=$((cg/1048576)) gpu=${g:-0} rss=$(( ${r:-0}/1024 ))" >> $out
  sleep 30
done
