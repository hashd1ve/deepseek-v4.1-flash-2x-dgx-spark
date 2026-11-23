#!/bin/bash
# start.sh [restart] [--stock|--abliterated] [--taps|--no-taps] [--edits|--no-edits] [tensorfold-native serve arguments...]:
# set up both Sparks if needed, start rank 1 on the worker and rank 0 (the API) here, wait until they load, and send one
# smoke-test request through both.
. "$(dirname "$0")/scripts/common.sh"
. "$ROOT/scripts/ablit.sh"
RESTART=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    restart) RESTART=1; shift ;;
    --abliterated) ABLIT=1; shift ;;
    --stock) ABLIT=0; shift ;;
    --taps) TAPS=1; shift ;;
    --no-taps) TAPS=0; shift ;;
    --edits) EDITS=1; shift ;;
    --no-edits) EDITS=0; shift ;;
    *) break ;;
  esac
done
mkdir -p "$DATA_DIR"
exec 9>"$DATA_DIR/.launch.lock"
flock -n 9 || die "another start.sh is running"
rm -f "$DATA_DIR/.stopped"  # (stop.sh's mark: ops/keep_serving.sh may start it again when it goes down)

step "Setup"
if [ "$(cat "$DATA_DIR/.prepared" 2>/dev/null | cut -d' ' -f1)" = "$IMAGE" ]; then
  note "ready ($DATA_DIR/.prepared)"
else
  bash "$ROOT/scripts/prepare.sh"
fi

ablit_prepare
step "Checks"
if running_here "$CONTAINER" || running_there "$CONTAINER"; then
  [ $RESTART = 1 ] || die "already running: ./start.sh restart, or ./stop.sh"
  note "stopping the running server (requests in flight are cut off)"
  remove_both "$CONTAINER"
else
  remove_both "$CONTAINER" >/dev/null   # a stopped container from an earlier run; its log is kept
fi
ablit_switch
# GB10 shares memory between GPU and CPU and reclaims page cache too slowly for CUDA (tools/evict_page_cache.py)
note "here: $(python3 "$ROOT/tools/evict_page_cache.py" "$DATA_DIR"/{model,engram,cache})"
note "$WORKER: $(on_worker python3 - "$WORKER_DATA_DIR"/{model,engram,cache} < "$ROOT/tools/evict_page_cache.py")"
for side in here there; do
  if [ $side = here ]; then avail=$(free -g | awk '/^Mem:/{print $7}'); else avail=$(on_worker "free -g" | awk '/^Mem:/{print $7}'); fi
  [ "$avail" -ge 110 ] || note "warning: only $avail GiB memory available $side (the engine wants ~110): stop other GPU work"
done
ss -ltn | grep -q ":$PORT " && die "port $PORT is in use here"

step "Launch"
# --taps (or TAPS=1 in scripts/local.sh, which ops/keep_serving.sh's restarts keep; docker/patches/0026): a request with
# "taps": "name" has its prompt pass write the streams entering each of TAPS_LAYERS to DATA_DIR/xtaps/name.xtaps.
# --edits (or EDITS=1; 0027; it turns the taps on too): a request with "edits": "name" has its prompt pass change the
# streams as DATA_DIR/xtaps/name.xedit says, and with "taps" its "lens" and "streams" go to the taps file; both ranks
# get EDITS_MIB of edit rows. TAPS_ROWS bounds a marked prompt (smaller buffers, so every layer can be listed).
# Off, the engine allocates and launches nothing for either and ignores the fields. See docs/taps.md.
TAPS_OPTS=""
[ "${EDITS:-0}" = 1 ] && TAPS=1
if [ "${TAPS:-0}" = 1 ]; then
  mkdir -p "$DATA_DIR/xtaps"
  TAPS_OPTS="-v $DATA_DIR/xtaps:/xtaps -e TF_DS_XTAPS=${TAPS_LAYERS:-8,12,16,20,24,28,32,36,40} -e TF_DS_XTAPS_DIR=/xtaps ${TAPS_ROWS:+-e TF_DS_XTAPS_ROWS=$TAPS_ROWS}"
  note "taps on: layers ${TAPS_LAYERS:-8,12,16,20,24,28,32,36,40}, files in $DATA_DIR/xtaps${TAPS_ROWS:+, marked prompts of at most $TAPS_ROWS tokens}"
fi
if [ "${EDITS:-0}" = 1 ]; then
  ENGINE_ENV="${ENGINE_ENV:-} -e TF_DS_XEDIT=${EDITS_MIB:-16}"
  note "edits on: ${EDITS_MIB:-16} MiB of edit rows on each rank, files in $DATA_DIR/xtaps"
fi
on_worker "$(rank1_cmd)" >/dev/null
note "rank 1 started on $WORKER"
eval "$(rank0_cmd "$@")" >/dev/null
note "rank 0 started here; loading"

step "Loading"
t0=$(date +%s)
until [ "$(api /v1/models -o /dev/null -w '%{http_code}')" = 200 ]; do
  if ! running_here "$CONTAINER" || ! running_there "$CONTAINER"; then
    echo "-- rank 0:"; docker logs --tail 25 "$CONTAINER" 2>&1
    echo "-- rank 1:"; on_worker "docker logs --tail 25 $CONTAINER" 2>&1
    remove_both "$CONTAINER"
    die "a rank stopped while loading (logs above)"
  fi
  [ $(( $(date +%s) - t0 )) -lt 3600 ] || die "not up after an hour: docker logs $CONTAINER"
  sleep 15
  note "$(( $(date +%s) - t0 )) s: $(docker logs --tail 1 "$CONTAINER" 2>&1 | cut -c1-110)"
done
note "up after $(( $(date +%s) - t0 )) s"

step "Smoke test"
reply=$(api /v1/chat/completions -H 'Content-Type: application/json' -d "{\"model\": \"$MODEL_NAME\",
  \"messages\": [{\"role\": \"user\", \"content\": \"What is 17 times 3? Answer with the number only.\"}],
  \"max_tokens\": 2000}")
echo "$reply" | python3 -c "import json,sys; m=json.load(sys.stdin)['choices'][0]['message']; print('   reply:', (m.get('content') or '').strip()[:200])" \
  || die "the smoke test failed: $reply"
echo "$reply" | grep -q '51' || note "warning: the reply does not contain 51"

h=$HOST; [ "$h" = 0.0.0.0 ] && h=$(hostname -I | awk '{print $1}')
step "$MODEL_NAME is live on http://$h:$PORT/v1"
