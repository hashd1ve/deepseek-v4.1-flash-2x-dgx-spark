# Sourced by start.sh after common.sh. The overlay is opt-in at each launch.
ABLIT=${ABLIT:-0}
ABLIT_REPO=drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated
ABLIT_REV=de412d2249632903dace23399526e00a145aada5
ABLIT_FILE=exl3_wo_b_l10_35_mia29.safetensors
ABLIT_SHA=dcefb4a735f4985e4a5097cad69a059d85453f3bba148af3fc98269c7ee1ff7c

# Do not let automatic setup rebuild a cache while it contains an overlay.
if [ -f "$DATA_DIR/ablit/rank0/pending" ] ||
   [ "$(cat "$DATA_DIR/ablit/rank0/active" 2>/dev/null || true)" = abliterated ]; then
  [ "$(cut -d' ' -f1 "$DATA_DIR/.prepared" 2>/dev/null || true)" = "$IMAGE" ] \
    || die "restore --stock using the previous IMAGE before rebuilding the rank caches"
fi

ablit_rank() { # action, mode, rank, data directory; runs on the corresponding host
  local action=$1 mode=$2 rank=$3 dd=$4
  local cmd="python3 $dd/tools/ablit_cache.py $action --cache $dd/cache/tf-ds-rank/rank${rank}of2-*.bin --state $dd/ablit/rank$rank --sidecar $dd/ablit/keys/$ABLIT_FILE --model $dd/model --mode $mode --container $CONTAINER"
  if [ "$action" = switch ]; then
    # The Python cache builder runs as root inside Docker; keep its ownership.
    cmd="if [ -w $dd/cache/tf-ds-rank/rank${rank}of2-*.bin ]; then $cmd; else sudo -n $cmd; fi"
  fi
  if [ "$rank" = 0 ]; then bash -c "$cmd"; else on_worker "$cmd"; fi
}

ablit_prepare() {
  case "$ABLIT" in 0|1) ;; *) die "ABLIT must be 0 or 1" ;; esac
  if [ "$ABLIT" = 0 ] && [ ! -f "$DATA_DIR/ablit/rank0/manifest.json" ]; then
    # A worker backup without the head's is an incomplete setup, not stock.
    on_worker "test ! -f $WORKER_DATA_DIR/ablit/rank1/manifest.json" \
      || die "worker has an overlay backup but head does not; recover before launch"
    return
  fi
  if [ "$ABLIT" = 1 ]; then
    mkdir -p "$DATA_DIR/ablit/keys" "$DATA_DIR/tools"
    if [ ! -f "$DATA_DIR/ablit/keys/$ABLIT_FILE" ]; then
      # Reuse a download already on the worker before contacting the gated Hub.
      if on_worker "test -f $WORKER_DATA_DIR/ablit/keys/$ABLIT_FILE"; then
        rsync -a "$WORKER:$WORKER_DATA_DIR/ablit/keys/$ABLIT_FILE" "$DATA_DIR/ablit/keys/"
      else
        python3 "$ROOT/tools/hub_fetch.py" "$ABLIT_REPO" "$ABLIT_REV" "$DATA_DIR/ablit/keys" "$ABLIT_FILE" \
          || die "overlay download failed; accept the Hugging Face repository terms with your account and set HF_TOKEN"
      fi
    fi
    echo "$ABLIT_SHA  $DATA_DIR/ablit/keys/$ABLIT_FILE" | sha256sum -c -
    on_worker "mkdir -p $WORKER_DATA_DIR/ablit/keys $WORKER_DATA_DIR/tools"
    rsync -a "$DATA_DIR/ablit/keys/$ABLIT_FILE" "$WORKER:$WORKER_DATA_DIR/ablit/keys/"
    on_worker "echo '$ABLIT_SHA  $WORKER_DATA_DIR/ablit/keys/$ABLIT_FILE' | sha256sum -c -"
  fi
  cp "$ROOT/tools/ablit_cache.py" "$DATA_DIR/tools/ablit_cache.py"
  rsync -a "$ROOT/tools/ablit_cache.py" "$WORKER:$WORKER_DATA_DIR/tools/"
  # Validate and save both ranks before stopping any running service.
  for rank in 0 1; do
    dd=$DATA_DIR; [ "$rank" = 1 ] && dd=$WORKER_DATA_DIR
    ablit_rank prepare stock "$rank" "$dd"
    cmd="test -w $dd/cache/tf-ds-rank/rank${rank}of2-*.bin || sudo -n test -w $dd/cache/tf-ds-rank/rank${rank}of2-*.bin"
    if [ "$rank" = 0 ]; then bash -c "$cmd"; else on_worker "$cmd"; fi
  done
}

ablit_switch() {
  [ -f "$DATA_DIR/ablit/rank0/manifest.json" ] || return 0
  local mode=stock
  [ "$ABLIT" = 1 ] && mode=abliterated
  step "Weights: $mode (checkpoint unchanged)"
  # Both preflight checks run before either cache is modified.
  ablit_rank check "$mode" 0 "$DATA_DIR"
  ablit_rank check "$mode" 1 "$WORKER_DATA_DIR"
  ablit_rank switch "$mode" 0 "$DATA_DIR"
  ablit_rank switch "$mode" 1 "$WORKER_DATA_DIR"
}
