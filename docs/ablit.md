# Optional Keys abliteration overlay

Stock weights remain the default. To enable the overlay for one launch:

```bash
./start.sh restart --abliterated
# Or: ABLIT=1 ./start.sh restart

# Return to stock:
./start.sh restart --stock
```

Switching requires restarting both ranks. Without an explicit flag or `ABLIT=1`,
the next launch restores stock, including supervisor restarts. To persist the
overlay, set `ABLIT=1` in `scripts/local.sh`; `--stock` overrides that setting
for one launch. The API name and request format stay the same.

## Overlay and compatibility

Uses [drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated](https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated),
revision `de412d2249632903dace23399526e00a145aada5`, file
`exl3_wo_b_l10_35_mia29.safetensors`, SHA256
`dcefb4a735f4985e4a5097cad69a059d85453f3bba148af3fc98269c7ee1ff7c`.
First download requires access to the gated repository: accept its terms with
your Hugging Face account and provide `HF_TOKEN` or the usual cached token.

This recipe serves the **easthambuilds** quant, rather than the Mia pack for
which this sidecar was published. Both use EXL3 mul1 K=5 for `wo_b`: trellis
`[512,320,80]`, F16 `suh[8192]`, F16 `svh[5120]`. The integration checks the
sidecar shapes and compares every selected cache fragment with the original
checkpoint before saving a backup. The author's Mia evaluation does not
establish quality for this combination.

Only `attn.wo_b` in layers 10–35 changes. Base checkpoint shards, DSpark/MTP
(including its dtypes), vision, experts and Engram stay untouched. Each rank
cache has 78 affected entries.

## Storage and recovery

Needs about **1.3 GiB extra per Spark**: the 651 MiB sidecar plus stock and
overlay cache fragments, about 325 MiB each. No full checkpoint or rank-cache
copy is needed. Python's standard library handles the cache and safetensors
format; no extra packages or engine rebuild are required.

Backups live under `$DATA_DIR/ablit/rank0` and `rank1`. Keep them while the
cache contains the overlay. Both ranks are checked before stopping the
service; cache writes require the engine and cache builder stopped. If Docker
created the cache as root, switching requires passwordless `sudo` on that
Spark; this is checked before stopping. Ownership is preserved.

Every backup and written range is verified by SHA256. A durable pending marker
allows an interrupted switch to be repeated from its backups. Unexpected cache
bytes, damaged backups, or a changed cache identity cause the launch to abort.

Restore stock before rebuilding caches or changing the engine image. The
launcher rejects automatic setup when a changed image would rebuild an active
overlay cache. Direct `scripts/prepare.sh` calls do not manage overlay state.

## Validation

Tested on two GB10 Sparks: stock -> overlay -> stock -> overlay, arithmetic
(`17*23=391`), Python function generation, and Spanish prose in both modes.
All 156 affected cache entries were verified against the selected mode.

Eight automated tests cover byte-exact restoration, running-engine rejection,
damaged backups, unexpected cache bytes, interrupted writes, changed cache
identity, incompatible pack shapes and expected-mode verification:

```bash
python3 -B -m unittest discover -s tests -p test_ablit_cache.py -v
```
