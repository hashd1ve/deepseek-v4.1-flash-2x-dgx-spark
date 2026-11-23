# Reading and editing the residual streams (optional)

`docker/patches/0026` and `0027` let a request look inside the model while it is served: its prompt pass can write the
residual streams entering chosen layers to a file (taps), read the head's logits at each of those layers (a logit
lens), and change the streams before a layer reads them (edits: steering, activation patching, directional
ablation). Both are off unless turned on at start; off, the engine allocates and launches nothing for them and ignores
the request fields, so served requests are unchanged.

## Turning them on

```bash
./start.sh restart --taps            # taps only, for this start
./start.sh restart --edits           # edits, lens and streams too (turns the taps on)
./start.sh restart --no-taps --no-edits
```

Or in `scripts/local.sh`, which `ops/keep_serving.sh`'s restarts keep:

```bash
TAPS=1
TAPS_LAYERS=8,12,16,20,24,28,32,36,40   # the layers whose input streams a marked pass writes (0-40; 41: after the last)
TAPS_ROWS=256                           # a marked prompt's tokens at most (smaller buffers, so every layer can be listed)
EDITS=1
EDITS_MIB=32                            # MiB of edit rows on each rank
```

Files go to `DATA_DIR/xtaps` on the head (`/xtaps` in rank 0's container). With all 41 layers at 256 rows, the taps
hold 184 MiB and the edits 31 MiB of GPU memory on rank 0, the edits 31 MiB on rank 1.

## A marked request

A chat request with `"taps": "name"` has its prompt pass write `DATA_DIR/xtaps/name.xtaps`. With `--edits`:

| Field | |
| --- | --- |
| `"edits": "name"` | change the streams as `DATA_DIR/xtaps/name.xedit` says |
| `"lens": 1` / `2` | with `"taps"`: for each tapped layer, the head's logits on the last position's streams entering it (1: the top logits; 2: the whole row, with the layer's own pre and each stream alone) |
| `"streams": K` | with `"taps"`: the streams themselves (not their mean) of the prompt's last K positions |

A name is 1 to 64 of `[A-Za-z0-9_.-]`, not starting with a dot. A marked pass runs every layer on every row from the
prompt's first token: it neither resumes nor keeps a prompt state, and nothing cuts into it. Only the prompt pass is
tapped or edited; the tokens generated after it are decoded as ever, on the edited prompt's caches. A name that ends
in `.replay` keeps the served bounded replay.

## `name.xtaps` (written by rank 0, little endian)

`"XTAP"`, u32 version, u32 hidden, u32 n, n u32 layer ids; then records, each a u32 kind:

1. u64 len, len u32: the prompt's ids (an image position holds the image token's).
2. u32 layer, u64 start, u32 rows, rows × hidden bf16: the mean of the streams entering the layer (what a DSpark tap reads).
3. u32 vocab, vocab f32: the prompt's logits at its last position.
4. u32 layer, u32 mode, u32 vocab, vocab f32: a lens row (`"lens": 2`); mode 0 with the layer's own pre, mode 1 + s on stream s alone.
5. u32 layer, u32 mode, f32 logsumexp, u32 k, k × (u32 id, f32 logit): the same row's largest logits (`"lens": 1`).
6. u32 layer, u32 sub (0 attention, 1 FFN), u64 start, u32 rows, u32 hc, then f32 pre [rows, hc], post [rows, hc] and
   comb [rows, hc, hc]: how each sublayer reads the streams and writes back into them, for every layer.
7. u32 layer, u64 start, u32 rows, u32 hc, rows × hc × hidden bf16: the streams themselves (`"streams": K`).

It is written as `name.xtaps.part` and renamed when the pass ends with its first token drawn. The engine's
`zig/src/families/dsv41/xtaps.zig` has the format.

## `name.xedit` (read by rank 0, sent to rank 1)

`"XEDT"`, u32 version, u32 hidden, u32 n; then n edits, each u32 op, u32 layer, i64 start, i64 end, f32 scale,
u32 rows, u32 mask, u32 0, rows × hidden f32. Positions are `[start, end)` of the prompt; a negative one counts from
its end, and `end` = 2^63−1 is its end. `mask` bit s is stream s (0: every stream).

| op | | rows |
| --- | --- | --- |
| 1 add | h_s += scale · v (steering) | 1 |
| 2 patch | h_s += scale · (t_p − mean_s h_s): at scale 1 the streams' mean becomes t_p, what another pass's tap holds | end − start |
| 3 ablate | h_s −= scale · (h_s·v / v·v) v (directional ablation) | 1 |
| 4 set | h_s += scale · (t_p − h_s): at scale 1 the stream becomes t_p | end − start |

`zig/src/families/dsv41/xedit.zig` has the details.

## Cost

On our pair, with both on at all 41 layers: the same 12 greedy replies, tool calls and template ids as without them, and
the same prompt speed for unmarked requests (`tests/prefill_speed.py`, 50K tokens: 2,040–2,058 tok/s on, 2,043–2,066
off).
