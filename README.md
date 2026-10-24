# DeepSeek-V4.1-Flash on 2× DGX Spark, for agent work

Serve [**DeepSeek-V4.1-Flash-1017**](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (EXL3, 2.9 bpw) from
two NVIDIA DGX Sparks (GB10, 128 GB each, linked by their ConnectX-7 ports) through an OpenAI-compatible API: the full
**1,048,576-token window**, **4 requests at once**, images, tool calls and DSpark drafting. On top of that, the fixes it
needed to carry long agent sessions all day (OpenCode, with subagents and screenshots).

It starts from [easthambuilds' recipe](https://github.com/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-2x-DGX-Sparks-TensorFold)
(Apache-2.0). The engine is [Bertholomus AI](https://github.com/bertholomus)'s `dsv41` family for
[TensorFold](https://github.com/ashhart/TensorFold). This repository adds:

| | |
| --- | --- |
| Engine | [`bertholomus/TensorFold`](https://github.com/bertholomus/TensorFold) `deepseek-v41-zig` @ `2aa3c47d`, which includes five of our fixes ([#24](https://github.com/bertholomus/TensorFold/pull/24), [#25](https://github.com/bertholomus/TensorFold/pull/25), [#26](https://github.com/bertholomus/TensorFold/pull/26), [#28](https://github.com/bertholomus/TensorFold/pull/28), [#29](https://github.com/bertholomus/TensorFold/pull/29)), plus [`docker/patches/`](docker/patches): tool calls streamed as they are written (`0005`), a request's memory held to its prompt (`0006`-`0008`), the text before tool calls kept as the content (`0009`, from Mia's AI Lab's recipe), quoted control tokens kept as text (`0010`, [#36](https://github.com/bertholomus/TensorFold/pull/36)) and the reasoning ended at the model's own close token (`0011`-`0013`, from TensorFold's main) |
| Chat template | [`model/chat_template.jinja`](model/chat_template.jinja): the published template, rendering the same prompt without rank 0's memory growing with the conversation |
| Recipe | easthambuilds' scripts with the changes from [easthambuilds#1](https://github.com/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-2x-DGX-Sparks-TensorFold/pull/1) (the 512/513-token kit variant, kept rank caches, a CDI preflight, settings for a tight memory budget) and prompt passes in pieces |
| Operations | [`ops/`](ops): the kernel memory settings, the memory watchdog and a supervisor that starts the server again when it goes down on its own |
| Checks | [`tests/`](tests): the checks behind every number below |

## What changed, and why

Each of these was a failure in daily use. The numbers are from our pair; [docs/FINDINGS.md](docs/FINDINGS.md) has the
details.

- **Rank 0's memory grew with each long conversation.** The chat template built the prompt in one growing string, and
  the engine copies it on every append: ~3 GiB a request at 288K tokens, enough for our memory watchdog to stop the
  server twice in 20 minutes. The template now writes each piece out as it is made; the rendered prompt is identical.
  40K tokens in 2,000 messages: +1,555 MiB at rank 0's peak before, +244 MiB after.
- **A subagent or a new session made the main session reprocess everything.** A new conversation resumed the shared
  system prompt inside the main one's extent and staled its kept states. The main session's next turn ran from 0
  (3–4 min at ~400K tokens). Fixed in [#26](https://github.com/bertholomus/TensorFold/pull/26) (merged); the main
  session now resumes at 67,584 of 69,286 tokens in our repro.
- **A session with screenshots reprocessed its whole history on every turn** (73K tokens, ~40 s), because prompts
  with images were never cached. [#28](https://github.com/bertholomus/TensorFold/pull/28) keys the cache by each
  image's digest: the next turn resumes past the image with the same reply.
- **A long prompt stopped every other session**, for as long as it ran (~4 min for a 400K-token reload).
  [#29](https://github.com/bertholomus/TensorFold/pull/29) runs it in pieces between the others' rounds. Another stream
  now gets 100 text chunks during a new 57K-token prompt (none before), and that prompt takes 32.7 s instead of 28.9.
- **A tool call quoted inside a file the model writes could stream as a real call**, and the model's own call was
  then never sent. [#25](https://github.com/bertholomus/TensorFold/pull/25) reads DSML blocks first and streams calls
  only from the text before them, as the engine's Python server already does.
- **A long tool call looked like a hang.** The engine held a DSML calls block until the reply ended, so while the model
  wrote a long call the client saw nothing: ~4 min still on one token in one of our sessions. `docker/patches/0005`
  streams each call's name as soon as it is written and its arguments as they come, as OpenAI-compatible servers do.
- **A session with screenshots got the server stopped by its own memory watchdog.** Each request cost rank 0
  +1.3 GiB at ~200K tokens with 30 screenshots (the client sends them all again on every turn): the tokenizer kept
  ~800 bytes a token, every screenshot was decoded again, the vision tower ran on all of them, and all of it stayed
  until the reply ended. `0006`-`0008` bring that to +43 MiB, with the same tokens and replies, and make each such turn
  ~10 s faster. When a rank goes down anyway, [`ops/keep_serving.sh`](ops/keep_serving.sh) starts the server again.
- **The text a reply wrote before its tool calls was lost.** Without streaming the server returned `"content": null`
  whenever a reply made calls; streamed, the calls' two newlines came as content, so the next turn stopped matching
  the generated tokens right before the calls. `0009` (by [Coinupbtc](https://github.com/Coinupbtc), from
  [Mia's AI Lab's recipe](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks), its `0027`) keeps the
  text as the content, as DeepSeek's reference parser reads it.
- **A control token quoted in a file or a web page reached the model as the token.** An agent that read `</think>`,
  `<｜User｜>` or a DSML block gave the model the end of its reasoning, a new turn or a call. `0010`
  ([#36](https://github.com/bertholomus/TensorFold/pull/36) by [Blackwellboy](https://github.com/Blackwellboy), not
  merged yet) keeps what a chat request quotes as text; the template's own tokens are unchanged.
- **A short prompt that shared another conversation's prefix ran from 0**, and the other conversation lost that state,
  when it ended less than 128 tokens past one of the other's chunk ends (a subagent, typically). Fixed upstream in
  [#35](https://github.com/bertholomus/TensorFold/pull/35) by [kky42](https://github.com/kky42); in our check the
  prompt now resumes at 4,096 (0 before).
- **A `</think>` the model wrote while reasoning ended its reasoning there.** The rest of the reasoning came as the
  answer. `0011`-`0012` ([ashhart](https://github.com/ashhart)'s fix for TensorFold
  [#1180](https://github.com/ashhart/TensorFold/issues/1180), on TensorFold's main since 1.1.4) end it at the model's
  own close token; `0013` fits their test to this branch. `tests/think_quote.py`: the reasoning keeps the quoted tag
  and the answer is the answer, whole and streamed (both failed before).
- **Prompts ~3 % faster** with `TF_DS_ENGRAM_LEAD=1` ([#41](https://github.com/bertholomus/TensorFold/pull/41) by
  Blackwellboy, opt-in upstream), on in `scripts/local.sh.example`: 2,031–2,047 tok/s against 1,961–1,995 without it
  (a 50K-token prompt, 5 runs right after a start, both builds), the same replies and the same decode.

## Results

On our pair, at `--context 1048576 --parallel 4` with the settings in `scripts/local.sh.example`:

| | |
| --- | --- |
| Decode, one stream (`kit_bench.py`, set a) | code 87.8, prose 57.6, structured 105.6 tok/s |
| Prompt processing | ~2,050–2,150 tok/s from 8K to 160K tokens (`kit_bench.py`); a 1,040,000-token prompt in 653 s; `tests/prefill_speed.py` on a 50K-token prompt of source code: 2,031–2,047 tok/s with `TF_DS_ENGRAM_LEAD=1`, 1,961–1,995 without it (the same on the `45c25b1b` pin) |
| Exactness | 12 greedy replies byte-identical to the unpatched engine; resumed prompts equal fresh ones; tool calls 10/10; `tests/quoted_prose_tail.py` 3/3 (0/3 before `0009`, `0010` and #35); `tests/think_quote.py` 2/2 (0/2 before `0011`-`0013`) |
| Template | `/tokenize` ids identical for 1,500 random conversations; 5,000 identical renders in jinja2 |
| Engine tests | `zig build test`: 232/232; the server's (`zig/tests/server/zig_test.sh`): 88 + 34, and `tool_parse_test.zig` 18 |

The checkpoint's quality is measured in its [model card](https://huggingface.co/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw):
against DeepSeek's FP8 release on the same 100 × 2,048 tokens, KL 0.0864 and top-1 agreement 90.8%.

## Requirements

- **Two DGX Sparks** (or two GB10 systems with 128 GB unified memory), with nothing else large on their GPUs. Each
  needs ~110 GiB free when the server starts.
- **A direct ConnectX-7 link** with an IPv4 address on each end and a RoCE v2 GID. A cabled port shows up as two RoCE
  devices (e.g. `rocep1s0f1` / `roceP2p1s0f1`); name both in `HCA`. Find the GID index with `show_gids`.
- **Key-based ssh** from the first Spark (the head, which runs the API) to the second (the worker).
- **Docker** with the NVIDIA runtime or a CDI spec for the GPU, plus `rsync` and `python3`, on both Sparks.
- **Disk, on each Spark:** ~520 GB under `DATA_DIR` (checkpoint ~211 GB, Engram shards ~190 GB, rank cache ~106 GB,
  kit ~2 GB), and ~30 GB for the image.
- **Memory settings:** we serve with [`ops/99-spark-memory.conf`](ops/99-spark-memory.conf) on both Sparks and
  `TF_MEMORY_RESERVE_GIB=4` (see Configuration).

## Quick start

On the head:

```bash
git clone https://github.com/hashd1ve/deepseek-v4.1-flash-2x-dgx-spark.git
cd deepseek-v4.1-flash-2x-dgx-spark
cp scripts/local.sh.example scripts/local.sh   # WORKER, HEAD_IP, IFACE, HCA, GID_INDEX
./start.sh
```

The first run sets up both Sparks, which takes hours, mostly downloading and writing the rank caches. Later starts
only load (about a minute). `./start.sh` ends with one request through both ranks and prints the endpoint:

```bash
curl -s http://127.0.0.1:18891/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "DeepSeek-V4.1-Flash-1017-EXL3",
  "messages": [{"role": "user", "content": "Write a Python fibonacci function."}],
  "max_tokens": 4000
}'

./start.sh restart        # restart both ranks (extra arguments go to tensorfold-native serve)
./stop.sh                 # stop both ranks; their logs are saved under DATA_DIR/logs
nohup ops/keep_serving.sh > /dev/null 2>&1 &   # optional: start it again when a rank dies (see below)
```

`ops/keep_serving.sh` starts the server again when a rank goes down without `./stop.sh` (a crash, or
`ops/memory_watchdog.sh` stopping it), at most 3 times in 2 hours; its log is `DATA_DIR/keep_serving.log`. With it
running, stop the server with `./stop.sh`, never with `docker stop`.

The model thinks before it answers (`reasoning_content`), so give replies enough `max_tokens`. The server listens on
`127.0.0.1` by default; set `HOST=0.0.0.0` together with `API_KEY_FILE` to serve the network. Images: OpenAI-style
`image_url` parts with `data:` URLs.

## What `scripts/prepare.sh` does

Each step skips work already done, so it is safe to re-run.

1. **Preflight on both Sparks:** ssh, the NVIDIA runtime (or its CDI spec), the RoCE devices and GID type, the route
   to the head, disk.
2. **Image:** builds `dsv41-1017-tf:<pin hash>` from `docker/Dockerfile`: NVIDIA's PyTorch 26.07 container, Zig 0.17.0
   (sha256-checked), both engine branches at their pinned commits, `docker/patches/` applied to the Zig engine, built
   for sm_121. The pin hash covers the patches. The image is copied to the worker and its ID compared.
3. **Weights:** downloads the checkpoint and DeepSeek's shards 47–48 at pinned revisions, each file checked against the
   Hub, and copies them to the worker. It then puts `model/chat_template.jinja` in place of the published template.
4. **Kit:** Bertholomus AI's kit at its pinned revision, checked against its `MANIFEST`, with the 1017 image routing
   bias and the `_topk_sel` N=KK=256 variant (without which a prompt of exactly 512 or 513 tokens stops the server).
5. **Rank caches:** starts the Python engine once on both ranks to write each rank's weights (~106 GB) and the Engram
   token map. A complete cache from an interrupted run is kept.

## Configuration

`scripts/config.sh` holds the pins and defaults; override them in `scripts/local.sh`. Beyond the recipe's (`CONTEXT`,
`PARALLEL`, `MAX_TOKENS`, `HOST`, `PORT`, `API_KEY_FILE`, `DATA_DIR`, `FABRIC_PEER`, `WORKER_IFACE` / `WORKER_HCA`):

| Setting | Ours | |
| --- | --- | --- |
| `TF_FILL_CHUNKS` | 1 | while other streams decode, a long prompt runs this many chunks (2,048 tokens, ~1 s) at a time; unset: each prompt runs whole and the others wait |
| `TF_FILL_ROUNDS` | 4 | the other streams' rounds between two pieces: more is smoother for them and slower for the long prompt |
| `TF_FILL_FIRST` | 4 | the first piece's chunks, so a turn's new tokens within it (8K) run whole |
| `TF_MEMORY_RESERVE_GIB` | 4 | rank 0's CUDA reserve; the default (a tenth of the memory, ~12.2 GiB) leaves the weights no room with our kernel settings |
| `SERVE_EXTRA` | `--prompt-cache-gib 0.5` | the kept prompt states' budget |
| `INPUTS_CONTEXT`, `INPUTS_PARALLEL` | 65536, 1 | the window of step 5's one Python start (less memory; the rank caches are the same) |
| `ENGINE_ENV` | `-e TF_DS_ENGRAM_LEAD=1` | extra `-e NAME=value` options for both ranks' containers (the engine's opt-in settings, such as `TF_DS_ENGRAM_LEAD=1`) |

On the Sparks themselves: [`ops/memory_watchdog.sh`](ops/memory_watchdog.sh) stops the engine's container before a
GB10 runs out of memory and stalls, and [`ops/memlog.sh`](ops/memlog.sh) logs where the memory goes every 30 s.

## Credits

This builds on other people's work.

- **[easthambuilds](https://github.com/easthambuilds):** the [recipe](https://github.com/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-2x-DGX-Sparks-TensorFold)
  this repository starts from, and the [checkpoint](https://huggingface.co/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw)
  it serves (the recipe is Apache-2.0; the checkpoint's license is on its model card).
- **[DeepSeek](https://huggingface.co/deepseek-ai):** [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017),
  its weights, Engram tables and DSpark draft blocks (MIT).
- **[Bertholomus AI](https://github.com/bertholomus) (Albert Lee):** the `dsv41` TensorFold family: the
  [Zig engine](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-zig), the
  [Python engine](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-tp2), the
  [kernel kit](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-TensorFold-TP2-2xGB10) and
  [the TP2 recipe](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10) (Apache-2.0).
- **[ashhart](https://github.com/ashhart) and the TensorFold contributors:** [TensorFold](https://github.com/ashhart/TensorFold)
  (Apache-2.0), including the template and JSON fixes by Nathan McBride and Jay Leaton that #24 brought over.
- **[Coinupbtc](https://github.com/Coinupbtc)** and **[Blackwellboy](https://github.com/Blackwellboy)**:
  `0009` and `0010`; **[ashhart](https://github.com/ashhart)**: `0011`-`0012`. Each patch carries its author.
- **[Mia's AI Lab](https://github.com/MiaAI-Lab):** the 2.9 bpw layout the checkpoint follows and the chat template it
  carries, which `model/chat_template.jinja` changes; the layout of these scripts comes from their GLM-5.3-Flash recipe.
- **[turboderp](https://github.com/turboderp-org):** [ExLlamaV3](https://github.com/turboderp-org/exllamav3) and the
  EXL3 format (MIT).
- **NVIDIA:** the DGX Spark and the [PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
  the image is built on.

The changes in this repository (the engine patches but `0009`-`0012`, the template change, the recipe changes, `ops/`
and `tests/`) are by [hashd1ve](https://github.com/hashd1ve). Not affiliated with or endorsed by DeepSeek, NVIDIA, the TensorFold authors,
Bertholomus AI, easthambuilds or Mia's AI Lab.

## License

The scripts, patches and documentation in this repository are Apache-2.0 ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)).
The weights, the engine and the kit keep their own licenses (above).
