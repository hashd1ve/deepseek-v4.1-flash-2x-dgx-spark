# Findings

What broke while serving DeepSeek-V4.1-Flash-1017 to agent sessions on two DGX Sparks, why, and what fixed it. All
numbers are from one pair of GB10s at `--context 1048576 --parallel 4`. The checks are in [`tests/`](../tests).

## 1. Rank 0's memory grew with messages × prompt length

**Symptom.** In a 288K-token OpenCode session, rank 0's RSS went from 2.9 to 6.0 GiB during every request and back
after it: about 10.9 KB per prompt token, although almost all of the prompt came from the prompt cache. At 1M × 4 a
GB10 has ~7 GiB left after loading, so the head reached its memory floor, and our watchdog stopped the server twice
in 20 minutes.

**Cause.** The chat template builds the prompt with `{% set ns.out = ns.out + ... %}`, 22 times per pass over the
messages. The engine's template evaluator made a new copy of the whole string on every `+`, and kept every copy until
the request ended. The server also renders the template twice per request (the prompt and its history).

| rank 0's peak, 40K tokens of prose | 1 message | 200 messages | 2,000 messages |
| --- | ---: | ---: | ---: |
| `/v1/chat/completions` | +65 MiB | +201 MiB | +1,555 MiB |
| `/tokenize` (the template only) | | +0 MiB | +648 MiB |

**Fix.** [`model/chat_template.jinja`](../model/chat_template.jinja) writes out each piece as it is made
(`{{- "" + X -}}`), which works on any engine. 2,000 messages now peak at +244 MiB (+21 MiB in `/tokenize`). The
rendered prompt is the same: 5,000 random conversations in jinja2, and the engine's own `/tokenize` ids for 1,500.
TensorFold had fixed the evaluator on `main` (append in place); [#24](https://github.com/bertholomus/TensorFold/pull/24)
brought that fix to `deepseek-v41-zig`. MiaAI-Lab's recipe adopted the template
([#1](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/1)); it is reported to the checkpoint's
recipe in [easthambuilds#2](https://github.com/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-2x-DGX-Sparks-TensorFold/issues/2).

## 2. A second conversation made the first one start over

**Symptom.** Whenever an agent started a subagent or a new session while the main session was idle, the main
session's next turn reprocessed its whole prompt: 3–4 min at ~400K tokens, with `resumed at 0, failed 1` in the log.

**Cause.** All streams share one pool of positions. A kept prompt state is a mark inside its conversation's extent,
and the shared prefix (the system prompt and tools, ~18K tokens) is such a mark inside the first conversation's. A new
conversation resumed that mark in place. `keptOver` skipped every state with the same base, meant for the same
conversation's earlier marks, so the first conversation's longer marks did not count, and the new one wrote over them.

**Fix.** [#26](https://github.com/bertholomus/TensorFold/pull/26), merged. A sibling longer than the state being
resumed now counts: the second conversation copies the shared rows to a free extent. Placement also first leaves room
for each kept conversation to grow, so a stream placed flush against one no longer makes its next turn move. Repro
(`tests/shared_prefix.py`): conversation A at 69,273 tokens, a new conversation B, then A's next turn. Before, A
resumed at 0; now at 67,584.

## 3. Prompts with images were never cached

**Symptom.** A session with six screenshots reprocessed its whole 73K-token history on every turn (~40 s), and every
other session waited.

**Cause.** The prompt cache matched tokens, and an image's span is the same placeholder token for every image, so the
engine kept nothing for a prompt with images and looked nothing up.

**Fix.** [#28](https://github.com/bertholomus/TensorFold/pull/28) ([`docker/patches/0002`](../docker/patches)): the
cache key spells each image span by the image's digest, one id per position past every vocabulary. A turn with the
same image resumes past it (6,144 of 6,912 tokens in `tests/images.py`), with the same reply as when the prompt ran
whole. The same text with another image resumes only before the image.

## 4. A long prompt stopped every other session

**Symptom.** A prompt pass ran whole inside its admission, so every decoding stream waited for it. That meant ~4 min
for a 400K-token session reloading after a restart.

**Fix.** [#29](https://github.com/bertholomus/TensorFold/pull/29) ([`docker/patches/0003`, `0004`](../docker/patches)).
With `TF_FILL_CHUNKS` set and another stream decoding, a prompt pass runs in pieces, with `TF_FILL_ROUNDS` of the
others' rounds between them.
- **Same frames on both ranks.** Rank 1 already handles any frame between two chunks of a fill, so both ranks run the
  same frames in the same order.
- **Fill state survives the rounds.** Rounds leave the fill's view, rings, spans and image rows alone.
- **Read-ahead.** A first version joined each piece's Engram read-ahead and then read every chunk again from the drive
  (38.4 s for the prompt below). The rows read ahead sit in buffers of their own, which rounds do not use, so the
  next piece now takes them.

| a new 56,840-token prompt while another stream decodes | its first token | the other stream's text chunks meanwhile |
| --- | ---: | ---: |
| whole passes | 28.9 s | 0 |
| in pieces (`1 / 4`, first piece 4 chunks) | 32.7 s | 100 |

The repeated prompt gives the same reply as the interleaved one.

## 5. Tool calls quoted in a file could run as real calls

**Cause.** The native server's call streamer read the whole answer, DSML blocks included. A Qwen-style
`<tool_call><function=NAME>` that the model writes into a file streamed as a real call to `NAME`, and the reply's own
DSML calls were then never sent. Separately, a `<tool_call>` in the prose with a `</tool_call>` inside an argument
swallowed the DSML block, and a call to a tool the request did not offer came back as text.

**Fix.** [#25](https://github.com/bertholomus/TensorFold/pull/25) ([`docker/patches/0001`](../docker/patches)), the
native side of the engine's Python fix: DSML blocks are read first, from their own tags; the streamer reads only the
text before the first block; an unoffered tool comes back as a call for the client to answer. Two tests fail before
the change and pass after it.

## 6. A long tool call looked like a hang

**Symptom.** In an OpenCode session at 142K tokens, the reply stopped on one token for ~4 min, then went on. The
engine was decoding the whole time (~95 tok/s, 32,000 tokens in 7 min); none of it reached the client.

**Cause.** While a reply streams, the server hides a tool-call block from the text and sends the calls when the reply
ends. A DSML calls block is one call's whole arguments, so an agent writing a large file sends nothing for as long as
the file takes to write.

**Fix.** [`docker/patches/0005`](../docker/patches), ours: the streamer reads DeepSeek-V4.1's calls block as the final
parser reads it and sends each invoke's opening delta (index, id, name) once its name is written, a string value's
text as it comes, and a JSON value (`string="false"`) at its closer, as the engine already did for Qwen's XML calls.
For a well-formed reply, what a client joins is the parsed call's arguments, byte for byte: a test feeds one with
escapes, non-ASCII text, a `</` inside a value, an unoffered tool and a value that is not JSON, one character at a
time. A value its invoke closes before its own closer ends that call, and the next one still streams. On the engine's
scripted server, a 68-chunk `write` call now arrives as 32 argument deltas from 0.76 s to 3.3 s instead of one at
3.3 s, with the same arguments and `finish_reason: tool_calls` (and as `input_json_delta` events on `/v1/messages`).
A call the reply's length cuts off now reaches the client open, as a streamed XML call already did; before, it was
dropped. The engine's server tests (70 + 33 + 17) and its parity goldens pass. On the pair, an ~80-line
`write_file` call ([`tests/stream_tool_call.py`](../tests/stream_tool_call.py), greedy, thinking on) arrived in two
runs as 159 and 163 argument deltas spread over the 7 and 12 s it took to write, each the same call as the
unstreamed reply; the other checks in [`tests/`](../tests) gave the same results as before the patch.

## 7. A session with screenshots got the server stopped

**Symptom.** An OpenCode session at 218K tokens stopped answering: our memory watchdog had stopped rank 0 (2.8 GiB
free, 0.78 GiB of page cache left, no memory pressure yet), and rank 1 went down with it. Nothing started it again,
so the client waited on a dead server.

**Cause.** Rank 0 used 3.6 GiB between requests and up to +1.25 GiB during each one, held for as long as its reply
ran. That afternoon, a session of the same length without images peaked at +0.45 GiB. Three things added up:
- the tokenizer ran every step of every word in one arena, ~800 bytes a token, and the server tokenizes the prompt
  twice (with and without the generation prompt);
- OpenCode sends every screenshot of the conversation on every turn (30 in that session), and the server decoded each
  one again: ~30 MiB allocated and ~35 ms for a 1280x633 PNG;
- each prompt pass ran the vision tower on every image and sent their rows to rank 1, also for the images its kept
  state already held.

**Fix.** [`docker/patches/0006`-`0008`](../docker/patches), ours:
- the tokenizer resets its scratch for each added-token segment and each word: 229 MiB → 5.6 MiB for a 296K-token
  prompt, the same ids over 24.5M tokens of random and real text;
- the server renders the prompt and prepares its images in scratch freed before the reply, so the request keeps its
  ids, cache key and payloads;
- a PNG is decoded the first time it comes, which still refuses a bad one, and later sized from its header. Its
  payload carries the file, decoded only when a prompt pass reads its span, and the pass runs the tower only for
  spans past the kept state it resumes from.

Each change has a test that fails without it. On the same synthetic session (`tests/agent_session.py 218 30 8`:
~195K tokens, 30-32 screenshots, 8 turns of a tool call and its result each):

| per turn, after the session is built | before | after |
| --- | ---: | ---: |
| rank 0's peak over its idle RSS | +1.3 GiB | +43 MiB |
| lowest MemFree on the head | 2.6–3.5 GiB | 4.4–4.6 GiB |
| seconds to the reply (200 tokens) | 13.4–19.4 | 3.0–8.0 |

The greedy replies, the template's ids, tool calls, image replies, resumed prompts and decode speed are unchanged
(the checks in [`tests/`](../tests)). A `/tokenize` of 296K tokens peaks at +44 MiB instead of +235.

**Down anyway.** [`ops/keep_serving.sh`](../ops/keep_serving.sh) runs `./start.sh restart` when a rank goes down without
`./stop.sh`, at most 3 times in 2 hours. Stopping rank 0 by hand, as the watchdog does, gave a 67-second outage.

## Also found

- **The prompt cache restored states exactly.** We checked this when the model answered in Chinese mid-session (a
  quantized model's slip, not a restore bug): a resumed ~7K-token prompt gives the same greedy reply as a fresh one,
  also after another conversation (`tests/resume_equals_fresh.py`).
- **GB10 memory.** With ~105 GiB held by a rank, the host runs within a few GiB. The kernel settings in
  [`ops/99-spark-memory.conf`](../ops/99-spark-memory.conf) and [`ops/memory_watchdog.sh`](../ops/memory_watchdog.sh)
  stop the engine cleanly instead of letting the Spark stall.
