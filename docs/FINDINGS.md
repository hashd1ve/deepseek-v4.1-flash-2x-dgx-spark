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

## Also found

- **The prompt cache restored states exactly.** We checked this when the model answered in Chinese mid-session (a
  quantized model's slip, not a restore bug): a resumed ~7K-token prompt gives the same greedy reply as a fresh one,
  also after another conversation (`tests/resume_equals_fresh.py`).
- **GB10 memory.** With ~105 GiB held by a rank, the host runs within a few GiB. The kernel settings in
  [`ops/99-spark-memory.conf`](../ops/99-spark-memory.conf) and [`ops/memory_watchdog.sh`](../ops/memory_watchdog.sh)
  stop the engine cleanly instead of letting the Spark stall.
