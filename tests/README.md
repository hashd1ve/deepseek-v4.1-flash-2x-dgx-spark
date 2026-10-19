# Checks

Small clients for a running server. Set `BASE_URL` (default `http://127.0.0.1:18891`) and, if yours differs, `MODEL`.
All of them send greedy requests, with thinking off except `stream_tool_call.py`, which sends it on as OpenCode does.

| Check | What it shows | Our result |
| --- | --- | --- |
| `template_jinja2.py OLD NEW N` | two chat templates render N random conversations identically (jinja2, Hugging Face's settings) | 5,000 identical |
| `template_ids.py N OUT` | the server's `/tokenize` ids for N random conversations; record before and after a change, then `cmp` | 1,500 identical |
| `bash memory_peak.sh python3 chat_messages.py chat 40 2000` | rank 0's memory peak for 40K tokens in 2,000 messages (run on the head) | +1,555 MiB with the published template, +244 MiB with ours |
| `shared_prefix.py` | a conversation keeps its state when a new one with the same system prompt starts | resumed at 67,584 of 69,286 (0 before #26) |
| `resume_equals_fresh.py DOC_A DOC_B` | a prompt resumed from its kept state gives the same reply as a fresh one, also after another conversation | identical |
| `images.py DOC record\|check FILE` | a turn with the same image resumes past it with the same reply; another image does not resume past it | resumed at 6,144 of 6,912, identical |
| `interleave.py DOC` | another stream keeps decoding while a new ~57K-token prompt runs, and the prompt's reply equals its repeat | 100 text chunks meanwhile (0 with whole passes), identical |
| `prefill_speed.py DOC REPS` | prompt tokens a second for a new prompt | ~2,100 tok/s alone |
| `stream_tool_call.py` | a long tool call's arguments arrive in many deltas while they are written, and join to the same call as the unstreamed reply | 159 and 163 deltas over 7 and 12 s, the same call |

`DOC` is any text file of a few thousand tokens (we used engine source files). `conversations.py` generates the
random conversations, covering every path of the DeepSeek-V4.1 template: tools, tool calls with object and string
arguments, several tool results out of order, reasoning, consecutive user turns, mid-conversation system messages,
`latest_reminder`, `response_format`, every effort and both thinking modes.
