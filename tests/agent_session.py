# agent_session.py TOKENS_K IMAGES TURNS [SEED]: an agent session as OpenCode sends it, grown to about TOKENS_K thousand tokens
# with IMAGES screenshots, then TURNS more turns (a tool call and its result each, a new screenshot every third one).
# Each turn prints the tokens resumed, the seconds to the reply and, run on the head, rank 0's RSS before and at its
# peak and the host's lowest MemFree and page cache while it ran (memory_watchdog.sh stops the engine under 3 GiB free
# with under 1 GiB of cache). Sessions with another SEED share the system prompt and tools, as a subagent's does.
import json, os, random, subprocess, sys, threading, time
from common import MODEL, post
from screenshot_turns import screenshot

tokens_k, images, turns = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 5
r = random.Random(5)  # the harness: the same for every session
WORDS = ("the of and to in is for on that with as by at from this be or are not file line return value self import def "
         "class if else while try except raise None True config server request response token model cache state").split()


def text(n):
    return " ".join(r.choice(WORDS) for _ in range(n // 5))[:n]


def step(i, size):
    cid = f"call_{i:05d}"
    call = {"role": "assistant", "content": "", "reasoning_content": text(r.choice([400, 1000, 2500])),
            "tool_calls": [{"id": cid, "type": "function", "function": {"name": "read", "arguments": json.dumps({"filePath": f"/work/src/m{i}.py"})}}]}
    return [call, {"role": "tool", "tool_call_id": cid, "content": text(size)}]


tool = lambda name: {"type": "function", "function": {"name": name, "description": text(1200), "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}}, "required": ["filePath"]}}}
TOOLS = [tool(n) for n in ["bash", "read", "write", "edit", "glob", "grep", "list", "task", "todowrite", "webfetch"]]
msgs = [{"role": "system", "content": text(38000)}, {"role": "user", "content": "Fix the failing tests."}]
if len(sys.argv) > 4:
    r = random.Random(seed)  # another conversation (without SEED, the one the numbers in docs/FINDINGS.md come from)
budget, i, shots = tokens_k * 1000 * 4, 0, 0  # ~4 characters a token
every = max(1, budget // 5600 // max(images, 1))  # a step is ~5,600 characters: the screenshots spread over the session
size = 38000 + sum(len(json.dumps(t)) for t in TOOLS)
while size < budget:  # the text only: a screenshot's tokens are its span's, not its base64's
    msgs += step(i, r.choice([1500, 2500, 4000, 8000]))
    size += sum(len(json.dumps(m)) for m in msgs[-2:])
    if shots < images and i % every == 0:
        msgs.append({"role": "user", "content": [{"type": "text", "text": "Here is the screen now."}, {"type": "image_url", "image_url": {"url": screenshot(1000 * (seed - 4) + shots)}}]})
        shots += 1
    i += 1
msgs.append({"role": "user", "content": "Go on."})

pid = subprocess.run(["pgrep", "-f", "^/opt/tf/native/bin/tensorfold-native"], capture_output=True, text=True).stdout.split()
pid = pid[0] if pid else None


def sample():
    rss = int(next(l.split()[1] for l in open(f"/proc/{pid}/status") if l.startswith("VmRSS"))) // 1024 if pid else 0
    m = {l.split(":")[0]: int(l.split()[1]) // 1024 for l in open("/proc/meminfo")}
    return rss, m["MemFree"], m["Active(file)"] + m["Inactive(file)"]


for t in range(turns + 1):
    base = sample()
    lows, done = [base], [False]
    def watch():
        while not done[0]:
            lows.append(sample())
            time.sleep(0.05)
    th = threading.Thread(target=watch)
    th.start()
    t0 = time.time()
    u = post("/v1/chat/completions", {"model": MODEL, "messages": msgs, "tools": TOOLS, "max_tokens": 200}, timeout=1800)["usage"]
    dt = time.time() - t0
    done[0] = True
    th.join()
    label = "build" if t == 0 else f"turn {t}"
    print(f"{label}: {shots} images, prompt {u['prompt_tokens']}, resumed {(u.get('prompt_tokens_details') or {}).get('cached_tokens')}, "
          f"{dt:.1f} s; rank 0 RSS {base[0]} -> peak {max(x[0] for x in lows)} MiB; lowest MemFree {min(x[1] for x in lows)} MiB, "
          f"page cache {min(x[2] for x in lows)} MiB", flush=True)
    msgs.pop()
    msgs += step(i, r.choice([1500, 2500, 4000]))
    i += 1
    if t % 3 == 2:
        msgs.append({"role": "user", "content": [{"type": "text", "text": "And now?"}, {"type": "image_url", "image_url": {"url": screenshot(1000 * (seed - 4) + shots)}}]})
        shots += 1
    msgs.append({"role": "user", "content": "Go on."})
