# stream_tool_call.py: a long tool call, greedy with thinking on (as OpenCode sends it), first whole and then streamed.
# The streamed arguments must arrive in many deltas while the model writes them, not in one at the reply's end, and
# join to the same call as the whole reply's.
import json, time, urllib.request
from common import BASE, MODEL, post

S = {"type": "string"}
TOOLS = [{"type": "function", "function": {"name": "write_file", "description": "Write a text file",
          "parameters": {"type": "object", "properties": {"path": S, "content": S}, "required": ["path", "content"]}}}]
BODY = {"model": MODEL, "tools": TOOLS, "max_tokens": 6000, "temperature": 0,
        "messages": [{"role": "user", "content": "Write intervals.py with the write_file tool: a class IntervalSet "
                      "(add, remove, contains, iteration, __repr__), docstrings on everything, and a small self-test "
                      "under __main__. About 80 lines. Write it straight away, no explanation."}]}

m = post("/v1/chat/completions", BODY)["choices"][0]
whole = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in m["message"].get("tool_calls") or []]
print(f"whole: finish={m['finish_reason']} calls={[n for n, _ in whole]}")

req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(dict(BODY, stream=True)).encode(),
                             {"content-type": "application/json"})
t0 = time.time()
names, args, times, finish = {}, {}, [], None
for line in urllib.request.urlopen(req, timeout=900):
    line = line.strip()
    if not line.startswith(b"data:") or line == b"data: [DONE]":
        continue
    for c in json.loads(line[5:]).get("choices") or []:
        for tc in (c.get("delta") or {}).get("tool_calls") or []:
            i, f = tc["index"], tc.get("function") or {}
            if f.get("name"):
                names[i] = f["name"]
            if f.get("arguments"):
                args[i] = args.get(i, "") + f["arguments"]
                times.append(time.time() - t0)
        finish = c.get("finish_reason") or finish
end = time.time() - t0
streamed = [(names[i], json.loads(args[i])) for i in sorted(args)]
print(f"streamed: finish={finish}, {len(times)} argument deltas"
      + (f" from {times[0]:.1f} s to {times[-1]:.1f} s, reply over at {end:.1f} s" if times else ""))
ok = streamed == whole and len(times) > 10 and times[-1] - times[0] > 0.5 * (end - times[0])
print("RESULT: " + ("streamed in pieces, the same call as the whole reply" if ok else "FAILED"))
