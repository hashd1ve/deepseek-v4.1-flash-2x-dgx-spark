# cut_in_ping.py HOST PORT DOC REPS: a long cold prompt L (streamed) and, 3 s later, a short S. Reports S's first token,
# the ": ping" comments L received while its prompt ran, and whether both replies equal the same requests run alone.
import http.client, json, random, sys, threading, time
host, port, doc, reps = sys.argv[1], int(sys.argv[2]), open(sys.argv[3]).read(), int(sys.argv[4])
salt = str(random.random())
base = {"model": "DeepSeek-V4.1-Flash-1017-EXL3", "temperature": 0, "max_tokens": 60, "chat_template_kwargs": {"thinking": False}}
L = dict(base, messages=[{"role": "user", "content": salt + "\n" + (doc + "\n") * reps + "\nSay in one sentence what this text is about."}])
S = dict(base, messages=[{"role": "user", "content": salt + " Name three rivers of Spain, comma separated."}])
def run(body, out, t0):
    c = http.client.HTTPConnection(host, port, timeout=1800)
    c.request("POST", "/v1/chat/completions", json.dumps(dict(body, stream=True, stream_options={"include_usage": True})), {"content-type": "application/json"})
    r = c.getresponse(); text, first, pings, usage = [], None, [], None
    for raw in r:
        line = raw.strip()
        if line.startswith(b":"): pings.append(round(time.time() - t0, 1)); continue
        if not line.startswith(b"data:") or line == b"data: [DONE]": continue
        d = json.loads(line[5:]); usage = d.get("usage") or usage; ch = d.get("choices") or []
        piece = (ch[0].get("delta") or {}).get("content") if ch else None
        if piece:
            if first is None: first = time.time() - t0
            text.append(piece)
    out.update(text="".join(text), first=first, pings=pings, usage=usage, total=time.time() - t0)
def alone(body):
    o = {}; run(body, o, time.time()); return o
l, s = {}, {}
t0 = time.time(); a = threading.Thread(target=run, args=(L, l, t0)); a.start(); time.sleep(3)
t1 = time.time(); b = threading.Thread(target=run, args=(S, s, t1)); b.start(); a.join(); b.join()
print(f"L: prompt {l['usage']['prompt_tokens']} tokens, first token {l['first']:.1f} s, pings at {l['pings']}")
print(f"S: first token {s['first']:.2f} s after it was sent (prompt {s['usage']['prompt_tokens']}), L still running then: {s['first'] + 3 < l['first']}")
l2, s2 = alone(L), alone(S)
print(f"alone: L again {l2['first']:.1f} s (cached {(l2['usage'].get('prompt_tokens_details') or {}).get('cached_tokens')}), S again {s2['first']:.2f} s")
print("RESULT L", "identical" if l["text"] == l2["text"] else f"DIFFERENT\n  {l['text']!r}\n  {l2['text']!r}")
print("RESULT S", "identical" if s["text"] == s2["text"] else f"DIFFERENT\n  {s['text']!r}\n  {s2['text']!r}")
