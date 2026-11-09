# cut_in_tail.py HOST PORT DOC_LONG DOC_CONV: Mia 0045, a cut-in counts the tokens past the kept state. A conversation C
# (past the 8192 bound, cached) gets (S) a short new turn and (W) a new turn of more than the bound while a long cold
# prompt L runs. With 0045, S cuts in and W waits for L; without it, both wait. Then S's reply is compared with S alone.
import http.client, json, random, sys, threading, time
host, port = sys.argv[1], int(sys.argv[2])
docl, docc = open(sys.argv[3]).read(), open(sys.argv[4]).read()
salt = str(random.random())
base = {"model": "DeepSeek-V4.1-Flash-1017-EXL3", "temperature": 0, "max_tokens": 40, "chat_template_kwargs": {"thinking": False}}
def run(body, out, t0):
    c = http.client.HTTPConnection(host, port, timeout=1800)
    c.request("POST", "/v1/chat/completions", json.dumps(dict(body, stream=True, stream_options={"include_usage": True})), {"content-type": "application/json"})
    r = c.getresponse(); text, first, usage = [], None, None
    for raw in r:
        line = raw.strip()
        if not line.startswith(b"data:") or line == b"data: [DONE]": continue
        d = json.loads(line[5:]); usage = d.get("usage") or usage; ch = d.get("choices") or []
        piece = (ch[0].get("delta") or {}).get("content") if ch else None
        if piece:
            if first is None: first = time.time() - t0
            text.append(piece)
    out.update(text="".join(text), first=first, usage=usage, total=time.time() - t0)
def alone(body):
    o = {}; run(body, o, time.time()); return o
cached = lambda o: (o["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
first = [{"role": "user", "content": salt + "\n" + (docc + "\n") * 3 + "\nSay in one sentence what this text is about."}]
C = alone(dict(base, messages=first))
print(f"C: prompt {C['usage']['prompt_tokens']} tokens, first token {C['first']:.1f} s")
hist = first + [{"role": "assistant", "content": C["text"]}]
S = dict(base, messages=hist + [{"role": "user", "content": "Now name three rivers of Spain, comma separated."}])
W = dict(base, messages=hist + [{"role": "user", "content": salt + "x\n" + (docl + "\n") * 2 + "\nAnd this text, in one sentence?"}])
L = dict(base, messages=[{"role": "user", "content": salt + "L\n" + (docl + "\n") * 20 + "\nSay in one sentence what this text is about."}])
l, s, w = {}, {}, {}
t0 = time.time(); a = threading.Thread(target=run, args=(L, l, t0)); a.start(); time.sleep(4)
t1 = time.time(); b = threading.Thread(target=run, args=(S, s, t1)); b.start(); time.sleep(2)
t2 = time.time(); d = threading.Thread(target=run, args=(W, w, t2)); d.start()
a.join(); b.join(); d.join()
print(f"L: prompt {l['usage']['prompt_tokens']} tokens, first token {l['first']:.1f} s")
print(f"S: prompt {s['usage']['prompt_tokens']} tokens, cached {cached(s)}, first token {s['first']:.2f} s after sent; before L's first token: {s['first'] + 4 < l['first']}")
print(f"W: prompt {w['usage']['prompt_tokens']} tokens, cached {cached(w)}, first token {w['first']:.2f} s after sent; before L's first token: {w['first'] + 6 < l['first']}")
s2 = alone(S)
print(f"alone: S again {s2['first']:.2f} s, cached {cached(s2)}")
print("RESULT S", "identical" if s["text"] == s2["text"] else f"DIFFERENT\n  {s['text']!r}\n  {s2['text']!r}")
