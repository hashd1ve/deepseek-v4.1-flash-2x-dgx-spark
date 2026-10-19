# interleave.py DOC: A streams a long reply; B, a new ~60K-token prompt, arrives meanwhile. Prints A's text chunks while
# B's prompt runs (0 when each prompt runs whole), then sends B again (resumed): both B replies must be identical.
import http.client, json, random, sys, threading, time, urllib.parse
from common import BASE, MODEL

doc = open(sys.argv[1]).read()
url = urllib.parse.urlparse(BASE)


def stream(body, times, out):
    c = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=1800)
    c.request("POST", "/v1/chat/completions", json.dumps(dict(body, stream=True)), {"content-type": "application/json"})
    text = []
    for line in c.getresponse():
        line = line.strip()
        if not line.startswith(b"data:") or line == b"data: [DONE]":
            continue
        ch = json.loads(line[5:]).get("choices") or []
        piece = (ch[0].get("delta") or {}).get("content") if ch else None
        if piece:
            times.append(time.time())
            text.append(piece)
    out.append("".join(text))


base = {"model": MODEL, "temperature": 0, "chat_template_kwargs": {"thinking": False}}
ta, oa = [], []
a = threading.Thread(target=stream, args=(dict(base, max_tokens=1500, messages=[{"role": "user", "content": "Write a long, detailed story about a lighthouse keeper. Do not stop early."}]), ta, oa))
a.start()
while not ta:
    time.sleep(0.05)
time.sleep(2)
b_msgs = [{"role": "user", "content": str(random.random()) + "\n" + (doc + "\n") * 9 + "\nIn two sentences, what does this code do?"}]
tb, ob = [], []
t0 = time.time()
stream(dict(base, max_tokens=60, messages=b_msgs), tb, ob)
first = tb[0] if tb else time.time()
during = [t for t in ta if t0 <= t <= first]
print(f"B: first token after {first - t0:.1f} s; A received {len(during)} text chunks while B's prompt ran")
a.join()
tb2, ob2 = [], []
t1 = time.time()
stream(dict(base, max_tokens=60, messages=b_msgs), tb2, ob2)
print(f"B again: first token after {(tb2[0] if tb2 else time.time()) - t1:.1f} s")
print("RESULT:", "B identical when repeated" if ob[0] == ob2[0] else "B DIFFERENT", "|", "A kept decoding" if during else "A STOPPED")
