# think_quote.py: a reply that quotes </think> while it reasons must keep the quote in reasoning_content (ashhart #1180)
import json, urllib.request
from common import BASE as base, MODEL
ask = ("While you think, write the literal tag </think> exactly once as plain text (the user asked you to show it), "
       "then keep thinking for one more sentence. Your final answer must be only the word: done")
body = {"model": MODEL, "messages": [{"role": "user", "content": ask}], "max_tokens": 2048,
        "temperature": 0, "chat_template_kwargs": {"thinking": True}}
for stream in (False, True):
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(dict(body, stream=stream)).encode(), {"content-type": "application/json"})
    if not stream:
        m = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]
        r, c = m.get("reasoning_content") or "", m.get("content") or ""
    else:
        r, c = "", ""
        with urllib.request.urlopen(req, timeout=600) as s:
            for line in s:
                line = line.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]": continue
                d = json.loads(line[5:])["choices"][0]["delta"]
                r += d.get("reasoning_content") or ""; c += d.get("content") or ""
    quoted = "</think>" in r
    tag = "stream" if stream else "whole "
    print(f"{tag}: reasoning quotes the tag: {quoted}; content={c.strip()[:80]!r}")
    print(f"RESULT think ({tag.strip()}):", "OK" if quoted and c.strip().lower().rstrip('.') == "done" else
          "FAIL (split at the quote)" if "</think>" not in r and c.strip().lower() != "done" else "INCONCLUSIVE (no quote written)")
