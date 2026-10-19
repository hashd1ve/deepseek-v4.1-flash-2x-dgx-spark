# template_ids.py N OUT: the server's /tokenize ids (sha256 and length) for N random conversations, one JSON line each.
# Record before and after a change to the template or the server, then compare the two files (cmp).
import hashlib, json, random, sys, urllib.error
from common import post
from conversations import conversation

with open(sys.argv[2], "w") as out:
    for i in range(int(sys.argv[1])):
        msgs, kw = conversation(random.Random(i))
        for m in msgs:  # what the server accepts: no latest_reminder role, text tool results, its own effort names
            if m["role"] == "latest_reminder":
                m["role"] = "user"
            if m["role"] == "tool" and not isinstance(m["content"], str):
                m["content"] = json.dumps(m["content"])
        if "reasoning_effort" in kw:
            kw["reasoning_effort"] = {"max": "xhigh", 30: "low", 100: "high"}.get(kw["reasoning_effort"], kw["reasoning_effort"])
        body = {"messages": msgs, "add_generation_prompt": kw["add_generation_prompt"]}
        for k in ("tools", "enable_thinking", "reasoning_effort"):
            if kw.get(k) is not None:
                body[k] = kw[k]
        try:
            ids = post("/tokenize", body, 60)["tokens"]
            rec = {"seed": i, "n": len(ids), "sha": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]}
        except urllib.error.HTTPError as e:
            rec = {"seed": i, "http": e.code}
        out.write(json.dumps(rec) + "\n")
