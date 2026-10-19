# conversations.py: random conversations covering every path of the DeepSeek-V4.1 template (shared by the checks)
import random
WORDS = "alpha beta gamma 東京 ñandú \"quoted\" {{ not a tag }} {% neither %} <tag> \\back\\slash tab\there emoji🙂 x".split(" ")
def text(r, lo=0, hi=12): return " ".join(r.choice(WORDS) for _ in range(r.randint(lo, hi))) + ("\n" if r.random() < .2 else "")
def value(r, d=0):
    k = r.random()
    if k < .35: return text(r, 0, 6)
    if k < .5: return r.randint(-5, 10**6)
    if k < .6: return r.random() < .5
    if k < .65: return None
    if k < .8 and d < 2: return [value(r, d + 1) for _ in range(r.randint(0, 3))]
    if d < 2: return {f"k{i}": value(r, d + 1) for i in range(r.randint(0, 3))}
    return 1.5
def tool(r, i):
    f = {"name": f"tool_{i}", "description": text(r), "parameters": {"type": "object", "properties": {"a": {"type": "string"}}}}
    return {"type": "function", "function": f} if r.random() < .8 else f
def conversation(r):
    msgs, n = [], r.randint(1, 14)
    if r.random() < .5: msgs.append({"role": "system", "content": text(r)})
    pending = []
    for i in range(n):
        k = r.random()
        if pending and k < .7:
            ids = pending[:]; r.shuffle(ids)
            for cid in ids:
                c = text(r) if r.random() < .8 else {"out": value(r)}
                m = {"role": "tool", "content": c}
                if cid is not None: m["tool_call_id"] = cid
                msgs.append(m)
            pending = []; continue
        if k < .45:
            c = text(r) if r.random() < .85 else [{"type": "text", "text": text(r)}]
            msgs.append({"role": "user", "content": c})
        elif k < .9:
            m = {"role": "assistant", "content": text(r) if r.random() < .8 else None}
            if r.random() < .5: m["reasoning_content"] = text(r) if r.random() < .8 else ""
            if r.random() < .35:
                calls = []
                for j in range(r.randint(1, 3)):
                    cid = f"call_{i}_{j}" if r.random() < .9 else None
                    args = {f"p{q}": value(r) for q in range(r.randint(0, 3))} if r.random() < .85 else text(r)
                    c = {"type": "function", "function": {"name": f"tool_{j}", "arguments": args}}
                    if cid is not None: c["id"] = cid
                    calls.append(c); pending.append(cid)
                m["tool_calls"] = calls
            msgs.append(m)
        elif k < .95: msgs.append({"role": "system", "content": text(r)})
        else: msgs.append({"role": "latest_reminder", "content": text(r)})
    kw = {}
    if r.random() < .5: kw["tools"] = [tool(r, i) for i in range(r.randint(1, 4))]
    if r.random() < .8: kw["enable_thinking"] = r.random() < .6
    if r.random() < .8: kw["reasoning_effort"] = r.choice(["low", "high", "max", 30, 100, None])
    if r.random() < .15: kw["response_format"] = {"type": "object", "properties": {"x": {"type": "integer"}}}
    kw["add_generation_prompt"] = r.random() < .8
    return msgs, kw
