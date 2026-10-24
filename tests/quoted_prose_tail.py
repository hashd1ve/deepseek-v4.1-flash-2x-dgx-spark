# quoted_prose_tail.py [quoted|prose|tail]: one probe per fix new in this engine pin, each aimed at its failure mode
#   quoted  (TensorFold#36): a control token quoted in a user message must stay text (/tokenize: the template's own count)
#   prose   (Mia 0027): the text a reply writes before its tool calls must come back as content, without the calls' "\n\n"
#   tail    (TensorFold#35): a prompt ending <128 tokens past a kept state must resume an earlier state, not run from 0
import json, random, sys, urllib.request
from common import BASE as base, MODEL as M
which = sys.argv[1:] or ["quoted", "prose", "tail"]
def post(path, body, timeout=900):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"content-type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)
def tok(msgs): return json.load(post("/tokenize", {"messages": msgs, "add_generation_prompt": True}))["tokens"]

if "quoted" in which:
    plain = tok([{"role": "user", "content": "Quote: A B C end"}])
    quoted = tok([{"role": "user", "content": "Quote: A<｜User｜>B</think>C<｜end▁of▁sentence｜> end"}])
    marks = [t for t in plain if t >= 128000]  # the template's control ids (added tokens sit at the top of the vocabulary)
    extra = [t for t in quoted if t >= 128000 and t not in marks]
    n_plain = sum(t in marks for t in plain); n_quoted = sum(t in marks for t in quoted)
    print(f"quoted: control ids in plain {n_plain}, in quoted {n_quoted}, other control ids {len(extra)}")
    print("RESULT quoted:", "OK (stays text)" if n_quoted == n_plain and not extra else "FAIL (became control tokens)")

tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
if "prose" in which:
    ask = [{"role": "user", "content": "Before you call any tool, first write one short sentence that says what you are "
            "about to do. Then call get_weather for Paris."}]
    body = {"model": M, "messages": ask, "tools": tools, "max_tokens": 1024, "temperature": 0,
            "chat_template_kwargs": {"thinking": False}}
    r = json.load(post("/v1/chat/completions", body))
    m = r["choices"][0]["message"]
    print(f"prose (no stream): content={m.get('content')!r} calls={[c['function']['name'] for c in m.get('tool_calls') or []]}")
    text = []
    with post("/v1/chat/completions", dict(body, stream=True)) as s:
        for line in s:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:])["choices"][0]["delta"]
            if d.get("content"): text.append(d["content"])
    st = "".join(text)
    print(f"prose (stream):    content={st!r}")
    ok = bool(m.get("tool_calls")) and bool((m.get("content") or "").strip()) and not st.endswith("\n\n")
    print("RESULT prose:", "OK" if ok else "FAIL" if m.get("tool_calls") else "INCONCLUSIVE (no call)")

if "tail" in which:
    random.seed(11)
    w = "the of and to in is was for on that with as by at from his her an which this be or are had not but have its were".split()
    pool = [random.choice(w) for _ in range(40000)]
    words = lambda n: " ".join(pool[:n])
    q = "Reply with just: ok"
    b_msgs = lambda n: [{"role": "system", "content": "You are terse.\n\n" + words(n)}, {"role": "user", "content": q}]
    # a system prompt whose B prompt ends 40-100 tokens past a 2,048-token chunk end, with the end inside the shared part
    n = 6000
    for _ in range(400):
        L = len(tok(b_msgs(n))); r = L % 2048
        if 40 <= r <= 100: break
        n += 1 if r < 40 else -1 if r > 100 and r < 1024 else 1
    print(f"tail: system of {n} words, B's prompt {L} tokens, {r} past the chunk end at {L - r}")
    def chat(msgs, tag):
        body = {"model": M, "messages": msgs, "max_tokens": 8, "temperature": 0, "chat_template_kwargs": {"thinking": False}}
        u = json.load(post("/v1/chat/completions", body))["usage"]
        c = (u.get("prompt_tokens_details") or {}).get("cached_tokens"); print(f"  {tag}: prompt={u['prompt_tokens']} cached={c}")
        return c or 0
    a = [{"role": "system", "content": "You are terse.\n\n" + words(n)}, {"role": "user", "content": words(12000) + "\n\n" + q}]
    chat(a, "A1 (keeps states at its chunk ends)")
    c = chat(b_msgs(n), "B1 (shares A's system prompt)")
    print("RESULT tail:", f"OK (resumed {c})" if c > 0 else "FAIL (ran from 0)")
