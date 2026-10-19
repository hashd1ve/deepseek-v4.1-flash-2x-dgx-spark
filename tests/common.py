# common.py: the server and model every check talks to (BASE_URL and MODEL in the environment)
import json, os, urllib.request

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:18891").rstrip("/")
MODEL = os.environ.get("MODEL", "DeepSeek-V4.1-Flash-1017-EXL3")


def post(path, body, timeout=900):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def chat(messages, max_tokens=48, **extra):
    """A greedy chat reply with thinking off: (prompt tokens, cached tokens, text)."""
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"thinking": False}, **extra}
    r = post("/v1/chat/completions", body)
    u = r["usage"]
    return u["prompt_tokens"], (u.get("prompt_tokens_details") or {}).get("cached_tokens"), r["choices"][0]["message"].get("content") or ""
