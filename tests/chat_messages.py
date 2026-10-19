# chat_messages.py chat|tokenize K M: one request with about K thousand prose tokens split into M messages (the chat
# template's cost grows with messages x prompt length when it builds the prompt in one growing string)
import random, sys
from common import MODEL, post

mode, k, m = sys.argv[1], int(sys.argv[2]) * 1000, int(sys.argv[3])
r = random.Random(2)
words = "the of and to in is was for on that with as by at from his her an which this be or are had not but have its were they one all their been".split()
msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": " ".join(r.choice(words) for _ in range(max(1, k // m)))} for i in range(m)]
if msgs[-1]["role"] != "user":
    msgs.append({"role": "user", "content": "Say ok."})
if mode == "tokenize":
    print(len(post("/tokenize", {"messages": msgs})["tokens"]), "tokens")
else:
    r = post("/v1/chat/completions", {"model": MODEL, "messages": msgs, "max_tokens": 2, "chat_template_kwargs": {"thinking": False}})
    print(r["usage"]["prompt_tokens"], "tokens")
