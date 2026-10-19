# shared_prefix.py: conversation A (a long system prompt and a tool, a 60K-word user turn), then a new conversation B
# with the same system prompt, then A's next turn. A's next turn must resume from A's own kept state, not from 0.
import random
from common import chat

r = random.Random(7)
words = "the of and to in is was for on that with as by at from his her an which this be or are had not but have its were".split()
text = lambda n: " ".join(r.choice(words) for _ in range(n))
system = "You are a careful assistant.\n\n" + text(9000)
tools = [{"type": "function", "function": {"name": "read_file", "description": "Read a file.",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
a = [{"role": "system", "content": system}, {"role": "user", "content": text(60000) + "\n\nReply with just: ok"}]
n, c, reply = chat(a, 32768, tools=tools)
print(f"A, turn 1: {n} tokens, cached {c}")
n, c, _ = chat([{"role": "system", "content": system}, {"role": "user", "content": text(800) + "\n\nReply with just: ok"}], 32768, tools=tools)
print(f"B, a new conversation: {n} tokens, cached {c} (the shared system prompt)")
a += [{"role": "assistant", "content": reply}, {"role": "user", "content": "Once more, reply with just: ok"}]
n, c, _ = chat(a, 32768, tools=tools)
print(f"A, turn 2: {n} tokens, cached {c}")
print("RESULT:", "A resumed from its own state" if c and c > n // 2 else "A RAN FROM 0")
