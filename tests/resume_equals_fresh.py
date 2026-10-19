# resume_equals_fresh.py DOC_A DOC_B: greedy replies about file A fresh, again (resumed from its kept state), then about
# file B (same system prompt), then about A again: the three replies about A must be byte-identical
import sys
from common import chat

doc_a, doc_b = open(sys.argv[1]).read(), open(sys.argv[2]).read()
system = ("You are a senior engineer reviewing source code. Answer in English, precisely and briefly. " * 40).strip()
tools = [{"type": "function", "function": {"name": "read_file", "description": "Read a file.",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
ask = lambda doc: chat([{"role": "system", "content": system}, {"role": "user", "content": doc + "\n\nIn three sentences, what does this code do?"}], 96, tools=tools)
replies = []
for tag, doc in (("A fresh", doc_a), ("A resumed", doc_a), ("B", doc_b), ("A after B", doc_a)):
    n, c, text = ask(doc)
    print(f"{tag:10} prompt={n} cached={c} | {text[:100]!r}")
    if doc is doc_a:
        replies.append(text)
print("RESULT:", "identical" if len(set(replies)) == 1 else "DIFFERENT")
