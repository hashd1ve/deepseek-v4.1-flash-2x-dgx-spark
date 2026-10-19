# prefill_speed.py DOC REPS: time to the first token of a new (uncached) prompt of DOC repeated REPS times
import random, sys, time
from common import chat

doc, reps = open(sys.argv[1]).read(), int(sys.argv[2])
t = time.time()
n, _, _ = chat([{"role": "user", "content": str(random.random()) + "\n" + (doc + "\n") * reps + "\nReply with: ok"}], 2)
dt = time.time() - t
print(f"{n} tokens in {dt:.1f} s = {n / dt:.0f} tok/s")
