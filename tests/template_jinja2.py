# template_jinja2.py OLD NEW N: renders N random conversations with both templates in jinja2 (Hugging Face's settings)
# and stops at the first one that differs
import json, random, sys
import jinja2, jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment
from conversations import conversation


def raise_exception(m):
    raise jinja2.exceptions.TemplateError(m)


env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
env.filters["tojson"] = lambda x, indent=None, separators=None, sort_keys=False: json.dumps(x, ensure_ascii=False, indent=indent, separators=separators, sort_keys=sort_keys)
env.globals["raise_exception"] = raise_exception
old, new = (env.from_string(open(p, encoding="utf-8").read()) for p in sys.argv[1:3])


def render(t, msgs, kw):
    try:
        return "ok", t.render(messages=msgs, bos_token="", **kw)
    except Exception as e:
        return "error", f"{type(e).__name__}: {e}"


for i in range(int(sys.argv[3])):
    msgs, kw = conversation(random.Random(i))
    a, b = render(old, msgs, kw), render(new, msgs, kw)
    if a != b:
        sys.exit(f"seed {i} differs: {a[1][:200]!r} / {b[1][:200]!r}")
print(f"{sys.argv[3]} conversations render identically")
