# images.py DOC record|check FILE: a conversation with an image and a file (two turns), then the same second turn with
# another image. Record the greedy replies on an engine that runs image prompts whole, then check them after a change:
# the second turn should resume past the image with the same reply, and the other image should not resume past it.
import base64, json, struct, sys, zlib
from common import chat

doc, mode, path = open(sys.argv[1]).read(), sys.argv[2], sys.argv[3]


def png(color, square):
    w = h = 224; rows = b""
    for y in range(h):
        row = bytearray(b"\x00")
        for x in range(w):
            inside = (abs(x - 112) < 70 and abs(y - 112) < 70) if square else ((x - 112) ** 2 + (y - 112) ** 2 < 70 ** 2)
            row += bytes(color if inside else (255, 255, 255))
        rows += bytes(row)
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    raw = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(raw).decode()


red_circle, blue_square = png((220, 20, 20), False), png((20, 40, 220), True)
system = ("You are a careful assistant. Answer in English, briefly. " * 30).strip()


def turn1(img):
    return [{"role": "system", "content": system}, {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": img}},
        {"type": "text", "text": "Here is the image, then a file.\n\n" + doc + "\n\nFirst: what shape and color is in the image? One sentence."}]}]


out = {}
again = {"role": "user", "content": "Now name the color again, in one word."}
n, c, out["A1"] = chat(turn1(red_circle))
print(f"A1 (red)   prompt={n} cached={c} | {out['A1'][:80]!r}")
n, c, out["A2"] = chat(turn1(red_circle) + [{"role": "assistant", "content": out["A1"]}, again])
print(f"A2 (red)   prompt={n} cached={c} | {out['A2'][:80]!r}")
n, c, out["B2"] = chat(turn1(blue_square) + [{"role": "assistant", "content": out["A1"]}, again])
print(f"B2 (blue)  prompt={n} cached={c} | {out['B2'][:80]!r}")
if mode == "record":
    json.dump(out, open(path, "w"))
    print("recorded")
else:
    ref = json.load(open(path))
    bad = [k for k in out if out[k] != ref[k]]
    print("RESULT:", "identical to the reference" if not bad else f"DIFFERENT: {bad}")
