# screenshot_turns.py TURNS: a conversation that sends one new 1280x633 screenshot a turn and every earlier one again, as
# OpenCode does. Prints each turn's prompt, the tokens the prompt cache resumed and the seconds to the reply: a turn
# should cost about its new image, not all of them (run it under memory_peak.sh for rank 0's peak).
import base64, random, struct, sys, time, zlib
from common import chat

w, h = 1280, 633


def screenshot(seed):
    r = random.Random(seed)
    bands = [bytes(r.choice([(250, 250, 250), (238, 241, 246), (255, 255, 255), (30, 30, 46)])) for _ in range(h // 40 + 1)]
    rows = []
    for y in range(h):
        row = bytearray(bands[y // 40] * w)
        if y % 40 in (12, 13, 14):  # a line of "text"
            for x in range(r.randrange(40, 600), r.randrange(700, w), 3):
                row[3 * x:3 * x + 3] = b"\x14\x14\x14"
        rows.append(b"\x00" + bytes(row))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(b"".join(rows), 6)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


if __name__ == "__main__":
    turns = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    msgs = [{"role": "system", "content": "You look at screenshots of a terminal and answer in a few words."}]
    for t in range(turns):
        msgs.append({"role": "user", "content": [{"type": "text", "text": f"Screenshot {t + 1}: how many text lines does it show, roughly?"},
                                                 {"type": "image_url", "image_url": {"url": screenshot(t)}}]})
        t0 = time.time()
        prompt, cached, text = chat(msgs, max_tokens=16)
        print(f"turn {t + 1}: {t + 1} images, prompt {prompt}, resumed {cached}, {time.time() - t0:.1f} s: {text!r}", flush=True)
        msgs.append({"role": "assistant", "content": text})
