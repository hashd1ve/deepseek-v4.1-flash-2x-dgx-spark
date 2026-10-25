#!/usr/bin/env python3
"""Reversible L10-35 wo_b overlay on an existing TensorFold TP2 rank cache.

Only the derived cache changes; checkpoint shards and MTP stay untouched.
prepare/check are read-only on the cache. switch requires the engine stopped.
"""
import argparse
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess


def digest(data):
    return hashlib.sha256(data).hexdigest()


def atomic(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def cache_index(f):
    f.seek(0, 2)
    size = f.tell()
    f.seek(-16, 2)
    off, magic = struct.unpack("<Q8s", f.read(16))
    if magic != b"TFDSRK01" or not 0 <= off < size - 16:
        raise ValueError("unsupported rank cache")
    f.seek(off)
    raw = f.read(size - 16 - off)
    rows = json.loads(raw)
    end = 0
    for key, dtype, shape, at, length in rows:
        if at != end or length <= 0 or at + length > off:
            raise ValueError("invalid rank cache index")
        end = at + length
    if end != off:
        raise ValueError("incomplete rank cache index")
    return rows, digest(raw), size


class Sidecar:
    def __init__(self, path):
        self.f = path.open("rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        if n > 16 * 1024 * 1024:
            raise ValueError("invalid safetensors header")
        self.header = json.loads(self.f.read(n))
        self.base = 8 + n

    def tensor(self, key, dtype, shape):
        t = self.header[key]
        if t["dtype"] != dtype or t["shape"] != shape:
            raise ValueError(f"incompatible sidecar tensor {key}: {t}")
        lo, hi = t["data_offsets"]
        self.f.seek(self.base + lo)
        data = self.f.read(hi - lo)
        if len(data) != hi - lo:
            raise ValueError("truncated sidecar")
        return data


def replacement(sidecar, key, dtype, shape):
    prefix, cols, rows, part = key.split("|")
    # This recipe is TP2: input halves, all 5120 output columns, K=5.
    span = ast.literal_eval(rows)
    if cols != "None" or span not in ((0, 4096), (4096, 8192)):
        raise ValueError(f"unsupported split: {key}")
    lo, hi = span
    if part == "tr" and dtype == "int16" and shape == [256, 320, 80]:
        data = sidecar.tensor(prefix + ".trellis", "I16", [512, 320, 80])
        stride = 320 * 80 * 2
        return data[lo // 16 * stride:hi // 16 * stride]
    if part == "suh" and dtype == "float16" and shape == [4096]:
        return sidecar.tensor(prefix + ".suh", "F16", [8192])[lo * 2:hi * 2]
    if part == "svh" and dtype == "float16" and shape == [5120]:
        return sidecar.tensor(prefix + ".svh", "F16", [5120])
    raise ValueError(f"unsupported tensor: {key}")


def selected(rows):
    out = [r for r in rows if re.fullmatch(r"layers\.(?:1[0-9]|2[0-9]|3[0-5])\.attn\.wo_b\|.*\|(?:tr|suh|svh)", r[0])]
    if len(out) != 78 or len({r[0] for r in out}) != 78:
        raise ValueError(f"expected 78 wo_b cache entries, found {len(out)}")
    return out


def verify(f, state, manifest):
    for row in manifest["entries"]:
        f.seek(row["offset"])
        current = digest(f.read(row["length"]))
        if current not in (row["stock"], row["abliterated"]) and not (state / "pending").exists():
            raise ValueError(f"cache changed outside overlay: {row['key']}")
        for mode in ("stock", "abliterated"):
            if digest((state / f"{row['id']}.{mode}").read_bytes()) != row[mode]:
                raise ValueError(f"damaged {mode} backup: {row['key']}")


def run(args):
    args.state.mkdir(parents=True, exist_ok=True)
    with (args.state / "lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        mode = "r+b" if args.action == "switch" else "rb"
        with args.cache.open(mode) as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            rows, index_sha, size = cache_index(f)
            mp = args.state / "manifest.json"
            if not mp.exists():
                if args.action != "prepare" or args.sidecar is None:
                    raise ValueError("prepare the overlay first with --sidecar")
                sidecar = Sidecar(args.sidecar)
                weight_map = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
                entries = []
                for i, (key, dtype, shape, offset, length) in enumerate(selected(rows)):
                    new = replacement(sidecar, key, dtype, shape)
                    if len(new) != length:
                        raise ValueError(f"size mismatch: {key}")
                    f.seek(offset)
                    stock = f.read(length)
                    if len(stock) != length:
                        raise ValueError("short cache read")
                    prefix, _, _, part = key.split("|")
                    suffix = "trellis" if part == "tr" else part
                    original = Sidecar(args.model / weight_map[prefix + "." + suffix])
                    try:
                        expected = replacement(original, key, dtype, shape)
                    finally:
                        original.f.close()
                    if stock != expected:
                        raise ValueError(f"cache differs from original checkpoint: {key}")
                    for name, data in (("stock", stock), ("abliterated", new)):
                        atomic(args.state / f"{i}.{name}", data)
                    entries.append(dict(id=i, key=key, offset=offset, length=length,
                                        stock=digest(stock), abliterated=digest(new)))
                manifest = dict(cache=args.cache.name, index_sha=index_sha, size=size, entries=entries)
                atomic(mp, json.dumps(manifest).encode())
            manifest = json.loads(mp.read_text())
            if (manifest["cache"], manifest["index_sha"], manifest["size"]) != (args.cache.name, index_sha, size):
                raise ValueError("cache identity differs from saved overlay; refuse to switch")
            verify(f, args.state, manifest)
            if getattr(args, "expect", None):
                for row in manifest["entries"]:
                    f.seek(row["offset"])
                    if digest(f.read(row["length"])) != row[args.expect]:
                        raise ValueError(f"expected {args.expect}: {row['key']}")
            if args.action == "switch":
                # Check the daemon, not merely whether the old container exists.
                names = subprocess.check_output(["docker", "ps", "--format", "{{.Names}}"], text=True).splitlines()
                if any(n in names for n in (args.container, args.container + "-inputs")):
                    raise ValueError("stop the engine and cache builder before switching")
                atomic(args.state / "pending", args.mode.encode())
                for row in manifest["entries"]:
                    data = (args.state / f"{row['id']}.{args.mode}").read_bytes()
                    f.seek(row["offset"])
                    f.write(data)
                f.flush()
                os.fsync(f.fileno())
                # Re-read every changed range before allowing a launch.
                for row in manifest["entries"]:
                    f.seek(row["offset"])
                    if digest(f.read(row["length"])) != row[args.mode]:
                        raise ValueError("post-write verification failed")
                atomic(args.state / "active", args.mode.encode())
                (args.state / "pending").unlink()
            print(f"{args.action}: {args.cache.name}: 78 tensors verified" +
                  (f", mode={args.mode}" if args.action == "switch" else ""))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "check", "switch"))
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--sidecar", type=Path)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--mode", choices=("stock", "abliterated"), default="stock")
    p.add_argument("--expect", choices=("stock", "abliterated"), help="require all current bytes to match this mode")
    p.add_argument("--container", default="dsv41-1017-tf")
    run(p.parse_args())
