import argparse
import contextlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import ablit_cache as m


class CacheSwitchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cache = self.root / "rank0of2-test.bin"
        body = b"untouched" + b"old!" + b"tail"
        rows = [["before", "uint8", [9], 0, 9],
                ["wo_b", "uint8", [4], 9, 4],
                ["after", "uint8", [4], 13, 4]]
        raw = json.dumps(rows).encode()
        self.original = body + raw + struct.pack("<Q8s", len(body), b"TFDSRK01")
        self.cache.write_bytes(self.original)
        self.state = self.root / "state"
        self.state.mkdir()
        (self.state / "0.stock").write_bytes(b"old!")
        (self.state / "0.abliterated").write_bytes(b"new!")
        manifest = dict(cache=self.cache.name, index_sha=m.digest(raw), size=len(self.original),
                        entries=[dict(id=0, key="wo_b", offset=9, length=4,
                                      stock=m.digest(b"old!"), abliterated=m.digest(b"new!"))])
        (self.state / "manifest.json").write_text(json.dumps(manifest))

    def run_action(self, action="switch", mode="abliterated", running="", expect=None):
        args = argparse.Namespace(action=action, mode=mode, state=self.state,
                                  cache=self.cache, sidecar=None, model=self.root,
                                  container="engine", expect=expect)
        with patch.object(m.subprocess, "check_output", return_value=running), contextlib.redirect_stdout(io.StringIO()):
            m.run(args)

    def test_roundtrip_preserves_entire_cache_and_footer(self):
        self.run_action()
        self.assertEqual(self.cache.read_bytes(), self.original[:9] + b"new!" + self.original[13:])
        self.run_action(mode="stock")
        self.assertEqual(self.cache.read_bytes(), self.original)

    def test_running_engine_and_builder_refused(self):
        for running in ("engine\n", "engine-inputs\n"):
            with self.assertRaisesRegex(ValueError, "stop the engine"):
                self.run_action(running=running)
            self.assertEqual(self.cache.read_bytes(), self.original)

    def test_damage_refused_before_writing(self):
        (self.state / "0.stock").write_bytes(b"oops")
        with self.assertRaisesRegex(ValueError, "damaged stock"):
            self.run_action()
        self.assertEqual(self.cache.read_bytes(), self.original)

    def test_unknown_cache_bytes_refused(self):
        with self.cache.open("r+b") as f:
            f.seek(9)
            f.write(b"oops")
        with self.assertRaisesRegex(ValueError, "cache changed"):
            self.run_action()

    def test_interrupted_write_recovers_from_durable_backup(self):
        (self.state / "pending").write_text("abliterated")
        with self.cache.open("r+b") as f:
            f.seek(9)
            f.write(b"ne")
        self.run_action(mode="stock")
        self.assertEqual(self.cache.read_bytes(), self.original)
        self.assertFalse((self.state / "pending").exists())

    def test_changed_cache_identity_refused(self):
        manifest = json.loads((self.state / "manifest.json").read_text())
        manifest["index_sha"] = "wrong"
        (self.state / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "identity differs"):
            self.run_action()
        self.assertEqual(self.cache.read_bytes(), self.original)

    def test_sidecar_rejects_wrong_pack(self):
        with self.assertRaisesRegex(ValueError, "unsupported split"):
            m.replacement(None, "layers.10.attn.wo_b|None|(0, 2048)|tr", "int16", [128, 320, 96])
        with self.assertRaisesRegex(ValueError, "unsupported tensor"):
            m.replacement(None, "layers.10.attn.wo_b|None|(0, 4096)|tr", "int16", [256, 320, 96])

    def test_expected_mode_checks_bytes(self):
        self.run_action(action="check", expect="stock")
        with self.assertRaisesRegex(ValueError, "expected abliterated"):
            self.run_action(action="check", expect="abliterated")
        self.run_action()
        self.run_action(action="check", expect="abliterated")


if __name__ == "__main__":
    unittest.main()
