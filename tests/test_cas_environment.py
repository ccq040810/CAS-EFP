"""Preflight must report incomplete resources rather than falsely certify them."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import check_cas_environment as check


def scratch():
    folder = ROOT/'results/test_scratch'
    folder.mkdir(parents=True, exist_ok=True)
    return folder


class PreflightTests(unittest.TestCase):
    def test_weight_index_missing_shard_is_not_complete(self):
        with tempfile.TemporaryDirectory(dir=scratch()) as temp:
            folder = Path(temp)
            (folder/"config.json").write_text("{}")
            (folder/"model-00001.safetensors").write_bytes(b"fixture")
            (folder/"model.safetensors.index.json").write_text(json.dumps({
                "weight_map":{"a":"model-00001.safetensors", "b":"model-00002.safetensors"}}))
            result = check.inspect_weights(folder)
            self.assertTrue(any("Missing weight shard" in e for e in result["errors"]))
            (folder/"model-00002.safetensors").write_bytes(b"fixture")
            self.assertFalse(check.inspect_weights(folder)["errors"])

    def test_missing_data_and_splits_are_listed(self):
        with tempfile.TemporaryDirectory(dir=scratch()) as temp:
            result = check.inspect_inputs(Path(temp))
            self.assertFalse(result["ready"])
            self.assertEqual(len(result["errors"]), 8)

    def test_one_failed_model_blocks_all_ready_and_reports_every_component(self):
        with tempfile.TemporaryDirectory(dir=scratch()) as temp:
            output = Path(temp)/"checks"
            commands = []
            def process(cmd, **kwargs):
                commands.append((cmd, kwargs))
                name = cmd[cmd.index("--probe")+1]
                ready = name != "moirai2"
                dest = Path(cmd[cmd.index("--report")+1])
                check.save(dest, dict(ready=ready, errors=[] if ready else ["No module named uni2ts"]))
                return SimpleNamespace(returncode=0 if ready else 2)
            argv = ["check", "--old-root", temp, "--output-root", str(output),
                    "--sundial-python", "separate-sundial-python"]
            with patch.object(sys,"argv",argv), patch.object(check,"inspect_inputs",return_value=dict(ready=True,errors=[])), \
                 patch.object(check.subprocess,"run",side_effect=process), contextlib.redirect_stdout(io.StringIO()):
                code = check.main()
            report = json.loads((output/"report.json").read_text())
            self.assertEqual(code, 2)
            self.assertFalse(report["all_ready"])
            self.assertEqual(len(report["components"]), 4)
            self.assertEqual(commands[-1][0][0], "separate-sundial-python")
            self.assertTrue(all(kw["env"]["HF_HUB_OFFLINE"] == "1" for _,kw in commands))

    def test_crashed_probe_cannot_reuse_stale_ready_report(self):
        with tempfile.TemporaryDirectory(dir=scratch()) as temp:
            output = Path(temp)
            for name in ("downstream", *check.FAMILIES):
                check.save(output/(name+".json"),dict(ready=True,errors=[]))
            with patch.object(sys,"argv",["check","--old-root",temp,"--output-root",temp]), \
                 patch.object(check,"inspect_inputs",return_value=dict(ready=True,errors=[])), \
                 patch.object(check.subprocess,"run",return_value=SimpleNamespace(returncode=1)), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = check.main()
            self.assertEqual(code,2)
            report = json.loads((output/"report.json").read_text())
            self.assertTrue(all(not x["ready"] for x in report["components"].values()))


if __name__ == "__main__":
    unittest.main()
