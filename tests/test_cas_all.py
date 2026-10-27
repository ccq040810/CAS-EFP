import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_cas_all import expected_jobs, prepared_manifest, run_lock
from scripts.run_mlp_transfer import sha


class AllLauncherTests(unittest.TestCase):
    def scratch(self):
        folder = ROOT/"results/test_scratch"
        folder.mkdir(parents=True,exist_ok=True)
        return tempfile.TemporaryDirectory(dir=folder)

    def test_full_plan_has_756_unique_jobs_and_all_families(self):
        jobs = expected_jobs(["cas_chronos2","cas_moirai2","cas_sundial"],
                             ["guangdong","reale_delu"],[1,2,3],[42,43,44])
        self.assertEqual(len(jobs),756)
        self.assertEqual(len(set(jobs)),756)
        self.assertEqual(sum(j[1]=="mlp" for j in jobs),378)

    def test_prepared_resume_checks_source_protocol_and_arrays(self):
        with self.scratch() as temp:
            root = Path(temp)
            (root/"features.npz").write_bytes(b"fixture")
            manifest = dict(upstream="cas_sundial",distribution_protocol="cas_common_q10_q90_v1",
                            arrays_sha256=sha(root/"features.npz"))
            (root/"manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(prepared_manifest(root,"cas_sundial"),manifest)
            with self.assertRaises(ValueError):
                prepared_manifest(root,"cas_moirai2")
            (root/"features.npz").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                prepared_manifest(root,"cas_sundial")

    @unittest.skipIf(os.name=="nt","POSIX server lock")
    def test_duplicate_launcher_lock_is_rejected_and_then_released(self):
        with self.scratch() as temp:
            with run_lock(temp):
                with self.assertRaises(RuntimeError):
                    with run_lock(temp):
                        pass
            with run_lock(temp):
                pass


if __name__ == "__main__":
    unittest.main()
