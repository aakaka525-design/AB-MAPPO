from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestGenerateFiguresStrictMode(unittest.TestCase):
    def test_missing_inputs_fail_by_default(self):
        with tempfile.TemporaryDirectory() as exp_root, tempfile.TemporaryDirectory() as out_dir:
            cmd = [
                sys.executable,
                "generate_figures.py",
                "--figs",
                "all",
                "--exp_root",
                exp_root,
                "--output_dir",
                out_dir,
            ]
            p = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
            self.assertEqual(p.returncode, 2, msg=p.stdout + "\n" + p.stderr)

    def test_allow_missing_keeps_success_exit(self):
        with tempfile.TemporaryDirectory() as exp_root, tempfile.TemporaryDirectory() as out_dir:
            cmd = [
                sys.executable,
                "generate_figures.py",
                "--figs",
                "all",
                "--exp_root",
                exp_root,
                "--output_dir",
                out_dir,
                "--allow-missing",
            ]
            p = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, msg=p.stdout + "\n" + p.stderr)
            self.assertIn("[skip]", p.stdout)


if __name__ == "__main__":
    unittest.main()
