import os
import sys
import tempfile
import time
import unittest

from run_all import _build_full_log_dir, _run_parallel_commands


class TestRunAllParallelStability(unittest.TestCase):
    def test_build_full_log_dir_is_unique(self):
        with tempfile.TemporaryDirectory() as td:
            d1 = _build_full_log_dir(td)
            d2 = _build_full_log_dir(td)
            self.assertNotEqual(d1, d2)
            self.assertTrue(os.path.isdir(d1))
            self.assertTrue(os.path.isdir(d2))

    def test_fail_fast_terminates_other_jobs(self):
        with tempfile.TemporaryDirectory() as td:
            slow_cmd = [
                sys.executable,
                "-c",
                "import signal,sys,time; signal.signal(signal.SIGTERM, lambda s,f: sys.exit(0)); time.sleep(30)",
            ]
            fail_cmd = [sys.executable, "-c", "import sys; sys.exit(3)"]

            start = time.monotonic()
            with self.assertRaises(RuntimeError) as cm:
                _run_parallel_commands(
                    [("slow_job", slow_cmd), ("fail_job", fail_cmd)],
                    max_parallel=2,
                    log_dir=td,
                    poll_interval_sec=0.05,
                )
            elapsed = time.monotonic() - start

            self.assertLess(elapsed, 5.0)
            self.assertIn("code 3", str(cm.exception))

    def test_job_timeout_raises_error(self):
        with tempfile.TemporaryDirectory() as td:
            hang_cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
            start = time.monotonic()
            with self.assertRaises(RuntimeError) as cm:
                _run_parallel_commands(
                    [("hang_job", hang_cmd)],
                    max_parallel=1,
                    log_dir=td,
                    job_timeout_sec=0.2,
                    poll_interval_sec=0.05,
                )
            elapsed = time.monotonic() - start

            self.assertLess(elapsed, 5.0)
            self.assertIn("timeout", str(cm.exception).lower())


if __name__ == "__main__":
    unittest.main()
