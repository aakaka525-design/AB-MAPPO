import sys
import tempfile
import unittest
from unittest import mock

import run_all


def _fake_specs(n):
    specs = []
    for i in range(n):
        specs.append(
            {
                "fig": str((i % 8) + 1),
                "algorithm": "AB-MAPPO",
                "setting_name": f"K60_M10_case{i}",
                "seed": 42 + (i % 3),
                "run_dir": f"/tmp/fake_run_{i}",
                "summary_path": f"/tmp/fake_run_{i}/summary.json",
                "cli_overrides": {"num_mus": 60 + (i % 4) * 10, "num_uavs": 10},
                "num_mus": 60 + (i % 4) * 10,
                "num_uavs": 10,
            }
        )
    return specs


class TestRunAllRunLevel(unittest.TestCase):
    @mock.patch("run_all._run")
    @mock.patch("run_all._build_full_log_dir", return_value="/tmp/fake_log_dir")
    @mock.patch("run_all._run_parallel_commands")
    @mock.patch("run_all._summary_matches", return_value=False)
    @mock.patch("run_all.build_run_specs")
    def test_run_full_builds_more_than_8_jobs(
        self,
        mock_build_specs,
        _mock_summary,
        mock_parallel,
        _mock_log_dir,
        _mock_run,
    ):
        mock_build_specs.return_value = _fake_specs(16)
        run_all.run_full(max_parallel=20, full_device="cpu", job_timeout_sec=0.0)

        args, kwargs = mock_parallel.call_args
        jobs = args[0]
        self.assertGreater(len(jobs), 8)
        self.assertEqual(kwargs["max_parallel"], 20)
        self.assertGreaterEqual(kwargs["threads_per_proc"], 1)

    @mock.patch("run_all._run")
    @mock.patch("run_all._build_full_log_dir", return_value="/tmp/fake_log_dir")
    @mock.patch("run_all._run_parallel_commands")
    @mock.patch("run_all._summary_matches", return_value=False)
    @mock.patch("run_all.build_run_specs")
    def test_cuda_full_forces_single_parallel(
        self,
        mock_build_specs,
        _mock_summary,
        mock_parallel,
        _mock_log_dir,
        _mock_run,
    ):
        mock_build_specs.return_value = _fake_specs(12)
        with mock.patch("run_all.os.cpu_count", return_value=24):
            run_all.run_full(max_parallel=20, full_device="cuda", job_timeout_sec=0.0)

        _, kwargs = mock_parallel.call_args
        self.assertEqual(kwargs["max_parallel"], 1)
        self.assertEqual(kwargs["threads_per_proc"], 24)

    def test_spawn_logged_process_sets_thread_env(self):
        with tempfile.TemporaryDirectory() as td:
            cmd = [
                sys.executable,
                "-c",
                "import os; print(os.getenv('OMP_NUM_THREADS')); print(os.getenv('MKL_NUM_THREADS'))",
            ]
            job = run_all._spawn_logged_process("env_job", cmd, td, threads_per_proc=3)
            rc = job["process"].wait(timeout=5.0)
            run_all._close_job_stream(job)
            self.assertEqual(rc, 0)

            with open(job["log_file"], "r", encoding="utf-8") as f:
                out = f.read()
            self.assertIn("3", out)


if __name__ == "__main__":
    unittest.main()
