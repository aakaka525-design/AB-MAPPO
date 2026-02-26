import os
import sys
import tempfile
import unittest
from unittest import mock

import run_all


def _gb_to_bytes(v):
    return int(float(v) * (1024**3))


def _fake_jobs(n):
    jobs = []
    for i in range(n):
        jobs.append((f"job_{i}", ["echo", str(i)]))
    return jobs


class _FakeCuda:
    def __init__(self, free_gb, total_gb):
        self._free = _gb_to_bytes(free_gb)
        self._total = _gb_to_bytes(total_gb)

    def is_available(self):
        return True

    def mem_get_info(self, *_args, **_kwargs):
        return self._free, self._total


class _FakeTorch:
    def __init__(self, free_gb, total_gb):
        self.cuda = _FakeCuda(free_gb=free_gb, total_gb=total_gb)


class TestRunAllCudaAdaptive(unittest.TestCase):
    def test_estimate_cuda_parallel_caps_to_hard_limit(self):
        fake_torch = _FakeTorch(free_gb=20.0, total_gb=24.0)
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            effective, free_gb, total_gb = run_all._estimate_cuda_parallel(32, "cuda:0")
        self.assertEqual(effective, run_all.CUDA_PARALLEL_HARD_CAP)
        self.assertGreater(free_gb, 19.0)
        self.assertGreater(total_gb, 23.0)

    def test_estimate_cuda_parallel_falls_back_to_one_on_low_memory(self):
        fake_torch = _FakeTorch(free_gb=1.8, total_gb=24.0)
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            effective, free_gb, total_gb = run_all._estimate_cuda_parallel(12, "cuda")
        self.assertEqual(effective, 1)
        self.assertIsNotNone(free_gb)
        self.assertIsNotNone(total_gb)

    @mock.patch("run_all._run")
    @mock.patch("run_all._estimate_cuda_parallel", return_value=(16, 10.0, 24.0))
    @mock.patch("run_all._build_run_job_specs")
    @mock.patch("run_all._run_parallel_commands")
    def test_run_full_retries_with_halved_parallel_on_cuda_oom(
        self,
        mock_run_parallel,
        mock_build_jobs,
        _mock_estimate,
        _mock_run,
    ):
        with tempfile.TemporaryDirectory() as td:
            oom_log = os.path.join(td, "oom.log")
            with open(oom_log, "w", encoding="utf-8") as f:
                f.write("RuntimeError: CUDA out of memory\n")

            mock_build_jobs.return_value = _fake_jobs(3)
            mock_run_parallel.side_effect = [
                RuntimeError(f"Job failed with code 1: x (cmd); log={oom_log}"),
                None,
            ]

            with mock.patch("run_all._build_full_log_dir", return_value=td), mock.patch(
                "run_all.os.cpu_count", return_value=24
            ):
                run_all.run_full(max_parallel=12, full_device="cuda", job_timeout_sec=0.0)

        self.assertEqual(mock_run_parallel.call_count, 2)
        first = mock_run_parallel.call_args_list[0].kwargs
        second = mock_run_parallel.call_args_list[1].kwargs
        self.assertEqual(first["max_parallel"], 16)
        self.assertEqual(second["max_parallel"], 8)
        self.assertEqual(mock_build_jobs.call_count, 2)


if __name__ == "__main__":
    unittest.main()
