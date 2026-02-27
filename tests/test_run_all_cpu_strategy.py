import unittest
from unittest import mock

import run_all


class TestRunAllCpuStrategy(unittest.TestCase):
    @mock.patch("run_all.os.cpu_count", return_value=32)
    @mock.patch("run_all._read_mem_available_gb", return_value=100.0)
    def test_estimate_cpu_parallel_capped_by_cores(self, _mock_mem, _mock_cpu):
        effective, free_gb = run_all._estimate_cpu_parallel(40)
        self.assertEqual(effective, 32)
        self.assertEqual(free_gb, 100.0)

    @mock.patch("run_all.os.cpu_count", return_value=32)
    @mock.patch("run_all._read_mem_available_gb", return_value=5.0)
    def test_estimate_cpu_parallel_falls_back_on_low_memory(self, _mock_mem, _mock_cpu):
        effective, _free_gb = run_all._estimate_cpu_parallel(20)
        self.assertEqual(effective, 1)

    def test_pop_next_schedulable_job_prefers_non_heavy_when_heavy_slots_full(self):
        pending = [
            {"name": "heavy_a", "cmd": ["echo", "a"], "is_heavy": True},
            {"name": "heavy_b", "cmd": ["echo", "b"], "is_heavy": True},
            {"name": "light_c", "cmd": ["echo", "c"], "is_heavy": False},
        ]
        active = [{"name": "heavy_running", "is_heavy": True}]
        picked = run_all._pop_next_schedulable_job(pending, active, heavy_parallel_limit=1)
        self.assertIsNotNone(picked)
        self.assertEqual(picked["name"], "light_c")
        self.assertEqual(len(pending), 2)

    def test_pop_next_schedulable_job_returns_none_when_only_heavy_blocked(self):
        pending = [
            {"name": "heavy_a", "cmd": ["echo", "a"], "is_heavy": True},
            {"name": "heavy_b", "cmd": ["echo", "b"], "is_heavy": True},
        ]
        active = [{"name": "heavy_running", "is_heavy": True}]
        picked = run_all._pop_next_schedulable_job(pending, active, heavy_parallel_limit=1)
        self.assertIsNone(picked)


if __name__ == "__main__":
    unittest.main()
