import importlib
import io
import unittest
from contextlib import redirect_stdout
from unittest import mock


class TestBenchmarkMainGuard(unittest.TestCase):
    def test_import_bench_parallel_does_not_spawn_processes(self):
        with mock.patch("subprocess.Popen") as mock_popen:
            mod = importlib.import_module("bench_parallel")
            self.assertTrue(hasattr(mod, "main"))
            mock_popen.assert_not_called()

    def test_import_benchmark_device_does_not_run_subprocess(self):
        with mock.patch("subprocess.run") as mock_run, mock.patch("subprocess.Popen") as mock_popen:
            mod = importlib.import_module("benchmark_device")
            self.assertTrue(hasattr(mod, "main"))
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_bench_parallel_run_parallel_uses_communicate_and_checks_rc(self):
        mod = importlib.import_module("bench_parallel")

        ok_proc = mock.Mock()
        ok_proc.communicate.return_value = (b"", b"")
        ok_proc.returncode = 0
        fail_proc = mock.Mock()
        fail_proc.communicate.return_value = (b"", b"err")
        fail_proc.returncode = 1

        with mock.patch.object(mod.subprocess, "Popen", side_effect=[ok_proc, fail_proc]):
            with self.assertRaises(RuntimeError):
                mod.run_parallel(2, device="cpu", threads_per_proc=1)

        self.assertTrue(ok_proc.communicate.called)
        self.assertTrue(fail_proc.communicate.called)

    def test_benchmark_device_run_parallel_uses_communicate_and_checks_rc(self):
        mod = importlib.import_module("benchmark_device")

        ok_proc = mock.Mock()
        ok_proc.communicate.return_value = (b"", b"")
        ok_proc.returncode = 0
        fail_proc = mock.Mock()
        fail_proc.communicate.return_value = (b"", b"err")
        fail_proc.returncode = 2

        with mock.patch.object(mod.subprocess, "Popen", side_effect=[ok_proc, fail_proc]):
            with self.assertRaises(RuntimeError):
                mod.run_parallel(2)

        self.assertTrue(ok_proc.communicate.called)
        self.assertTrue(fail_proc.communicate.called)

    def test_benchmark_device_main_ignores_failed_results_when_selecting_best(self):
        mod = importlib.import_module("benchmark_device")

        with mock.patch.object(
            mod,
            "run_bench",
            side_effect=[
                (10.0, "0.8", 0),
                (9.0, "0.9", 0),
                (1.0, "?", 1),
                (1.2, "?", 1),
            ],
        ), mock.patch.object(mod, "run_parallel", return_value=20.0):
            buf = io.StringIO()
            with redirect_stdout(buf):
                mod.main()

        out = buf.getvalue()
        self.assertIn("最快方案: CPU single (OMP=1)", out)


if __name__ == "__main__":
    unittest.main()
