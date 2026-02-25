import unittest

from run_all import _build_full_cmd


class TestRunAllFullDevice(unittest.TestCase):
    def test_default_cpu_device_in_full_cmd(self):
        cmd = _build_full_cmd("6", "cpu")
        self.assertIn("--device", cmd)
        idx = cmd.index("--device")
        self.assertEqual(cmd[idx + 1], "cpu")

    def test_override_cuda_device_in_full_cmd(self):
        cmd = _build_full_cmd("8", "cuda")
        idx = cmd.index("--device")
        self.assertEqual(cmd[idx + 1], "cuda")


if __name__ == "__main__":
    unittest.main()
