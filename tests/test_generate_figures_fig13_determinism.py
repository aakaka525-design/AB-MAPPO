import unittest

import numpy as np

from generate_figures import _run_heuristic_trajectory


class TestFig13Determinism(unittest.TestCase):
    def test_same_seed_same_trajectory(self):
        scenario = {"K": 12, "M": 3, "W": 500, "T": 30, "cluster": True, "uav_start": "corner"}
        mu1, tr1 = _run_heuristic_trajectory(scenario, seed=20260225)
        mu2, tr2 = _run_heuristic_trajectory(scenario, seed=20260225)
        np.testing.assert_allclose(mu1, mu2, rtol=0.0, atol=0.0)
        self.assertEqual(len(tr1), len(tr2))
        for a, b in zip(tr1, tr2):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0.0, atol=0.0)

    def test_different_seed_different_trajectory(self):
        scenario = {"K": 12, "M": 3, "W": 500, "T": 30, "cluster": True, "uav_start": "corner"}
        mu1, tr1 = _run_heuristic_trajectory(scenario, seed=7)
        mu2, tr2 = _run_heuristic_trajectory(scenario, seed=8)
        all_equal = np.allclose(mu1, mu2)
        if all_equal:
            all_equal = all(np.allclose(np.asarray(a), np.asarray(b)) for a, b in zip(tr1, tr2))
        self.assertFalse(all_equal)


if __name__ == "__main__":
    unittest.main()
