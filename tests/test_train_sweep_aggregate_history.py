import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from train_sweep import _aggregate_history


class _FakeNpz:
    def __init__(self, data):
        self._data = data
        self.closed = False
        self.files = list(data.keys())

    def __getitem__(self, key):
        return self._data[key]

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class TestAggregateHistory(unittest.TestCase):
    def test_aggregate_history_closes_loaded_npz(self):
        with tempfile.TemporaryDirectory() as td:
            algo_dir = f"{td}/seed_42"
            import os

            os.makedirs(algo_dir, exist_ok=True)
            h_path = f"{algo_dir}/history.npz"
            open(h_path, "wb").close()

            payload = {
                "episode": np.array([0, 1, 2], dtype=np.float32),
                "step": np.array([10, 20, 30], dtype=np.float32),
                "mu_reward": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "uav_reward": np.array([0.5, 0.6, 0.7], dtype=np.float32),
                "weighted_energy_mu_avg": np.array([4.0, 3.0, 2.0], dtype=np.float32),
                "jain_fairness": np.array([0.2, 0.3, 0.4], dtype=np.float32),
            }
            fake = _FakeNpz(payload)
            out = f"{td}/agg.json"

            with patch("train_sweep.np.load", return_value=fake):
                _aggregate_history({"AB-MAPPO": [algo_dir]}, out)

            self.assertTrue(fake.closed)
            with open(out, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("AB-MAPPO", data["algorithms"])


if __name__ == "__main__":
    unittest.main()
