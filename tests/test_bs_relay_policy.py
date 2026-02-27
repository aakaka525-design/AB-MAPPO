import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import train
from environment import UAVMECEnv


class TestBSRelayPolicy(unittest.TestCase):
    def test_invalid_bs_relay_policy_raises(self):
        with self.assertRaises(ValueError):
            UAVMECEnv(num_mus=4, num_uavs=2, bs_relay_policy="invalid")

    def test_nearest_policy_maps_bs_relay_to_nearest_uav(self):
        env = UAVMECEnv(num_mus=2, num_uavs=2, seed=21, bs_relay_policy="nearest")
        env.reset()
        env.mu_positions[0] = np.array([10.0, 10.0], dtype=np.float32)
        env.uav_positions[0] = np.array([0.0, 0.0], dtype=np.float32)
        env.uav_positions[1] = np.array([200.0, 200.0], dtype=np.float32)
        bs_label = env.M + 1
        association = np.array([bs_label, 0], dtype=np.int64)
        offload = np.array([0.9, 0.0], dtype=np.float32)
        uav_actions = np.full((env.M, env.uav_continuous_dim), 0.5, dtype=np.float32)
        env.step({"association": association, "offload_ratio": offload}, uav_actions)
        # prev_association stores effective UAV association (1-based)
        self.assertEqual(int(env.prev_association[0]), 1)

    def test_min_load_policy_balances_relay_assignments(self):
        env = UAVMECEnv(num_mus=4, num_uavs=2, seed=22, bs_relay_policy="min_load")
        env.reset()
        env.mu_positions[:] = np.array([[50.0, 50.0], [52.0, 48.0], [48.0, 52.0], [49.0, 49.0]], dtype=np.float32)
        env.uav_positions[0] = np.array([40.0, 40.0], dtype=np.float32)
        env.uav_positions[1] = np.array([60.0, 60.0], dtype=np.float32)
        bs_label = env.M + 1
        association = np.full((env.K,), bs_label, dtype=np.int64)
        offload = np.full((env.K,), 0.8, dtype=np.float32)
        uav_actions = np.full((env.M, env.uav_continuous_dim), 0.5, dtype=np.float32)
        env.step({"association": association, "offload_ratio": offload}, uav_actions)
        counts = np.bincount(env.prev_association - 1, minlength=env.M)
        self.assertEqual(int(counts[0]), int(counts[1]))

    def test_train_cli_accepts_bs_relay_policy_and_writes_summary(self):
        captured = {}

        def _fake_env_ctor(*args, **kwargs):
            captured["env_kwargs"] = kwargs
            return type("FakeEnv", (), {"max_steps": 60})()

        class _FakeAgent:
            def collect_episode(self):
                return {
                    "mu_reward": 0.1,
                    "uav_reward": 0.2,
                    "total_cost": -0.15,
                    "weighted_energy": 1.0,
                    "weighted_energy_mu_avg": 1.1,
                    "weighted_energy_mu_total": 2.2,
                    "mu_energy": 0.3,
                    "uav_energy": 0.4,
                    "jain_fairness": 0.8,
                    "delay_violation": 0.05,
                }

            def update(self):
                return {"actor_loss": 0.01, "critic_loss": 0.02, "entropy": 0.03}

            def save(self, _path):
                return None

        def _fake_make_agent(*_args, **_kwargs):
            return _FakeAgent()

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            args = train.namespace_from_kwargs(
                algorithm="AB-MAPPO",
                total_steps=300,
                episode_length=300,
                run_dir=str(run_dir),
                disable_tensorboard=True,
                bs_relay_policy="min_load",
            )
            with patch("train.UAVMECEnv", side_effect=_fake_env_ctor), patch("train._make_agent", side_effect=_fake_make_agent):
                train.train(args)
            self.assertEqual(captured["env_kwargs"]["bs_relay_policy"], "min_load")
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["bs_relay_policy"], "min_load")


if __name__ == "__main__":
    unittest.main()
