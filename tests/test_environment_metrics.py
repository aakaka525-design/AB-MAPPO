import unittest

import numpy as np

import config as cfg
from environment import UAVMECEnv


class TestEnvironmentMetrics(unittest.TestCase):
    def test_collision_penalty_positive_when_too_close(self):
        env = UAVMECEnv(num_mus=2, num_uavs=2)
        env.reset()
        env.uav_positions[0] = np.array([100.0, 100.0], dtype=np.float32)
        env.uav_positions[1] = np.array([100.0 + cfg.D_MIN * 0.5, 100.0], dtype=np.float32)
        penalty = env._penalty_collision(0)
        self.assertGreater(penalty, 0.0)

    def test_boundary_clip_uses_width_and_height(self):
        env = UAVMECEnv(num_mus=2, num_uavs=1, area_width=100.0, area_height=50.0)
        env.reset()
        env.uav_positions[0] = np.array([99.0, 49.0], dtype=np.float32)
        env.uav_velocities[0] = np.array([0.0, 0.0], dtype=np.float32)
        env._update_uav_positions(np.array([[1.0, 1.0]], dtype=np.float32))
        self.assertGreaterEqual(env.uav_positions[0, 0], 0.0)
        self.assertLessEqual(env.uav_positions[0, 0], 100.0)
        self.assertGreaterEqual(env.uav_positions[0, 1], 0.0)
        self.assertLessEqual(env.uav_positions[0, 1], 50.0)

    def test_jain_fairness_in_valid_range(self):
        env = UAVMECEnv(num_mus=6, num_uavs=2)
        obs = env.reset()
        assoc = np.random.randint(0, 3, size=6)
        offload = np.random.uniform(0, 1, size=6)
        uav_act = np.random.uniform(0, 1, size=(2, env.uav_continuous_dim))
        _, _, _, info = env.step({"association": assoc, "offload_ratio": offload}, uav_act)
        self.assertGreaterEqual(info["jain_fairness"], 0.0)
        self.assertLessEqual(info["jain_fairness"], 1.0 + 1e-6)
        self.assertIn("weighted_energy_mu_total", info)
        self.assertIn("weighted_energy_mu_avg", info)

    def test_jain_fairness_matches_inverse_energy_definition(self):
        env = UAVMECEnv(num_mus=8, num_uavs=2)
        env.reset()
        assoc = np.random.randint(0, 3, size=8)
        offload = np.random.uniform(0.0, 1.0, size=8)
        uav_act = np.random.uniform(0.0, 1.0, size=(2, env.uav_continuous_dim))
        _, _, _, info = env.step({"association": assoc, "offload_ratio": offload}, uav_act)
        mu_energy = info["mu_energy"]
        utility = 1.0 / (mu_energy + 1e-8)
        expected = float((utility.sum() ** 2) / (len(utility) * np.sum(utility**2) + 1e-8))
        self.assertAlmostEqual(info["jain_fairness"], expected, places=6)

    def test_wo_dt_noise_is_clipped(self):
        env = UAVMECEnv(num_mus=6, num_uavs=2, wo_dt_noise_mode=True)
        obs = env.reset()
        self.assertTrue(np.all(obs["mu_obs"] >= 0.0))
        self.assertTrue(np.all(obs["mu_obs"] <= 1.0))
        self.assertTrue(np.all(obs["uav_obs"] >= 0.0))
        self.assertTrue(np.all(obs["uav_obs"] <= 1.0))


if __name__ == "__main__":
    unittest.main()
