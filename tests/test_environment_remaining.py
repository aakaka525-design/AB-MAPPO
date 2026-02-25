import unittest
from unittest.mock import patch

import numpy as np

import config as cfg
from environment import UAVMECEnv


class TestEnvironmentRemaining(unittest.TestCase):
    def test_env_reset_is_reproducible_with_seed(self):
        env1 = UAVMECEnv(num_mus=5, num_uavs=2, seed=123)
        env2 = UAVMECEnv(num_mus=5, num_uavs=2, seed=123)
        env3 = UAVMECEnv(num_mus=5, num_uavs=2, seed=124)

        env1.reset()
        env2.reset()
        env3.reset()

        np.testing.assert_allclose(env1.get_state(), env2.get_state(), rtol=0.0, atol=0.0)
        self.assertFalse(np.allclose(env1.get_state(), env3.get_state()))

    def test_collision_penalty_vector_matches_scalar_method(self):
        env = UAVMECEnv(num_mus=3, num_uavs=3)
        env.reset()
        env.uav_positions = np.array(
            [[0.0, 0.0], [cfg.D_MIN * 0.4, 0.0], [cfg.D_MIN * 1.2, 0.0]],
            dtype=np.float32,
        )

        vector_penalty = env._collision_penalties()
        scalar_penalty = np.array([env._penalty_collision(m) for m in range(env.M)], dtype=np.float32)
        np.testing.assert_allclose(vector_penalty, scalar_penalty, rtol=1e-6, atol=1e-6)

    def test_step_uses_batch_rate_api_for_edge_offload(self):
        env = UAVMECEnv(num_mus=4, num_uavs=2, seed=7)
        env.reset()

        association = np.array([1, 1, 2, 0], dtype=np.int64)
        offload_ratio = np.array([0.6, 0.8, 0.5, 0.0], dtype=np.float32)
        uav_actions = np.full((2, env.uav_continuous_dim), 0.5, dtype=np.float32)

        def fake_batch_rate(mu_pos, uav_pos, bandwidth):
            self.assertEqual(mu_pos.shape[0], 3)
            self.assertEqual(uav_pos.shape[0], 3)
            self.assertEqual(bandwidth.shape[0], 3)
            return np.full((3,), 2e6, dtype=np.float32)

        with patch("environment.compute_mu_uav_rate", side_effect=RuntimeError("scalar API should not be used")), patch(
            "environment.compute_mu_uav_rate_batch", side_effect=fake_batch_rate
        ) as mock_batch:
            env.step({"association": association, "offload_ratio": offload_ratio}, uav_actions)
            self.assertEqual(mock_batch.call_count, 1)


if __name__ == "__main__":
    unittest.main()
