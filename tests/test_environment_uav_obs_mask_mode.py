import unittest

import numpy as np

from environment import UAVMECEnv


class TestEnvironmentUAVObsMaskMode(unittest.TestCase):
    def test_prev_assoc_masks_non_associated_mu_features(self):
        env = UAVMECEnv(num_mus=4, num_uavs=2, seed=3, uav_obs_mask_mode="prev_assoc")
        env.reset()
        env.prev_association[:] = np.array([1, 2, 1, 0], dtype=np.int64)
        obs = env._get_observations()
        per_mu = obs["uav_obs"][:, 2 : 2 + env.K * 5].reshape(env.M, env.K, 5)

        # UAV-1 sees MU-0 and MU-2, others should be zeroed.
        self.assertTrue(np.allclose(per_mu[0, 1], 0.0))
        self.assertTrue(np.allclose(per_mu[0, 3], 0.0))
        self.assertFalse(np.allclose(per_mu[0, 0], 0.0))
        self.assertFalse(np.allclose(per_mu[0, 2], 0.0))

        # UAV-2 sees MU-1 only.
        self.assertFalse(np.allclose(per_mu[1, 1], 0.0))
        self.assertTrue(np.allclose(per_mu[1, 0], 0.0))
        self.assertTrue(np.allclose(per_mu[1, 2], 0.0))
        self.assertTrue(np.allclose(per_mu[1, 3], 0.0))

    def test_none_mode_keeps_full_mu_features(self):
        env = UAVMECEnv(num_mus=4, num_uavs=2, seed=4, uav_obs_mask_mode="none")
        env.reset()
        env.prev_association[:] = np.array([1, 2, 1, 0], dtype=np.int64)
        obs = env._get_observations()
        per_mu = obs["uav_obs"][:, 2 : 2 + env.K * 5].reshape(env.M, env.K, 5)
        # In full mode, no MU slot is hard-zeroed by association mask.
        self.assertFalse(np.allclose(per_mu[0, 1], 0.0))
        self.assertFalse(np.allclose(per_mu[1, 0], 0.0))


if __name__ == "__main__":
    unittest.main()
