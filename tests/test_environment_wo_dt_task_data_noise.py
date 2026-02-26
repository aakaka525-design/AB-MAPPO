import unittest

import numpy as np

from environment import UAVMECEnv


class TestEnvironmentWODTTaskDataNoise(unittest.TestCase):
    def test_task_data_field_has_noise_and_is_clipped(self):
        env_clean = UAVMECEnv(num_mus=8, num_uavs=3, wo_dt_noise_mode=False, seed=123)
        env_noise = UAVMECEnv(num_mus=8, num_uavs=3, wo_dt_noise_mode=True, seed=123)

        obs_clean = env_clean.reset()
        obs_noise = env_noise.reset()

        task_data_idx = 2 + env_noise.M * 2
        clean_col = obs_clean["mu_obs"][:, task_data_idx]
        noise_col = obs_noise["mu_obs"][:, task_data_idx]

        self.assertFalse(np.allclose(clean_col, noise_col))
        self.assertTrue(np.all(noise_col >= 0.0))
        self.assertTrue(np.all(noise_col <= 1.0))


if __name__ == "__main__":
    unittest.main()
