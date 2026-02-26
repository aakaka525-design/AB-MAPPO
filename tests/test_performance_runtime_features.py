import unittest

import numpy as np

from buffer import RolloutBuffer
from environment import UAVMECEnv
from mappo import ABMAPPO


class TestPerformanceRuntimeFeatures(unittest.TestCase):
    def test_environment_has_preallocated_step_caches(self):
        env = UAVMECEnv(num_mus=6, num_uavs=3, seed=5)
        self.assertTrue(hasattr(env, "_bw_alloc_cache"))
        self.assertTrue(hasattr(env, "_cpu_alloc_cache"))
        self.assertTrue(hasattr(env, "_edge_load_cache"))
        self.assertTrue(hasattr(env, "_t_loc_cache"))
        self.assertTrue(hasattr(env, "_e_loc_cache"))
        self.assertTrue(hasattr(env, "_t_edge_cache"))
        self.assertTrue(hasattr(env, "_uav_comp_cache"))

    def test_environment_prev_edge_load_not_aliased_to_step_cache(self):
        env = UAVMECEnv(num_mus=6, num_uavs=3, seed=6)
        env.reset()
        mu_actions = {"association": np.zeros((env.K,), dtype=np.int64), "offload_ratio": np.zeros((env.K,), dtype=np.float32)}
        uav_actions = np.full((env.M, env.uav_continuous_dim), 0.5, dtype=np.float32)
        env.step(mu_actions, uav_actions)
        self.assertIsNot(env.prev_edge_load, env._edge_load_cache)

    def test_mappo_has_amp_scaler(self):
        env = UAVMECEnv(num_mus=4, num_uavs=2, seed=7)
        agent = ABMAPPO(env, algorithm="AB-MAPPO", device="cpu")
        self.assertTrue(hasattr(agent, "scaler"))
        self.assertFalse(agent.scaler.is_enabled())

    def test_rollout_buffer_has_fast_gae_path(self):
        buf = RolloutBuffer(num_agents=2, obs_dim=3, action_dim=1, buffer_size=4)
        self.assertTrue(hasattr(buf, "_compute_gae_impl"))


if __name__ == "__main__":
    unittest.main()
