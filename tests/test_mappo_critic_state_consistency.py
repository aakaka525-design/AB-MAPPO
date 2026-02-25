import unittest
from unittest.mock import patch

import numpy as np
import torch

import config as cfg
from environment import UAVMECEnv
from mappo import ABMAPPO


class TestMAPPOCriticStateConsistency(unittest.TestCase):
    def test_mlp_critic_uses_buffered_state(self):
        with patch.object(cfg, "EPISODE_LENGTH", 4), patch.object(cfg, "PPO_EPOCHS", 1):
            env = UAVMECEnv(num_mus=3, num_uavs=2)
            agent = ABMAPPO(env, algorithm="B-MAPPO", device="cpu")

            agent.collect_episode()
            batches = agent.buffer.get_batches()

            self.assertIn("states", batches["mu"])
            self.assertIn("states", batches["uav"])
            self.assertEqual(batches["mu"]["states"].shape[1], env.state_dim)
            self.assertEqual(batches["uav"]["states"].shape[1], env.state_dim)
            self.assertTrue(torch.allclose(batches["mu"]["states"], batches["uav"]["states"]))

            update_info = agent.update()
            self.assertIn("critic_loss", update_info)
            self.assertTrue(np.isfinite(update_info["critic_loss"]))


if __name__ == "__main__":
    unittest.main()
