import unittest
from unittest.mock import patch

from environment import UAVMECEnv
from maddpg import MADDPG
from mappo import ABMAPPO


class TestCheckpointLoadCompat(unittest.TestCase):
    def test_mappo_load_prefers_weights_only_and_fallbacks_on_type_error(self):
        env = UAVMECEnv(num_mus=2, num_uavs=1)
        agent = ABMAPPO(env, algorithm="AB-MAPPO", device="cpu")
        ckpt = {
            "mu_actor": agent.mu_actor.state_dict(),
            "uav_actor": agent.uav_actor.state_dict(),
            "critic": agent.critic.state_dict(),
            "algorithm": agent.algorithm,
        }

        with patch("mappo.torch.load", side_effect=[TypeError("unsupported arg"), ckpt]) as mock_load:
            agent.load("dummy.pt")

        self.assertEqual(mock_load.call_count, 2)
        self.assertTrue(mock_load.call_args_list[0].kwargs.get("weights_only", False))
        self.assertNotIn("weights_only", mock_load.call_args_list[1].kwargs)

    def test_maddpg_load_prefers_weights_only_and_fallbacks_on_type_error(self):
        env = UAVMECEnv(num_mus=2, num_uavs=1)
        agent = MADDPG(env, device="cpu")
        ckpt = {
            "mu_actor": agent.mu_actor.state_dict(),
            "uav_actor": agent.uav_actor.state_dict(),
            "critic": agent.critic.state_dict(),
            "noise_std": 0.123,
        }

        with patch("maddpg.torch.load", side_effect=[TypeError("unsupported arg"), ckpt]) as mock_load:
            agent.load("dummy.pt")

        self.assertEqual(mock_load.call_count, 2)
        self.assertTrue(mock_load.call_args_list[0].kwargs.get("weights_only", False))
        self.assertNotIn("weights_only", mock_load.call_args_list[1].kwargs)


if __name__ == "__main__":
    unittest.main()
