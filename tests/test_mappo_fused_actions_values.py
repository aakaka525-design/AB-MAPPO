import unittest
from unittest.mock import patch

import numpy as np

import config as cfg
from environment import UAVMECEnv
from mappo import ABMAPPO


class TestMAPPOFusedActionsValues(unittest.TestCase):
    def test_collect_episode_uses_fused_path_and_returns_finite_metrics(self):
        with patch.object(cfg, "EPISODE_LENGTH", 5):
            env = UAVMECEnv(num_mus=4, num_uavs=2, seed=7)
            agent = ABMAPPO(env, algorithm="AB-MAPPO", device="cpu")

            with patch.object(agent, "get_actions_and_values", wraps=agent.get_actions_and_values) as fused_mock:
                metrics = agent.collect_episode()

            self.assertEqual(fused_mock.call_count, cfg.EPISODE_LENGTH)
            expected_keys = [
                "mu_reward",
                "uav_reward",
                "total_cost",
                "weighted_energy",
                "weighted_energy_mu_avg",
                "weighted_energy_mu_total",
                "mu_energy",
                "uav_energy",
                "mu_energy_avg",
                "uav_energy_avg",
                "jain_fairness",
                "delay_violation",
            ]
            for key in expected_keys:
                self.assertIn(key, metrics)
                self.assertTrue(np.isfinite(metrics[key]))

    def test_collect_episode_state_without_copy_semantics(self):
        with patch.object(cfg, "EPISODE_LENGTH", 4):
            env = UAVMECEnv(num_mus=4, num_uavs=2, seed=11)
            agent = ABMAPPO(env, algorithm="B-MAPPO", device="cpu")

            with patch.object(env, "get_state", wraps=env.get_state) as get_state_mock:
                agent.collect_episode()

            # once per step for buffer store + once after rollout for last bootstrap value
            self.assertEqual(get_state_mock.call_count, cfg.EPISODE_LENGTH + 1)

    def test_update_uses_set_to_none_for_zero_grad(self):
        with patch.object(cfg, "EPISODE_LENGTH", 4), patch.object(cfg, "PPO_EPOCHS", 1):
            env = UAVMECEnv(num_mus=4, num_uavs=2, seed=13)
            agent = ABMAPPO(env, algorithm="B-MAPPO", device="cpu")
            agent.collect_episode()

            with patch.object(
                agent.mu_actor_optimizer,
                "zero_grad",
                wraps=agent.mu_actor_optimizer.zero_grad,
            ) as mu_zero, patch.object(
                agent.uav_actor_optimizer,
                "zero_grad",
                wraps=agent.uav_actor_optimizer.zero_grad,
            ) as uav_zero, patch.object(
                agent.critic_optimizer,
                "zero_grad",
                wraps=agent.critic_optimizer.zero_grad,
            ) as critic_zero:
                agent.update()

            for zero_mock in (mu_zero, uav_zero, critic_zero):
                self.assertGreaterEqual(zero_mock.call_count, 1)
                for call in zero_mock.call_args_list:
                    self.assertTrue(call.kwargs.get("set_to_none", False))

    def test_mlp_update_does_not_require_allclose_by_default(self):
        with patch.object(cfg, "EPISODE_LENGTH", 4), patch.object(cfg, "PPO_EPOCHS", 1):
            env = UAVMECEnv(num_mus=4, num_uavs=2, seed=17)
            agent = ABMAPPO(env, algorithm="B-MAPPO", device="cpu")
            agent.collect_episode()

            with patch("mappo.torch.allclose", side_effect=AssertionError("unexpected sync check")):
                update_info = agent.update()

            self.assertIn("critic_loss", update_info)
            self.assertTrue(np.isfinite(update_info["critic_loss"]))


if __name__ == "__main__":
    unittest.main()
