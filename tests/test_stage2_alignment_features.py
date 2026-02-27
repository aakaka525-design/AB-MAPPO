import unittest
from unittest.mock import patch

import numpy as np
import torch

import config as cfg
from environment import UAVMECEnv
from maddpg import MADDPG
from mappo import ABMAPPO


class TestStage2AlignmentFeatures(unittest.TestCase):
    def test_mu_discrete_dim_includes_bs_relay_choice(self):
        env = UAVMECEnv(num_mus=4, num_uavs=2, seed=9)
        # 0=local, 1..M=UAV, M+1=BS relay
        self.assertEqual(env.mu_discrete_dim, env.M + 2)

    def test_bs_relay_path_calls_uav_bs_rate(self):
        env = UAVMECEnv(num_mus=4, num_uavs=2, seed=10)
        env.reset()
        bs_label = env.M + 1
        association = np.array([bs_label, 1, 0, bs_label], dtype=np.int64)
        offload = np.array([0.8, 0.7, 0.0, 0.6], dtype=np.float32)
        uav_actions = np.full((env.M, env.uav_continuous_dim), 0.5, dtype=np.float32)

        with patch("environment.compute_uav_bs_rate", return_value=np.full((2,), 2e6, dtype=np.float32)) as mock_bs:
            obs, rewards, done, info = env.step({"association": association, "offload_ratio": offload}, uav_actions)
        self.assertTrue(mock_bs.called)
        self.assertEqual(obs["mu_obs"].shape[0], env.K)
        self.assertEqual(rewards["mu_rewards"].shape[0], env.K)
        self.assertTrue(np.isfinite(info["weighted_energy_mu_avg"]))
        self.assertIsInstance(bool(done), bool)

    def test_mlp_critic_outputs_per_agent_values(self):
        env = UAVMECEnv(num_mus=3, num_uavs=2, seed=11)
        agent = ABMAPPO(env, algorithm="B-MAPPO", device="cpu")
        obs = env.reset()
        mu_values, uav_values = agent.get_values(obs)
        self.assertEqual(mu_values.shape[0], env.K)
        self.assertEqual(uav_values.shape[0], env.M)
        all_values = np.concatenate([mu_values, uav_values], axis=0)
        self.assertGreater(float(np.std(all_values)), 0.0)
        state_t = torch.as_tensor(env.get_state(), dtype=torch.float32).unsqueeze(0)
        critic_out = agent.critic(state_t)
        self.assertEqual(tuple(critic_out.shape), (1, env.K + env.M))

    def test_env_episode_rollout_mode_collects_single_env_horizon(self):
        env = UAVMECEnv(num_mus=4, num_uavs=2, seed=12)
        agent = ABMAPPO(env, algorithm="AB-MAPPO", device="cpu", rollout_mode="env_episode")
        agent.collect_episode()
        self.assertEqual(agent.buffer.size, env.max_steps)

    def test_maddpg_assoc_dim_matches_env_discrete_space(self):
        env = UAVMECEnv(num_mus=4, num_uavs=2, seed=13)
        agent = MADDPG(env, device="cpu")
        self.assertEqual(agent.mu_assoc_dim, env.mu_discrete_dim)

    def test_bs_relay_bandwidth_shares_uav_allocator_pool(self):
        env = UAVMECEnv(num_mus=2, num_uavs=1, seed=14)
        env.reset()
        bs_label = env.M + 1
        association = np.array([1, bs_label], dtype=np.int64)
        offload = np.array([0.9, 0.9], dtype=np.float32)
        uav_actions = np.full((env.M, env.uav_continuous_dim), 0.5, dtype=np.float32)
        # raw_bw: direct MU gets larger weight, relay MU gets smaller weight
        uav_actions[0, 2] = 1.0
        uav_actions[0, 3] = 0.0

        captured_bw = []

        def _fake_rate(mu_pos, uav_pos, bw):
            captured_bw.append(np.asarray(bw, dtype=np.float32).copy())
            return np.full((len(bw),), 2e6, dtype=np.float32)

        with patch("environment.compute_mu_uav_rate_batch", side_effect=_fake_rate), patch(
            "environment.compute_uav_bs_rate", return_value=np.full((1,), 2e6, dtype=np.float32)
        ):
            env.step({"association": association, "offload_ratio": offload}, uav_actions)

        self.assertGreaterEqual(len(captured_bw), 2)
        direct_bw = float(captured_bw[0][0])
        relay_bw = float(captured_bw[1][0])
        self.assertGreater(direct_bw, relay_bw)
        self.assertLess(relay_bw, cfg.BANDWIDTH * 0.5)

    def test_bandwidth_softmax_temperature_zero_gives_uniform_weights(self):
        env = UAVMECEnv(num_mus=2, num_uavs=1, seed=15)
        env.reset()
        association = np.array([1, 1], dtype=np.int64)
        offload = np.array([0.9, 0.9], dtype=np.float32)
        uav_actions = np.full((env.M, env.uav_continuous_dim), 0.5, dtype=np.float32)
        uav_actions[0, 2] = 1.0
        uav_actions[0, 3] = 0.0

        captured_bw = []

        def _fake_rate(mu_pos, uav_pos, bw):
            captured_bw.append(np.asarray(bw, dtype=np.float32).copy())
            return np.full((len(bw),), 2e6, dtype=np.float32)

        with patch.object(cfg, "RESOURCE_SOFTMAX_TEMPERATURE", 0.0), patch(
            "environment.compute_mu_uav_rate_batch", side_effect=_fake_rate
        ):
            env.step({"association": association, "offload_ratio": offload}, uav_actions)

        self.assertGreaterEqual(len(captured_bw), 1)
        bw = captured_bw[0]
        self.assertEqual(bw.shape[0], 2)
        self.assertAlmostEqual(float(bw[0]), float(bw[1]), places=3)
        self.assertAlmostEqual(float(bw.sum()), float(cfg.BANDWIDTH), places=2)


if __name__ == "__main__":
    unittest.main()
