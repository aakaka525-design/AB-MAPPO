import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import train


class _FakeAgent:
    def __init__(self):
        self.saved = []

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

    def save(self, path):
        self.saved.append(path)


class _FakeAgentCollectedSteps(_FakeAgent):
    def __init__(self, collected_steps):
        super().__init__()
        self.collected_steps = int(collected_steps)
        self.collect_calls = 0

    def collect_episode(self):
        self.collect_calls += 1
        payload = super().collect_episode()
        payload["collected_steps"] = self.collected_steps
        return payload


class TestTrainCliPaperFlags(unittest.TestCase):
    def test_parse_args_supports_paper_flags(self):
        with patch(
            "sys.argv",
            [
                "train.py",
                "--paper_mode",
                "on",
                "--normalize_reward",
                "off",
                "--reward_scale",
                "1.0",
                "--uav_obs_mask_mode",
                "prev_assoc",
                "--rollout_mode",
                "env_episode",
            ],
        ):
            args = train.parse_args()
        self.assertEqual(args.paper_mode, "on")
        self.assertEqual(args.normalize_reward, "off")
        self.assertAlmostEqual(args.reward_scale, 1.0)
        self.assertEqual(args.uav_obs_mask_mode, "prev_assoc")
        self.assertEqual(args.rollout_mode, "env_episode")

    def test_train_paper_mode_forces_preset_flags(self):
        captured = {}

        def _fake_env_ctor(*args, **kwargs):
            captured["env_kwargs"] = kwargs
            return SimpleNamespace()

        def _fake_make_agent(env, algorithm, device, normalize_reward=True, rollout_mode="fixed"):
            captured["agent_kwargs"] = {
                "algorithm": algorithm,
                "device": device,
                "normalize_reward": normalize_reward,
                "rollout_mode": rollout_mode,
            }
            return _FakeAgent()

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            args = train.namespace_from_kwargs(
                algorithm="AB-MAPPO",
                total_steps=300,
                episode_length=300,
                run_dir=str(run_dir),
                disable_tensorboard=True,
                paper_mode="on",
                normalize_reward="on",
                reward_scale=10.0,
                uav_obs_mask_mode="none",
                rollout_mode="fixed",
                bs_relay_policy="nearest",
            )
            with patch("train.UAVMECEnv", side_effect=_fake_env_ctor), patch(
                "train._make_agent", side_effect=_fake_make_agent
            ):
                train.train(args)

            self.assertEqual(captured["env_kwargs"]["uav_obs_mask_mode"], "prev_assoc")
            self.assertEqual(captured["env_kwargs"]["bs_relay_policy"], "best_snr")
            self.assertFalse(captured["agent_kwargs"]["normalize_reward"])
            self.assertEqual(captured["agent_kwargs"]["rollout_mode"], "env_episode")
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["paper_mode"])
            self.assertEqual(summary["uav_obs_mask_mode"], "prev_assoc")
            self.assertFalse(summary["normalize_reward"])
            self.assertAlmostEqual(summary["reward_scale"], 1.0)
            self.assertEqual(summary["rollout_mode"], "env_episode")
            self.assertEqual(summary["bs_relay_policy"], "best_snr")

    def test_train_passes_flags_and_writes_summary(self):
        captured = {}

        def _fake_env_ctor(*args, **kwargs):
            captured["env_kwargs"] = kwargs
            return SimpleNamespace()

        def _fake_make_agent(env, algorithm, device, normalize_reward=True, rollout_mode="fixed"):
            captured["agent_kwargs"] = {
                "algorithm": algorithm,
                "device": device,
                "normalize_reward": normalize_reward,
                "rollout_mode": rollout_mode,
            }
            return _FakeAgent()

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            args = train.namespace_from_kwargs(
                algorithm="AB-MAPPO",
                total_steps=300,
                episode_length=300,
                run_dir=str(run_dir),
                disable_tensorboard=True,
                normalize_reward="off",
                reward_scale=1.0,
                uav_obs_mask_mode="prev_assoc",
                rollout_mode="env_episode",
            )
            with patch("train.UAVMECEnv", side_effect=_fake_env_ctor), patch(
                "train._make_agent", side_effect=_fake_make_agent
            ):
                train.train(args)

            self.assertEqual(captured["env_kwargs"]["uav_obs_mask_mode"], "prev_assoc")
            self.assertFalse(captured["agent_kwargs"]["normalize_reward"])
            self.assertEqual(captured["agent_kwargs"]["rollout_mode"], "env_episode")
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["uav_obs_mask_mode"], "prev_assoc")
            self.assertFalse(summary["normalize_reward"])
            self.assertAlmostEqual(summary["reward_scale"], 1.0)
            self.assertEqual(summary["rollout_mode"], "env_episode")

    def test_train_uses_collected_steps_for_progress(self):
        captured = {}
        fake_agent = _FakeAgentCollectedSteps(collected_steps=60)

        def _fake_env_ctor(*args, **kwargs):
            captured["env_kwargs"] = kwargs
            return SimpleNamespace(max_steps=60)

        def _fake_make_agent(env, algorithm, device, normalize_reward=True, rollout_mode="fixed"):
            captured["agent_kwargs"] = {
                "algorithm": algorithm,
                "device": device,
                "normalize_reward": normalize_reward,
                "rollout_mode": rollout_mode,
            }
            return fake_agent

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            args = train.namespace_from_kwargs(
                algorithm="AB-MAPPO",
                total_steps=300,
                episode_length=300,
                run_dir=str(run_dir),
                disable_tensorboard=True,
            )
            with patch("train.UAVMECEnv", side_effect=_fake_env_ctor), patch(
                "train._make_agent", side_effect=_fake_make_agent
            ):
                history = train.train(args)

            self.assertEqual(fake_agent.collect_calls, 5)
            self.assertEqual(int(history["step"][-1]), 300)
            self.assertEqual(len(history["episode"]), 5)
            saved = np.load(run_dir / "history.npz")
            self.assertEqual(int(saved["step"][-1]), 300)


if __name__ == "__main__":
    unittest.main()
