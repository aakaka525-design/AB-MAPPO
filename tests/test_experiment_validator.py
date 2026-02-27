import json
import os
import tempfile
import unittest
from unittest import mock

from experiment_validator import validate_aggregate_freshness, validate_experiment_outputs, validate_run_summaries


class TestExperimentValidator(unittest.TestCase):
    def test_validate_run_summaries_reports_missing_summary(self):
        specs = [
            {
                "fig": "10",
                "algorithm": "AB-MAPPO",
                "setting_name": "K60_M10_muCPU1p0",
                "seed": 42,
                "summary_path": "/tmp/not_exists_summary.json",
                "num_mus": 60,
                "num_uavs": 10,
            }
        ]
        errors = validate_run_summaries(specs, total_steps=80000, episode_length=300)
        self.assertTrue(any("missing summary" in e.lower() for e in errors))

    def test_validate_run_summaries_reports_parameter_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            summary_path = os.path.join(td, "summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "algorithm": "AB-MAPPO",
                        "seed": 42,
                        "num_mus": 70,
                        "num_uavs": 10,
                        "total_steps": 80000,
                        "episode_length": 300,
                    },
                    f,
                )
            specs = [
                {
                    "fig": "10",
                    "algorithm": "AB-MAPPO",
                    "setting_name": "K60_M10_muCPU1p0",
                    "seed": 42,
                    "summary_path": summary_path,
                    "num_mus": 60,
                    "num_uavs": 10,
                }
            ]
            errors = validate_run_summaries(specs, total_steps=80000, episode_length=300)
            self.assertTrue(any("num_mus" in e.lower() for e in errors))

    def test_validate_aggregate_freshness_reports_stale_aggregate(self):
        with tempfile.TemporaryDirectory() as td:
            summary_path = os.path.join(td, "summary.json")
            aggregate_path = os.path.join(td, "results.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump({"total_steps": 80000, "episode_length": 300}, f)
            with open(aggregate_path, "w", encoding="utf-8") as f:
                f.write("{}")

            now = int(os.path.getmtime(summary_path))
            os.utime(summary_path, (now + 5, now + 5))
            os.utime(aggregate_path, (now, now))

            specs = [{"fig": "10", "summary_path": summary_path}]
            aggregate_paths = {"10": aggregate_path}
            errors = validate_aggregate_freshness(specs, aggregate_paths)
            self.assertTrue(any("stale aggregate" in e.lower() for e in errors))

    @mock.patch("experiment_validator.validate_aggregate_freshness", return_value=[])
    @mock.patch("experiment_validator.build_run_specs")
    def test_validate_experiment_outputs_paper_mode_requires_profile_fields(self, mock_build_specs, _mock_freshness):
        with tempfile.TemporaryDirectory() as td:
            summary_path = os.path.join(td, "summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "algorithm": "AB-MAPPO",
                        "seed": 42,
                        "num_mus": 60,
                        "num_uavs": 10,
                        "total_steps": 80000,
                        "episode_length": 300,
                        "paper_mode": True,
                        "normalize_reward": False,
                        "reward_scale": 10.0,  # should be 1.0 in paper_mode profile
                        "uav_obs_mask_mode": "prev_assoc",
                        "rollout_mode": "env_episode",
                        "bs_relay_policy": "best_snr",
                    },
                    f,
                )

            mock_build_specs.return_value = [
                {
                    "fig": "10",
                    "algorithm": "AB-MAPPO",
                    "setting_name": "K60_M10_case",
                    "seed": 42,
                    "summary_path": summary_path,
                    "num_mus": 60,
                    "num_uavs": 10,
                }
            ]

            with self.assertRaises(RuntimeError) as ctx:
                validate_experiment_outputs(
                    total_steps=80000,
                    episode_length=300,
                    seeds=[42],
                    fig="all",
                    paper_mode=True,
                    experiment_root=td,
                )
            self.assertIn("reward_scale", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
