import unittest

from train_sweep import RunOptions, build_run_specs


class TestTrainSweepRunSpecs(unittest.TestCase):
    def test_full_run_spec_count_and_heavy_figs(self):
        options = RunOptions(
            seeds=[42, 43, 44],
            device="cpu",
            total_steps=80000,
            episode_length=300,
            resume=False,
            skip_existing=False,
            disable_tensorboard=True,
            smoke=False,
        )
        specs = build_run_specs("all", options)
        self.assertEqual(len(specs), 555)

        by_fig = {}
        for spec in specs:
            by_fig[spec["fig"]] = by_fig.get(spec["fig"], 0) + 1
            self.assertIn("cli_overrides", spec)
            self.assertIn("summary_path", spec)
            self.assertIn("run_dir", spec)

        self.assertEqual(by_fig["10"], 108)
        self.assertEqual(by_fig["11"], 72)


if __name__ == "__main__":
    unittest.main()
