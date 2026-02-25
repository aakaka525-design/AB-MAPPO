import unittest

import torch

from networks import GaussianActor


class TestNetworkStability(unittest.TestCase):
    def test_gaussian_actor_evaluate_action_is_finite_near_bounds(self):
        actor = GaussianActor(obs_dim=8, continuous_dim=2, discrete_dim=0)
        obs = torch.randn(4, 8)
        actions = torch.tensor(
            [
                [1e-8, 1 - 1e-8],
                [1e-7, 1 - 1e-7],
                [1e-6, 1 - 1e-6],
                [0.5, 0.5],
            ],
            dtype=torch.float32,
        )
        log_prob, entropy = actor.evaluate_action(obs, actions)
        self.assertTrue(torch.isfinite(log_prob).all().item())
        self.assertTrue(torch.isfinite(entropy).all().item())


if __name__ == "__main__":
    unittest.main()
