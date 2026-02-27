import unittest
from unittest.mock import patch

import torch

import networks
from networks import AttentionCritic


class TestAttentionCriticMaskAndEncoders(unittest.TestCase):
    def test_attention_critic_has_separate_mu_and_uav_encoders(self):
        critic = AttentionCritic(
            obs_dim=32,
            num_agents=5,
            num_mus=3,
            num_uavs=2,
            mu_obs_dim=12,
            uav_obs_dim=20,
            num_heads=4,
        )
        self.assertTrue(hasattr(critic, "mu_encoder"))
        self.assertTrue(hasattr(critic, "uav_encoder"))

    def test_attention_mask_blocks_diagonal(self):
        critic = AttentionCritic(
            obs_dim=32,
            num_agents=4,
            num_mus=2,
            num_uavs=2,
            mu_obs_dim=12,
            uav_obs_dim=20,
            num_heads=4,
        )
        all_obs = torch.randn(2, 4, 32)
        captured = {}

        orig = networks.F.scaled_dot_product_attention

        def _spy(q, k, v, attn_mask=None, dropout_p=0.0):
            captured["mask"] = attn_mask
            return orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)

        with patch.object(networks.F, "scaled_dot_product_attention", side_effect=_spy):
            out = critic(all_obs)
        self.assertEqual(tuple(out.shape), (2, 4, 1))
        self.assertIn("mask", captured)
        self.assertIsNotNone(captured["mask"])
        mask = captured["mask"]
        self.assertEqual(tuple(mask.shape), (1, 1, 4, 4))
        diag = torch.diagonal(mask[0, 0], dim1=-2, dim2=-1)
        self.assertTrue(torch.isinf(diag).all().item())
        self.assertTrue((diag < 0).all().item())


if __name__ == "__main__":
    unittest.main()
