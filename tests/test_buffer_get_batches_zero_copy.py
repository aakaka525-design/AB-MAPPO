import unittest

import numpy as np
import torch

from buffer import RolloutBuffer


class TestRolloutBufferZeroCopy(unittest.TestCase):
    def test_get_batches_full_uses_shared_memory(self):
        buf = RolloutBuffer(num_agents=2, obs_dim=3, action_dim=2, buffer_size=4, state_dim=5)

        for t in range(3):
            obs = np.full((2, 3), t, dtype=np.float32)
            act = np.full((2, 2), t + 1, dtype=np.float32)
            logp = np.full((2,), -0.1 * (t + 1), dtype=np.float32)
            rew = np.full((2,), 0.5 * (t + 1), dtype=np.float32)
            val = np.full((2,), 0.3 * (t + 1), dtype=np.float32)
            state = np.full((5,), t + 2, dtype=np.float32)
            buf.add(obs, act, logp, rew, val, done=0.0, state=state)

        batch = buf.get_batches()
        self.assertTrue(np.shares_memory(batch["observations"].numpy(), buf.observations[:3]))
        self.assertTrue(np.shares_memory(batch["states"].numpy(), buf.states[:3]))

    def test_get_batches_subsample_keeps_shape_and_dtype(self):
        buf = RolloutBuffer(num_agents=3, obs_dim=4, action_dim=2, buffer_size=6)
        for t in range(6):
            obs = np.full((3, 4), t, dtype=np.float32)
            act = np.full((3, 2), t + 1, dtype=np.float32)
            logp = np.zeros((3,), dtype=np.float32)
            rew = np.ones((3,), dtype=np.float32)
            val = np.zeros((3,), dtype=np.float32)
            buf.add(obs, act, logp, rew, val, done=0.0)

        batch = buf.get_batches(batch_size=2)
        self.assertEqual(batch["observations"].shape, (2, 3, 4))
        self.assertEqual(batch["actions"].shape, (2, 3, 2))
        self.assertEqual(batch["returns"].dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
