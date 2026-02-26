import unittest

import numpy as np

from buffer import RolloutBuffer


class TestBufferOverflowGuard(unittest.TestCase):
    def test_add_raises_clear_error_when_overflow(self):
        buf = RolloutBuffer(num_agents=2, obs_dim=3, action_dim=2, buffer_size=2)
        obs = np.zeros((2, 3), dtype=np.float32)
        act = np.zeros((2, 2), dtype=np.float32)
        lp = np.zeros((2,), dtype=np.float32)
        rew = np.zeros((2,), dtype=np.float32)
        val = np.zeros((2,), dtype=np.float32)

        buf.add(obs, act, lp, rew, val, done=0.0)
        buf.add(obs, act, lp, rew, val, done=0.0)

        with self.assertRaises(RuntimeError) as cm:
            buf.add(obs, act, lp, rew, val, done=0.0)

        msg = str(cm.exception).lower()
        self.assertIn("overflow", msg)
        self.assertIn("reset", msg)


if __name__ == "__main__":
    unittest.main()
