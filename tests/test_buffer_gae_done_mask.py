import unittest

import numpy as np

from buffer import RolloutBuffer


class TestGAEDoneMask(unittest.TestCase):
    def test_done_step_advantage_not_bootstrapped_from_next_value(self):
        buf = RolloutBuffer(
            num_agents=1,
            obs_dim=1,
            action_dim=1,
            buffer_size=3,
            gamma=1.0,
            gae_lambda=1.0,
        )

        buf.add(
            obs=np.array([[0.0]], dtype=np.float32),
            action=np.array([[0.0]], dtype=np.float32),
            log_prob=np.array([0.0], dtype=np.float32),
            reward=np.array([0.0], dtype=np.float32),
            value=np.array([5.0], dtype=np.float32),
            done=0.0,
        )
        buf.add(
            obs=np.array([[0.0]], dtype=np.float32),
            action=np.array([[0.0]], dtype=np.float32),
            log_prob=np.array([0.0], dtype=np.float32),
            reward=np.array([10.0], dtype=np.float32),
            value=np.array([1.0], dtype=np.float32),
            done=1.0,
        )
        buf.add(
            obs=np.array([[0.0]], dtype=np.float32),
            action=np.array([[0.0]], dtype=np.float32),
            log_prob=np.array([0.0], dtype=np.float32),
            reward=np.array([100.0], dtype=np.float32),
            value=np.array([50.0], dtype=np.float32),
            done=0.0,
        )

        buf.compute_gae(last_values=np.array([7.0], dtype=np.float32))

        # At done step t=1: A_t = r_t - V_t (no bootstrap from V_{t+1}).
        self.assertAlmostEqual(float(buf.advantages[1, 0]), 9.0, places=6)


if __name__ == "__main__":
    unittest.main()
