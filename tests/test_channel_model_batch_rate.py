import unittest

import numpy as np

from channel_model import compute_mu_uav_rate, compute_mu_uav_rate_batch


class TestChannelModelBatchRate(unittest.TestCase):
    def test_batch_rate_matches_scalar(self):
        mu_pos = np.array([[0.0, 0.0], [10.0, 5.0], [100.0, 80.0]], dtype=np.float32)
        uav_pos = np.array([[50.0, 20.0], [25.0, 30.0], [150.0, 110.0]], dtype=np.float32)
        bw = np.array([1e6, 2e6, 3e6], dtype=np.float32)

        batch = compute_mu_uav_rate_batch(mu_pos, uav_pos, bw)
        scalar = np.array([compute_mu_uav_rate(mu_pos[i], uav_pos[i], bw[i]) for i in range(3)], dtype=np.float64)
        np.testing.assert_allclose(batch, scalar, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
