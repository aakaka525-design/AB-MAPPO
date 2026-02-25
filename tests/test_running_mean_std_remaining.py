import unittest

import numpy as np

from mappo import RunningMeanStd


class TestRunningMeanStdRemaining(unittest.TestCase):
    def test_vector_reward_batch_counts_as_one_step(self):
        rms = RunningMeanStd()
        before = float(rms.count)
        rms.update(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        self.assertAlmostEqual(float(rms.count), before + 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
