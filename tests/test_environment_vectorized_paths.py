import unittest

import numpy as np

import config as cfg
from environment import UAVMECEnv


class TestEnvironmentVectorizedPaths(unittest.TestCase):
    def test_bincount_matches_add_at_reference(self):
        rng = np.random.default_rng(20260226)
        for _ in range(20):
            m = 6
            size = 80
            uav_idx = rng.integers(0, m, size=size)
            e_edge = rng.random(size).astype(np.float32)

            ref = np.zeros((m,), dtype=np.float32)
            np.add.at(ref, uav_idx, e_edge)

            fast = np.bincount(uav_idx, weights=e_edge, minlength=m).astype(np.float32)
            self.assertTrue(np.allclose(ref, fast, atol=1e-6))

    def test_other_uav_vectorized_layout(self):
        env = UAVMECEnv(num_mus=3, num_uavs=4, seed=123)
        obs = env.reset()

        other_start = 2 + env.K * 5
        other_uav_flat = obs["uav_obs"][:, other_start:]
        self.assertEqual(other_uav_flat.shape, (env.M, (env.M - 1) * 2))

        w = max(env.area_width, 1.0)
        h = max(env.area_height, 1.0)
        uav_pos_norm = np.column_stack([env.uav_positions[:, 0] / w, env.uav_positions[:, 1] / h]).astype(np.float32)

        for m in range(env.M):
            got = other_uav_flat[m].reshape(env.M - 1, 2)
            expect = np.delete(uav_pos_norm, m, axis=0)
            self.assertTrue(np.allclose(got, expect, atol=1e-6))

    def test_cooperative_penalty_vector_matches_loop_reference(self):
        rng = np.random.default_rng(20260227)
        m = 7
        k = 60
        association = rng.integers(0, m + 1, size=k)
        valid_assoc = association > 0
        assoc_counts = np.bincount(association, minlength=m + 1)[1:]
        n_assoc_per_uav = np.maximum(assoc_counts, 1.0).astype(np.float32)
        mu_energy = rng.random(k).astype(np.float32)
        latency = rng.random(k).astype(np.float32)

        loop_penalty = np.zeros((m,), dtype=np.float32)
        for u in range(m):
            assoc_mus = np.where(association == (u + 1))[0]
            if assoc_mus.size > 0:
                loop_penalty[u] = float(np.mean(mu_energy[assoc_mus])) + float(np.mean(latency[assoc_mus]))

        vec_penalty = np.zeros((m,), dtype=np.float32)
        if np.any(valid_assoc):
            assoc_uav = association[valid_assoc] - 1
            mu_sum = np.bincount(assoc_uav, weights=mu_energy[valid_assoc], minlength=m)
            lat_sum = np.bincount(assoc_uav, weights=latency[valid_assoc], minlength=m)
            vec_penalty = ((mu_sum + lat_sum) / n_assoc_per_uav).astype(np.float32)

        self.assertTrue(np.allclose(loop_penalty, vec_penalty, atol=1e-6))

    def test_get_state_matches_reference_formula(self):
        env = UAVMECEnv(num_mus=5, num_uavs=3, seed=99)
        env.reset()

        state = env.get_state()
        w = max(env.area_width, 1.0)
        h = max(env.area_height, 1.0)
        reference = np.concatenate(
            [
                env.mu_positions[:, 0] / w,
                env.mu_positions[:, 1] / h,
                env.uav_positions[:, 0] / w,
                env.uav_positions[:, 1] / h,
                env.uav_velocities.flatten() / max(cfg.UAV_MAX_VELOCITY, 1e-8),
                env.task_data / max(cfg.TASK_DATA_MAX, 1e-8),
                env.task_cpu / max(cfg.TASK_CPU_CYCLES_MAX, 1e-8),
            ]
        ).astype(np.float32)

        self.assertEqual(state.dtype, np.float32)
        self.assertTrue(np.allclose(state, reference, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
