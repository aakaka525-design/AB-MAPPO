"""
UAV-MEC environment for AB-MAPPO paper reproduction.

Key updates:
- Two-dimensional boundary handling uses AREA_WIDTH and AREA_HEIGHT separately.
- Collision penalty sign is corrected (no accidental collision reward).
- Additional metrics are returned for Fig.3-13 plotting.
- Optional w/o DT observation noise mode for Fig.12.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

import config as cfg
from channel_model import compute_mu_uav_rate, compute_mu_uav_rate_batch, compute_uav_bs_rate


class UAVMECEnv:
    """Multi-agent UAV-assisted MEC environment."""

    def __init__(
        self,
        num_mus: int | None = None,
        num_uavs: int | None = None,
        dt_deviation_rate: float | None = None,
        wo_dt_noise_mode: bool = False,
        uav_obs_mask_mode: str = cfg.UAV_OBS_MASK_MODE,
        bs_relay_policy: str = cfg.BS_RELAY_POLICY,
        area_width: float | None = None,
        area_height: float | None = None,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.K = int(num_mus or cfg.NUM_MUS)
        self.M = int(num_uavs or cfg.NUM_UAVS)
        self.dt_deviation_rate = (
            float(dt_deviation_rate) if dt_deviation_rate is not None else cfg.DT_DEVIATION_RATE
        )
        self.wo_dt_noise_mode = bool(wo_dt_noise_mode)
        if uav_obs_mask_mode not in {"none", "prev_assoc"}:
            raise ValueError("uav_obs_mask_mode must be one of {'none', 'prev_assoc'}")
        self.uav_obs_mask_mode = str(uav_obs_mask_mode)
        if bs_relay_policy not in {"nearest", "best_snr", "min_load"}:
            raise ValueError("bs_relay_policy must be one of {'nearest', 'best_snr', 'min_load'}")
        self.bs_relay_policy = str(bs_relay_policy)

        self.area_width = float(area_width if area_width is not None else cfg.AREA_WIDTH)
        self.area_height = float(area_height if area_height is not None else cfg.AREA_HEIGHT)
        if rng is not None and seed is not None:
            raise ValueError("Provide either `seed` or `rng`, not both.")
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self._norm_xy = np.array(
            [max(self.area_width, 1.0), max(self.area_height, 1.0)],
            dtype=np.float32,
        )
        self._task_data_scale = np.float32(1.0 / max(cfg.TASK_DATA_MAX, 1e-8))
        self._task_cpu_scale = np.float32(1.0 / max(cfg.TASK_CPU_CYCLES_MAX, 1e-8))
        self._edge_load_scale = np.float32(1.0 / (cfg.TASK_DATA_MAX * cfg.TASK_CPU_CYCLES_MAX + 1e-8))
        self._dist_penalty_scale = np.float32(1.0 / max(self.area_width, self.area_height))
        self._uav_velocity_scale = np.float32(1.0 / max(cfg.UAV_MAX_VELOCITY, 1e-8))
        if self.M > 1:
            rows = np.arange(self.M)
            mask = rows[:, None] != rows[None, :]
            self._other_uav_col_idx = np.where(mask)[1]
        else:
            self._other_uav_col_idx = None

        # MU observation/action dimensions
        self.mu_obs_dim = 2 + self.M * 2 + 2 + self.M
        # association: 0=local, 1..M=direct UAV offload, M+1=UAV-relayed BS offload
        self.mu_discrete_dim = self.M + 2
        self.mu_continuous_dim = 1
        self.mu_action_dim = 2

        # UAV observation/action dimensions
        self.uav_obs_dim = 2 + self.K * 5 + (self.M - 1) * 2
        self.uav_continuous_dim = 2 + self.K * 2
        self.uav_action_dim = self.uav_continuous_dim

        self.num_agents = self.K + self.M
        self.time_step = 0
        self.max_steps = cfg.NUM_SLOTS

        # dynamic states
        self.mu_positions = np.zeros((self.K, 2), dtype=np.float32)
        self.mu_velocities = np.zeros((self.K,), dtype=np.float32)
        self.mu_directions = np.zeros((self.K,), dtype=np.float32)
        self.uav_positions = np.zeros((self.M, 2), dtype=np.float32)
        self.uav_velocities = np.zeros((self.M, 2), dtype=np.float32)
        self.task_data = np.zeros((self.K,), dtype=np.float32)
        self.task_cpu = np.zeros((self.K,), dtype=np.float32)
        self.prev_edge_load = np.zeros((self.M, self.K), dtype=np.float32)
        self.prev_offload_ratio = np.zeros((self.K,), dtype=np.float32)
        self.prev_association = np.zeros((self.K,), dtype=np.int64)
        self._boundary_violation = np.zeros((self.M,), dtype=np.float32)
        # Pre-allocated work buffers for step() hot path.
        self._bw_alloc_cache = np.zeros((self.K,), dtype=np.float32)
        self._cpu_alloc_cache = np.zeros((self.K,), dtype=np.float32)
        self._edge_load_cache = np.zeros((self.M, self.K), dtype=np.float32)
        self._t_loc_cache = np.zeros((self.K,), dtype=np.float32)
        self._e_loc_cache = np.zeros((self.K,), dtype=np.float32)
        self._mu_energy_cache = np.zeros((self.K,), dtype=np.float32)
        self._t_edge_cache = np.zeros((self.K,), dtype=np.float32)
        self._uav_comp_cache = np.zeros((self.M,), dtype=np.float32)
        self._uav_total_cache = np.zeros((self.M,), dtype=np.float32)
        self._latency_penalty_cache = np.zeros((self.K,), dtype=np.float32)
        self._mu_rewards_cache = np.zeros((self.K,), dtype=np.float32)
        self._uav_rewards_cache = np.zeros((self.M,), dtype=np.float32)

    def set_wo_dt_noise_mode(self, enabled: bool) -> None:
        self.wo_dt_noise_mode = bool(enabled)

    def reset(self) -> Dict:
        self.time_step = 0
        self.mu_positions = np.column_stack(
            [
                self.rng.uniform(0.0, self.area_width, size=self.K),
                self.rng.uniform(0.0, self.area_height, size=self.K),
            ]
        ).astype(np.float32)
        self.mu_velocities = np.full((self.K,), cfg.MU_MEAN_VELOCITY, dtype=np.float32)
        self.mu_directions = self.rng.uniform(-np.pi, np.pi, size=self.K).astype(np.float32)

        self.uav_positions = np.column_stack(
            [
                self.rng.uniform(self.area_width * 0.1, self.area_width * 0.9, size=self.M),
                self.rng.uniform(self.area_height * 0.1, self.area_height * 0.9, size=self.M),
            ]
        ).astype(np.float32)
        self.uav_velocities = np.zeros((self.M, 2), dtype=np.float32)
        self.prev_edge_load = np.zeros((self.M, self.K), dtype=np.float32)
        self.prev_offload_ratio = np.zeros((self.K,), dtype=np.float32)
        self.prev_association = np.zeros((self.K,), dtype=np.int64)
        self._boundary_violation = np.zeros((self.M,), dtype=np.float32)
        self._generate_tasks()
        return self._get_observations()

    def _generate_tasks(self) -> None:
        self.task_data = self.rng.uniform(cfg.TASK_DATA_MIN, cfg.TASK_DATA_MAX, size=self.K).astype(np.float32)
        self.task_cpu = self.rng.uniform(cfg.TASK_CPU_CYCLES_MIN, cfg.TASK_CPU_CYCLES_MAX, size=self.K).astype(
            np.float32
        )

    def _update_mu_mobility(self) -> None:
        noise_v = self.rng.normal(0.0, cfg.MU_VELOCITY_STD, size=self.K)
        noise_theta = self.rng.normal(0.0, cfg.MU_DIRECTION_STD, size=self.K)

        self.mu_velocities = (
            cfg.MU_MEMORY_FACTOR_V * self.mu_velocities
            + (1.0 - cfg.MU_MEMORY_FACTOR_V) * cfg.MU_MEAN_VELOCITY
            + np.sqrt(max(0.0, 1.0 - cfg.MU_MEMORY_FACTOR_V**2)) * noise_v
        ).astype(np.float32)
        self.mu_velocities = np.clip(self.mu_velocities, 0.0, 10.0)

        self.mu_directions = (
            cfg.MU_MEMORY_FACTOR_THETA * self.mu_directions
            + (1.0 - cfg.MU_MEMORY_FACTOR_THETA) * cfg.MU_MEAN_DIRECTION
            + np.sqrt(max(0.0, 1.0 - cfg.MU_MEMORY_FACTOR_THETA**2)) * noise_theta
        ).astype(np.float32)

        self.mu_positions[:, 0] += self.mu_velocities * np.cos(self.mu_directions) * cfg.TIME_SLOT
        self.mu_positions[:, 1] += self.mu_velocities * np.sin(self.mu_directions) * cfg.TIME_SLOT

        # reflect on x boundary
        below_x = self.mu_positions[:, 0] < 0.0
        above_x = self.mu_positions[:, 0] > self.area_width
        self.mu_positions[below_x, 0] *= -1.0
        self.mu_positions[above_x, 0] = 2 * self.area_width - self.mu_positions[above_x, 0]

        # reflect on y boundary
        below_y = self.mu_positions[:, 1] < 0.0
        above_y = self.mu_positions[:, 1] > self.area_height
        self.mu_positions[below_y, 1] *= -1.0
        self.mu_positions[above_y, 1] = 2 * self.area_height - self.mu_positions[above_y, 1]

        bounced_x = below_x | above_x
        bounced_y = below_y | above_y
        self.mu_directions[bounced_x] = np.pi - self.mu_directions[bounced_x]
        self.mu_directions[bounced_y] = -self.mu_directions[bounced_y]
        self.mu_positions[:, 0] = np.clip(self.mu_positions[:, 0], 0.0, self.area_width)
        self.mu_positions[:, 1] = np.clip(self.mu_positions[:, 1], 0.0, self.area_height)

    def _update_uav_positions(self, uav_velocity_actions: np.ndarray) -> None:
        target_v = (uav_velocity_actions * 2.0 - 1.0) * cfg.UAV_MAX_VELOCITY
        v_prev = self.uav_velocities.copy()
        dv = target_v - v_prev
        dv_norm = np.linalg.norm(dv, axis=1, keepdims=True)
        max_dv = cfg.UAV_MAX_ACCELERATION * cfg.TIME_SLOT
        scale = np.where(dv_norm > max_dv, max_dv / (dv_norm + 1e-8), 1.0)
        applied_dv = dv * scale
        applied_acc = applied_dv / max(cfg.TIME_SLOT, 1e-8)
        v_next = v_prev + applied_dv

        v_norm = np.linalg.norm(v_next, axis=1, keepdims=True)
        scale_v = np.where(v_norm > cfg.UAV_MAX_VELOCITY, cfg.UAV_MAX_VELOCITY / (v_norm + 1e-8), 1.0)
        self.uav_velocities = v_next * scale_v

        dt = cfg.TIME_SLOT
        raw_pos = self.uav_positions + v_prev * dt + 0.5 * applied_acc * (dt**2)
        clipped = raw_pos.copy()
        clipped[:, 0] = np.clip(clipped[:, 0], 0.0, self.area_width)
        clipped[:, 1] = np.clip(clipped[:, 1], 0.0, self.area_height)

        self._boundary_violation = np.linalg.norm(raw_pos - clipped, axis=1)
        self.uav_positions = clipped

        at_boundary = (self.uav_positions[:, 0] <= 0.0) | (self.uav_positions[:, 0] >= self.area_width)
        at_boundary |= (self.uav_positions[:, 1] <= 0.0) | (self.uav_positions[:, 1] >= self.area_height)
        self.uav_velocities[at_boundary] = 0.0

    def _compute_local_energy_delay(self, k: int, rho_k: float) -> Tuple[float, float]:
        l_k = float(self.task_data[k])
        c_k = float(self.task_cpu[k])
        y_loc = (1.0 - rho_k) * l_k * c_k
        if y_loc <= 1e-8:
            return 0.0, 0.0

        f_est = min(y_loc / cfg.TIME_SLOT, cfg.MU_MAX_CPU_FREQ)
        f_dev = f_est * self.dt_deviation_rate * self.rng.uniform(-1.0, 1.0)
        f_actual = max(f_est + f_dev, f_est * 0.1)
        t_loc = y_loc / max(f_actual, 1e3)
        e_loc = cfg.EFFECTIVE_CAPACITANCE * (f_actual**2) * y_loc
        return float(e_loc), float(t_loc)

    def _compute_edge_energy_delay(
        self, k: int, rho_k: float, uav_idx: int, bw_k: float, cpu_k: float
    ) -> Tuple[float, float, float, float]:
        l_k = float(self.task_data[k])
        c_k = float(self.task_cpu[k])
        if rho_k <= 1e-8 or uav_idx < 0:
            return 0.0, 0.0, 0.0, 0.0

        offload_data = rho_k * l_k
        rate = max(float(compute_mu_uav_rate(self.mu_positions[k], self.uav_positions[uav_idx], bw_k)), 1e3)
        t_off = offload_data / rate
        e_off = cfg.MU_TRANSMIT_POWER * t_off

        y_edge = rho_k * l_k * c_k
        f_dev = cpu_k * self.dt_deviation_rate * self.rng.uniform(-1.0, 1.0)
        f_actual = max(cpu_k + f_dev, cpu_k * 0.1)
        t_ecmp = y_edge / max(f_actual, 1e3)
        e_edge_uav = cfg.EFFECTIVE_CAPACITANCE * (f_actual**2) * y_edge
        t_edge_total = t_off + t_ecmp
        return float(e_off), float(t_edge_total), float(e_edge_uav), float(y_edge)

    def _compute_uav_flying_energy(self, m: int) -> float:
        v = float(np.linalg.norm(self.uav_velocities[m]))
        term1 = 0.5 * cfg.FUSELAGE_DRAG_RATIO * cfg.AIR_DENSITY * cfg.ROTOR_SOLIDITY * cfg.ROTOR_DISC_AREA * (v**3)
        term2 = cfg.BLADE_PROFILE_POWER * (1.0 + 3.0 * v**2 / (cfg.TIP_SPEED**2))
        v0 = cfg.MEAN_ROTOR_VELOCITY
        inner = max(np.sqrt(1.0 + v**4 / (4.0 * v0**4)) - v**2 / (2.0 * v0**2), 0.0)
        term3 = cfg.INDUCED_POWER * np.sqrt(inner)
        return float((term1 + term2 + term3) * cfg.TIME_SLOT)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        z = x - np.max(x)
        exp = np.exp(z)
        return exp / (np.sum(exp) + 1e-8)

    def _penalty_latency(self, t_loc: float, t_edge: float) -> float:
        p = (max(t_loc - cfg.TIME_SLOT, 0.0) + max(t_edge - cfg.TIME_SLOT, 0.0)) / cfg.TIME_SLOT
        return float(cfg.PENALTY_DELAY * p)

    def _penalty_boundary(self, m: int) -> float:
        return float(cfg.PENALTY_BOUNDARY * self._boundary_violation[m])

    def _penalty_distance(self, m: int, association: np.ndarray) -> float:
        assoc_mus = np.where(association == (m + 1))[0]
        return self._penalty_distance_from_assoc(m, assoc_mus)

    def _penalty_distance_from_assoc(self, m: int, assoc_mus: np.ndarray) -> float:
        if assoc_mus.size == 0:
            return 0.0
        center = np.mean(self.mu_positions[assoc_mus], axis=0)
        dist = float(np.linalg.norm(self.uav_positions[m] - center))
        return float(self._dist_penalty_scale * max(dist - cfg.D_TH, 0.0))

    def _collision_penalties(self) -> np.ndarray:
        # Pairwise UAV distance matrix; diagonal excluded from penalty accumulation.
        diff = self.uav_positions[:, None, :] - self.uav_positions[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, cfg.D_MIN + 1.0)
        overlap = np.maximum((cfg.D_MIN - dists) / cfg.D_MIN, 0.0)
        return (cfg.PENALTY_COLLISION * overlap.sum(axis=1)).astype(np.float32)

    def _penalty_collision(self, m: int) -> float:
        # Positive penalty value, caller subtracts it from reward.
        return float(self._collision_penalties()[m])

    def _apply_wo_dt_noise(self, observations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not self.wo_dt_noise_mode:
            return observations

        mu_obs = observations["mu_obs"].copy()
        uav_obs = observations["uav_obs"].copy()

        noise = lambda shape: self.rng.uniform(-0.5, 0.5, size=shape).astype(np.float32)

        # MU obs: [mu_pos(2), uav_pos(2M), task_data(1), task_cpu(1), prev_edge(M)]
        mu_pos_end = 2
        uav_pos_end = 2 + self.M * 2
        task_data_idx = uav_pos_end
        task_cpu_idx = uav_pos_end + 1
        mu_obs[:, :mu_pos_end] = np.clip(mu_obs[:, :mu_pos_end] + noise(mu_obs[:, :mu_pos_end].shape), 0.0, 1.0)
        mu_obs[:, mu_pos_end:uav_pos_end] = np.clip(
            mu_obs[:, mu_pos_end:uav_pos_end] + noise(mu_obs[:, mu_pos_end:uav_pos_end].shape), 0.0, 1.0
        )
        mu_obs[:, task_data_idx] = np.clip(mu_obs[:, task_data_idx] + noise((self.K,)), 0.0, 1.0)
        mu_obs[:, task_cpu_idx] = np.clip(mu_obs[:, task_cpu_idx] + noise((self.K,)), 0.0, 1.0)

        # UAV obs: [uav_pos(2), per MU (mu_pos(2), offload(1), task_data(1), task_cpu(1)), other_uav_pos]
        uav_obs[:, :2] = np.clip(uav_obs[:, :2] + noise((self.M, 2)), 0.0, 1.0)
        per_mu_start = 2
        per_mu_end = 2 + self.K * 5
        per_mu_block = uav_obs[:, per_mu_start:per_mu_end].reshape(self.M, self.K, 5)
        per_mu_block[:, :, :2] = np.clip(
            per_mu_block[:, :, :2] + noise((self.M, self.K, 2)),
            0.0,
            1.0,
        )
        per_mu_block[:, :, 4] = np.clip(
            per_mu_block[:, :, 4] + noise((self.M, self.K)),
            0.0,
            1.0,
        )
        uav_obs[:, per_mu_start:per_mu_end] = per_mu_block.reshape(self.M, self.K * 5)

        other_start = 2 + self.K * 5
        if other_start < self.uav_obs_dim:
            uav_obs[:, other_start:] = np.clip(
                uav_obs[:, other_start:] + noise(uav_obs[:, other_start:].shape),
                0.0,
                1.0,
            )

        return {"mu_obs": mu_obs, "uav_obs": uav_obs}

    def _get_observations(self) -> Dict[str, np.ndarray]:
        mu_pos_norm = self.mu_positions / self._norm_xy
        uav_pos_norm = self.uav_positions / self._norm_xy
        task_data_norm = self.task_data * self._task_data_scale
        task_cpu_norm = self.task_cpu * self._task_cpu_scale

        mu_obs = np.zeros((self.K, self.mu_obs_dim), dtype=np.float32)
        mu_obs[:, 0:2] = mu_pos_norm
        mu_obs[:, 2 : 2 + self.M * 2] = uav_pos_norm.reshape(1, -1)
        mu_obs[:, 2 + self.M * 2] = task_data_norm
        mu_obs[:, 3 + self.M * 2] = task_cpu_norm
        mu_obs[:, 4 + self.M * 2 :] = self.prev_edge_load.T * self._edge_load_scale

        uav_obs = np.zeros((self.M, self.uav_obs_dim), dtype=np.float32)
        uav_obs[:, 0:2] = uav_pos_norm

        per_mu_view = uav_obs[:, 2 : 2 + self.K * 5].reshape(self.M, self.K, 5)
        per_mu_view[:, :, 0:2] = mu_pos_norm[None, :, :]
        per_mu_view[:, :, 2] = self.prev_offload_ratio[None, :]
        per_mu_view[:, :, 3] = task_data_norm[None, :]
        per_mu_view[:, :, 4] = task_cpu_norm[None, :]
        if self.uav_obs_mask_mode == "prev_assoc":
            assoc_mask = self.prev_association[None, :] == (np.arange(self.M, dtype=np.int64)[:, None] + 1)
            per_mu_view *= assoc_mask[:, :, None].astype(np.float32)

        if self.M > 1:
            other_uav = uav_pos_norm[self._other_uav_col_idx].reshape(self.M, self.M - 1, 2).reshape(self.M, -1)
            uav_obs[:, 2 + self.K * 5 :] = other_uav

        return self._apply_wo_dt_noise({"mu_obs": mu_obs, "uav_obs": uav_obs})

    def _select_bs_relay_uav(self, bs_indices: np.ndarray, effective_uav_idx: np.ndarray) -> np.ndarray:
        """Select relay UAV index for BS-offloaded MUs according to configured policy."""
        if bs_indices.size == 0:
            return np.empty((0,), dtype=np.int64)

        bs_mu_pos = self.mu_positions[bs_indices]
        deltas = bs_mu_pos[:, None, :] - self.uav_positions[None, :, :]
        dists = np.sqrt(np.sum(deltas**2, axis=2)).astype(np.float32)
        if self.bs_relay_policy == "nearest":
            return np.argmin(dists, axis=1).astype(np.int64)
        if self.bs_relay_policy == "best_snr":
            # Use equal per-link bandwidth to score first-hop MU->UAV quality.
            bw = np.full((bs_indices.size,), cfg.BANDWIDTH, dtype=np.float32)
            score = compute_mu_uav_rate_batch(
                np.repeat(bs_mu_pos[:, None, :], self.M, axis=1).reshape(-1, 2),
                np.repeat(self.uav_positions[None, :, :], bs_indices.size, axis=0).reshape(-1, 2),
                np.repeat(bw[:, None], self.M, axis=1).reshape(-1),
            ).reshape(bs_indices.size, self.M)
            return np.argmax(score, axis=1).astype(np.int64)

        # min_load: greedily assign each BS relay MU to the currently least-loaded UAV.
        load = np.bincount(np.maximum(effective_uav_idx, -1) + 1, minlength=self.M + 1)[1:].astype(np.int64)
        chosen = np.empty((bs_indices.size,), dtype=np.int64)
        for i in range(bs_indices.size):
            min_load = np.min(load)
            candidates = np.where(load == min_load)[0]
            if candidates.size == 1:
                picked = int(candidates[0])
            else:
                picked = int(candidates[np.argmin(dists[i, candidates])])
            chosen[i] = picked
            load[picked] += 1
        return chosen

    def step(self, mu_actions: Dict, uav_actions: np.ndarray) -> Tuple[Dict, Dict, bool, Dict]:
        association = np.asarray(mu_actions["association"], dtype=np.int64).copy()
        offload_ratio = np.asarray(mu_actions["offload_ratio"], dtype=np.float32).copy()

        bs_label = self.M + 1
        association = np.clip(association, 0, bs_label)
        offload_ratio = np.clip(offload_ratio, 0.0, 1.0)

        local_mask = association == 0
        offload_ratio[local_mask] = 0.0
        bs_mask = association == bs_label

        effective_uav_idx = np.full((self.K,), -1, dtype=np.int64)
        direct_assoc = (association > 0) & (association <= self.M)
        effective_uav_idx[direct_assoc] = association[direct_assoc] - 1
        if np.any(bs_mask):
            bs_indices = np.where(bs_mask)[0]
            effective_uav_idx[bs_indices] = self._select_bs_relay_uav(bs_indices, effective_uav_idx)

        uav_actions = np.asarray(uav_actions, dtype=np.float32)
        if uav_actions.shape != (self.M, self.uav_continuous_dim):
            raise ValueError(
                f"uav_actions shape mismatch: expected {(self.M, self.uav_continuous_dim)}, got {uav_actions.shape}"
            )
        uav_actions = np.clip(uav_actions, 0.0, 1.0)

        # 1) UAV movement
        self._update_uav_positions(uav_actions[:, :2])

        # 2) Resource allocation
        raw_bw = uav_actions[:, 2 : 2 + self.K]
        raw_cpu = uav_actions[:, 2 + self.K : 2 + 2 * self.K]
        direct_assoc_mus_list = [np.where(association == (m + 1))[0] for m in range(self.M)]
        assoc_mus_list = [np.where(effective_uav_idx == m)[0] for m in range(self.M)]

        bandwidth_alloc = self._bw_alloc_cache
        bandwidth_alloc.fill(0.0)
        cpu_alloc = self._cpu_alloc_cache
        cpu_alloc.fill(0.0)
        edge_load = self._edge_load_cache
        edge_load.fill(0.0)

        for m in range(self.M):
            # OFDMA bandwidth is shared by all MU links associated with UAV m,
            # including BS-relay first-hop links.
            assoc_mus = assoc_mus_list[m]
            if len(assoc_mus) > 0:
                temp = float(cfg.RESOURCE_SOFTMAX_TEMPERATURE)
                if temp <= 0.0:
                    bw_weights = np.full((len(assoc_mus),), 1.0 / float(len(assoc_mus)), dtype=np.float32)
                else:
                    bw_weights = self._softmax(raw_bw[m, assoc_mus] * temp)
                bandwidth_alloc[assoc_mus] = bw_weights * cfg.BANDWIDTH

            # UAV CPU is only used by direct edge-compute tasks.
            direct_assoc_mus = direct_assoc_mus_list[m]
            if len(direct_assoc_mus) > 0:
                temp = float(cfg.RESOURCE_SOFTMAX_TEMPERATURE)
                if temp <= 0.0:
                    cpu_weights = np.full((len(direct_assoc_mus),), 1.0 / float(len(direct_assoc_mus)), dtype=np.float32)
                else:
                    cpu_weights = self._softmax(raw_cpu[m, direct_assoc_mus] * temp)
                cpu_alloc[direct_assoc_mus] = cpu_weights * cfg.UAV_MAX_CPU_FREQ

        # 3) Energy and latency
        l_all = self.task_data
        c_all = self.task_cpu

        y_loc = (1.0 - offload_ratio) * l_all * c_all
        f_est_loc = np.minimum(y_loc / cfg.TIME_SLOT, cfg.MU_MAX_CPU_FREQ)
        f_dev_loc = f_est_loc * self.dt_deviation_rate * self.rng.uniform(-1.0, 1.0, size=self.K)
        f_actual_loc = np.maximum(f_est_loc + f_dev_loc, f_est_loc * 0.1)

        valid_loc = y_loc > 1e-8
        t_loc_all = self._t_loc_cache
        t_loc_all.fill(0.0)
        t_loc_all[valid_loc] = (y_loc[valid_loc] / np.maximum(f_actual_loc[valid_loc], 1e3)).astype(np.float32)

        e_loc_all = self._e_loc_cache
        e_loc_all.fill(0.0)
        e_loc_all[valid_loc] = (cfg.EFFECTIVE_CAPACITANCE * (f_actual_loc[valid_loc] ** 2) * y_loc[valid_loc]).astype(
            np.float32
        )

        mu_energy = self._mu_energy_cache
        np.copyto(mu_energy, e_loc_all)
        t_edge_all = self._t_edge_cache
        t_edge_all.fill(0.0)
        # Non-flight UAV energy accumulator:
        # - direct branch: UAV computing energy
        # - relay branch: UAV forwarding transmission energy
        uav_comp_energy = self._uav_comp_cache
        uav_comp_energy.fill(0.0)
        offload_mask = (effective_uav_idx >= 0) & (offload_ratio > 1e-8)
        offload_indices = np.where(offload_mask)[0]
        if offload_indices.size > 0:
            offload_assoc = association[offload_indices]
            direct_mask = offload_assoc <= self.M
            if np.any(direct_mask):
                direct_indices = offload_indices[direct_mask]
                uav_idx = effective_uav_idx[direct_indices]
                bw = np.maximum(bandwidth_alloc[direct_indices], 1e3)
                cpu = np.maximum(cpu_alloc[direct_indices], 1e3)

                offload_data = offload_ratio[direct_indices] * l_all[direct_indices]
                rates = compute_mu_uav_rate_batch(
                    self.mu_positions[direct_indices],
                    self.uav_positions[uav_idx],
                    bw,
                ).astype(np.float32)
                rates = np.maximum(rates, 1e3)

                t_off = offload_data / rates
                e_off = cfg.MU_TRANSMIT_POWER * t_off

                y_edge = offload_ratio[direct_indices] * l_all[direct_indices] * c_all[direct_indices]
                f_dev_edge = cpu * self.dt_deviation_rate * self.rng.uniform(-1.0, 1.0, size=direct_indices.size)
                f_actual_edge = np.maximum(cpu + f_dev_edge, cpu * 0.1)
                t_ecmp = y_edge / np.maximum(f_actual_edge, 1e3)
                e_edge = cfg.EFFECTIVE_CAPACITANCE * (f_actual_edge**2) * y_edge

                mu_energy[direct_indices] += e_off.astype(np.float32)
                t_edge_all[direct_indices] = (t_off + t_ecmp).astype(np.float32)
                uav_comp_energy += np.bincount(uav_idx, weights=e_edge, minlength=self.M).astype(np.float32)
                edge_load[uav_idx, direct_indices] = y_edge.astype(np.float32)

            relay_mask = offload_assoc == bs_label
            if np.any(relay_mask):
                relay_indices = offload_indices[relay_mask]
                relay_uav = effective_uav_idx[relay_indices]
                bw_first_hop = np.maximum(bandwidth_alloc[relay_indices], 1e3)
                offload_data = offload_ratio[relay_indices] * l_all[relay_indices]

                rates_mu_uav = compute_mu_uav_rate_batch(
                    self.mu_positions[relay_indices],
                    self.uav_positions[relay_uav],
                    bw_first_hop,
                ).astype(np.float32)
                rates_mu_uav = np.maximum(rates_mu_uav, 1e3)
                t_mu_uav = offload_data / rates_mu_uav
                e_mu_uav = cfg.MU_TRANSMIT_POWER * t_mu_uav

                rates_uav_bs = np.maximum(compute_uav_bs_rate(self.uav_positions[relay_uav]).astype(np.float32), 1e3)
                t_uav_bs = offload_data / rates_uav_bs
                e_uav_relay_tx = cfg.UAV_TRANSMIT_POWER * t_uav_bs

                mu_energy[relay_indices] += e_mu_uav.astype(np.float32)
                # BS-side compute delay is assumed negligible in the paper setting.
                t_edge_all[relay_indices] = (t_mu_uav + t_uav_bs).astype(np.float32)
                uav_comp_energy += np.bincount(relay_uav, weights=e_uav_relay_tx, minlength=self.M).astype(np.float32)
                y_relay = offload_ratio[relay_indices] * l_all[relay_indices] * c_all[relay_indices]
                edge_load[relay_uav, relay_indices] = y_relay.astype(np.float32)

        v = np.linalg.norm(self.uav_velocities, axis=1)
        term1 = 0.5 * cfg.FUSELAGE_DRAG_RATIO * cfg.AIR_DENSITY * cfg.ROTOR_SOLIDITY * cfg.ROTOR_DISC_AREA * (v**3)
        term2 = cfg.BLADE_PROFILE_POWER * (1.0 + 3.0 * v**2 / (cfg.TIP_SPEED**2))
        v0 = cfg.MEAN_ROTOR_VELOCITY
        inner = np.maximum(np.sqrt(1.0 + v**4 / (4.0 * v0**4)) - v**2 / (2.0 * v0**2), 0.0)
        term3 = cfg.INDUCED_POWER * np.sqrt(inner)
        uav_fly_energy = ((term1 + term2 + term3) * cfg.TIME_SLOT).astype(np.float32)
        uav_total_energy = self._uav_total_cache
        np.add(uav_fly_energy, uav_comp_energy, out=uav_total_energy)

        # 4) Rewards
        valid_assoc = effective_uav_idx >= 0
        if np.any(valid_assoc):
            assoc_counts = np.bincount(effective_uav_idx[valid_assoc], minlength=self.M)
        else:
            assoc_counts = np.zeros((self.M,), dtype=np.int64)
        n_assoc_per_uav = np.maximum(assoc_counts, 1.0).astype(np.float32)

        latency_penalty_all = self._latency_penalty_cache
        latency_penalty_all[:] = (
            cfg.PENALTY_DELAY
            * (np.maximum(t_loc_all - cfg.TIME_SLOT, 0.0) + np.maximum(t_edge_all - cfg.TIME_SLOT, 0.0))
            / cfg.TIME_SLOT
        )

        mu_rewards = self._mu_rewards_cache
        mu_rewards[:] = -mu_energy - latency_penalty_all
        if np.any(valid_assoc):
            uav_indices = effective_uav_idx[valid_assoc]
            mu_rewards[valid_assoc] -= (
                cfg.WEIGHT_FACTOR * uav_total_energy[uav_indices] / n_assoc_per_uav[uav_indices]
            ).astype(np.float32)
        mu_rewards *= cfg.REWARD_SCALE

        uav_rewards = self._uav_rewards_cache
        uav_rewards[:] = -np.float32(cfg.UAV_REWARD_SELF_ENERGY_COEFF) * uav_total_energy
        collision_penalties = self._collision_penalties()
        boundary_penalties = (cfg.PENALTY_BOUNDARY * self._boundary_violation).astype(np.float32)
        if np.any(valid_assoc):
            assoc_uav = effective_uav_idx[valid_assoc]
            mu_energy_sum_per_uav = np.bincount(assoc_uav, weights=mu_energy[valid_assoc], minlength=self.M)
            latency_sum_per_uav = np.bincount(assoc_uav, weights=latency_penalty_all[valid_assoc], minlength=self.M)
            cooperative_penalty = (mu_energy_sum_per_uav + latency_sum_per_uav) / n_assoc_per_uav
            uav_rewards -= cooperative_penalty.astype(np.float32)
        uav_rewards -= boundary_penalties
        uav_rewards -= collision_penalties
        for m in range(self.M):
            uav_rewards[m] -= np.float32(self._penalty_distance_from_assoc(m, assoc_mus_list[m]))
        uav_rewards *= cfg.REWARD_SCALE

        # 5) State update
        self.prev_edge_load[:] = edge_load
        self.prev_offload_ratio[:] = offload_ratio
        self.prev_association[:] = np.where(effective_uav_idx >= 0, effective_uav_idx + 1, 0).astype(np.int64)
        self._update_mu_mobility()
        self._generate_tasks()
        self.time_step += 1

        done = self.time_step >= self.max_steps
        obs = self._get_observations()

        # 6) Metrics
        mu_energy_avg = float(np.mean(mu_energy))
        uav_energy_avg = float(np.mean(uav_total_energy))
        weighted_energy_mu_avg = mu_energy_avg + cfg.WEIGHT_FACTOR * uav_energy_avg
        weighted_energy_mu_total = float(np.sum(mu_energy) + cfg.WEIGHT_FACTOR * np.sum(uav_total_energy))

        # Paper-aligned fairness on MU energy costs.
        fairness_denom = float(self.K * np.sum(mu_energy**2) + 1e-8)
        jain_fairness = float((np.sum(mu_energy) ** 2) / fairness_denom) if fairness_denom > 0 else 1.0
        # Legacy utility-based fairness retained for backward-compatible analysis scripts.
        mu_utility = 1.0 / (mu_energy + 1e-8)
        utility_fairness_denom = float(self.K * np.sum(mu_utility**2) + 1e-8)
        jain_fairness_utility = (
            float((np.sum(mu_utility) ** 2) / utility_fairness_denom) if utility_fairness_denom > 0 else 1.0
        )

        max_delay = np.maximum(t_loc_all, t_edge_all)
        info = {
            "mu_energy": mu_energy.copy(),
            "uav_energy": uav_total_energy.copy(),
            "uav_fly_energy": uav_fly_energy.copy(),
            "uav_comp_energy": uav_comp_energy.copy(),
            "weighted_energy": weighted_energy_mu_avg,  # compatibility alias
            "weighted_energy_mu_avg": weighted_energy_mu_avg,
            "weighted_energy_mu_total": weighted_energy_mu_total,
            "mu_energy_avg": mu_energy_avg,
            "uav_energy_avg": uav_energy_avg,
            "jain_fairness": jain_fairness,
            "jain_fairness_utility": jain_fairness_utility,
            "avg_delay": float(np.mean(max_delay)),
            "delay_violation_rate": float(np.mean(max_delay > cfg.TIME_SLOT)),
        }

        rewards = {"mu_rewards": mu_rewards.copy(), "uav_rewards": uav_rewards.copy()}
        return obs, rewards, done, info

    def get_state(self) -> np.ndarray:
        mu_pos_norm = self.mu_positions / self._norm_xy
        uav_pos_norm = self.uav_positions / self._norm_xy
        state_parts = [
            mu_pos_norm[:, 0],
            mu_pos_norm[:, 1],
            uav_pos_norm[:, 0],
            uav_pos_norm[:, 1],
            self.uav_velocities.reshape(-1) * self._uav_velocity_scale,
            self.task_data * self._task_data_scale,
            self.task_cpu * self._task_cpu_scale,
        ]
        return np.concatenate(state_parts)

    @property
    def state_dim(self) -> int:
        # x/y for K MUs + x/y for M UAVs + velocity(2M) + task_data(K) + task_cpu(K)
        return self.K * 2 + self.M * 2 + self.M * 2 + self.K + self.K
