"""
AB-MAPPO 论文复现 — 信道与通信模型
实现论文公式 (9)-(14): LoS概率信道、传输速率计算
"""

import numpy as np
import config as cfg

try:
    from numba import njit
except Exception:  # pragma: no cover - optional acceleration path
    njit = None


_UAV_HEIGHT = float(cfg.UAV_HEIGHT)
_ENV_A = float(cfg.ENV_PARAM_A)
_ENV_B = float(cfg.ENV_PARAM_B)
_CH_GAIN = float(cfg.CHANNEL_POWER_GAIN)
_NLOS_ATT = float(cfg.NLOS_ATTENUATION)
_PATH_LOSS = float(cfg.PATH_LOSS_EXP)
_MU_TX_PWR = float(cfg.MU_TRANSMIT_POWER)
_NOISE_DENS = float(cfg.NOISE_POWER_DENSITY)


if njit is not None:
    @njit(cache=True, fastmath=True)
    def _compute_mu_uav_rate_batch_jit(mu_pos, uav_pos, bandwidths):
        n = mu_pos.shape[0]
        dims = mu_pos.shape[1]
        rates = np.empty((n,), dtype=np.float32)
        for i in range(n):
            bw = bandwidths[i]
            if bw < 1e-3:
                bw = 1e-3

            dx = mu_pos[i, 0] - uav_pos[i, 0]
            dy = mu_pos[i, 1] - uav_pos[i, 1]
            horiz_dist_sq = dx * dx + dy * dy
            horiz_dist = np.sqrt(horiz_dist_sq)
            if horiz_dist < 1e-6:
                horiz_dist = 1e-6

            if dims == 2:
                dist_3d = np.sqrt(horiz_dist_sq + _UAV_HEIGHT * _UAV_HEIGHT)
            else:
                dz = mu_pos[i, 2] - uav_pos[i, 2]
                dist_3d = np.sqrt(horiz_dist_sq + dz * dz)
            if dist_3d < 1.0:
                dist_3d = 1.0

            elev_angle = np.arctan(_UAV_HEIGHT / horiz_dist) * (180.0 / np.pi)
            p_los = 1.0 / (1.0 + _ENV_A * np.exp(-_ENV_B * (elev_angle - _ENV_A)))
            gain = _CH_GAIN * (p_los + _NLOS_ATT * (1.0 - p_los)) / (dist_3d ** _PATH_LOSS)
            snr = _MU_TX_PWR * gain / (bw * _NOISE_DENS)
            rates[i] = bw * np.log2(1.0 + snr)
        return rates
else:
    _compute_mu_uav_rate_batch_jit = None


def compute_elevation_angle(mu_pos, uav_pos):
    """
    计算仰角 (公式10)
    ϑ = (180/π) * arctan(H / ||u_k - q_m||_horizontal)

    Args:
        mu_pos: MU位置 [x, y, 0], shape (..., 3) 或 (2,)
        uav_pos: UAV位置 [x, y, H], shape (..., 3) 或 (2,)
    Returns:
        仰角 (度)
    """
    if mu_pos.shape[-1] == 2:
        horizontal_dist = np.linalg.norm(mu_pos - uav_pos, axis=-1)
    else:
        horizontal_dist = np.linalg.norm(mu_pos[..., :2] - uav_pos[..., :2], axis=-1)
    horizontal_dist = np.maximum(horizontal_dist, 1e-6)  # 避免除零
    angle_rad = np.arctan(cfg.UAV_HEIGHT / horizontal_dist)
    return np.degrees(angle_rad)


def compute_los_probability(elevation_angle):
    """
    计算 LoS 概率 (公式9)
    P(LoS) = 1 / (1 + a * exp(-b * (ϑ - a)))

    Args:
        elevation_angle: 仰角 (度)
    Returns:
        LoS概率
    """
    return 1.0 / (1.0 + cfg.ENV_PARAM_A * np.exp(-cfg.ENV_PARAM_B * (elevation_angle - cfg.ENV_PARAM_A)))


def compute_distance(mu_pos, uav_pos):
    """
    计算 MU 与 UAV 之间的 3D 距离
    """
    if mu_pos.shape[-1] == 2:
        dx = mu_pos[..., 0] - uav_pos[..., 0]
        dy = mu_pos[..., 1] - uav_pos[..., 1]
        return np.sqrt(dx**2 + dy**2 + cfg.UAV_HEIGHT**2)
    else:
        return np.linalg.norm(mu_pos - uav_pos, axis=-1)


def compute_channel_gain(mu_pos, uav_pos):
    """
    计算 MU-UAV 信道增益 (公式11)
    h = β₀ * [P(LoS) + ν * P(NLoS)] / d^ι̃

    Args:
        mu_pos: MU位置 shape (..., 2) 或 (..., 3)
        uav_pos: UAV位置 shape (..., 2) 或 (..., 3)
    Returns:
        信道功率增益
    """
    dist_3d = compute_distance(mu_pos, uav_pos)
    dist_3d = np.maximum(dist_3d, 1.0)  # 最小距离 1m

    elev_angle = compute_elevation_angle(mu_pos, uav_pos)
    p_los = compute_los_probability(elev_angle)
    p_nlos = 1.0 - p_los

    gain = cfg.CHANNEL_POWER_GAIN * (p_los + cfg.NLOS_ATTENUATION * p_nlos) / (dist_3d ** cfg.PATH_LOSS_EXP)
    return gain


def compute_mu_uav_rate(mu_pos, uav_pos, bandwidth):
    """
    计算 MU-UAV 传输速率 (公式13)
    R = B_k,m * log2(1 + p_k * h / (B_k,m * N₀))

    Args:
        mu_pos: MU位置
        uav_pos: UAV位置
        bandwidth: 分配给该链路的带宽 (Hz)
    Returns:
        传输速率 (bits/s)
    """
    bandwidth = np.maximum(bandwidth, 1e-3)  # 避免零带宽
    h = compute_channel_gain(mu_pos, uav_pos)
    snr = cfg.MU_TRANSMIT_POWER * h / (bandwidth * cfg.NOISE_POWER_DENSITY)
    rate = bandwidth * np.log2(1.0 + snr)
    return rate


def compute_mu_uav_rate_batch(mu_positions, uav_positions, bandwidths):
    """
    批量计算 MU-UAV 传输速率。

    Args:
        mu_positions: shape (N, 2|3)
        uav_positions: shape (N, 2|3)
        bandwidths: shape (N,)
    Returns:
        shape (N,) 的速率数组
    """
    mu_positions = np.asarray(mu_positions, dtype=np.float32)
    uav_positions = np.asarray(uav_positions, dtype=np.float32)
    bandwidths = np.asarray(bandwidths, dtype=np.float32)

    if mu_positions.shape != uav_positions.shape:
        raise ValueError(f"mu_positions and uav_positions shape mismatch: {mu_positions.shape} vs {uav_positions.shape}")
    if bandwidths.ndim != 1 or bandwidths.shape[0] != mu_positions.shape[0]:
        raise ValueError(
            f"bandwidths shape mismatch: expected ({mu_positions.shape[0]},), got {bandwidths.shape}"
        )
    if _compute_mu_uav_rate_batch_jit is not None:
        return _compute_mu_uav_rate_batch_jit(mu_positions, uav_positions, bandwidths)
    return compute_mu_uav_rate(mu_positions, uav_positions, bandwidths).astype(np.float32, copy=False)


def compute_uav_bs_rate(uav_pos):
    """
    计算 UAV-BS 中继速率 (公式12, 14)
    使用准静态分组衰落 LoS 链路
    h_rel = β₀ / ||q_m - u_BS||²
    R_rel = B_u * log2(1 + p_m * h_rel / (B_u * N₀))

    Args:
        uav_pos: UAV位置 shape (..., 2) 或 (..., 3)
    Returns:
        中继传输速率 (bits/s)
    """
    if uav_pos.shape[-1] == 2:
        dx = uav_pos[..., 0] - cfg.BS_POSITION[0]
        dy = uav_pos[..., 1] - cfg.BS_POSITION[1]
        dist_sq = dx**2 + dy**2 + (cfg.UAV_HEIGHT - cfg.BS_POSITION[2])**2
    else:
        dist_sq = np.sum((uav_pos - cfg.BS_POSITION)**2, axis=-1)

    dist_sq = np.maximum(dist_sq, 1.0)
    h_rel = cfg.CHANNEL_POWER_GAIN / dist_sq

    snr = cfg.UAV_TRANSMIT_POWER * h_rel / (cfg.UAV_RELAY_BANDWIDTH * cfg.NOISE_POWER_DENSITY)
    rate = cfg.UAV_RELAY_BANDWIDTH * np.log2(1.0 + snr)
    return rate
