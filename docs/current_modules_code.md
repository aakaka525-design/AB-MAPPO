# 当前代码模块源码汇总

> 说明：该文档由脚本自动生成，收录当前仓库中非 tests 的 Python 模块完整源码。

## bench_parallel.py

```python
"""
Fast parallel benchmark: CPU vs CUDA, proper thread pinning.
Each test: 600 steps (2 episodes x 300), short sanity benchmark.
"""
import os
import subprocess
import sys
import time

STEPS, EPL, K, M, SEED = 600, 300, 60, 10, 42
CORES = os.cpu_count() or 24


def run_parallel(n, device="cpu", threads_per_proc=None):
    env = os.environ.copy()
    if threads_per_proc:
        env["OMP_NUM_THREADS"] = str(threads_per_proc)
        env["MKL_NUM_THREADS"] = str(threads_per_proc)
    procs = []
    t0 = time.perf_counter()
    for i in range(n):
        cmd = [
            sys.executable,
            "train.py",
            "--algorithm",
            "AB-MAPPO",
            "--num_mus",
            str(K),
            "--num_uavs",
            str(M),
            "--total_steps",
            str(STEPS),
            "--episode_length",
            str(EPL),
            "--seed",
            str(SEED + i),
            "--device",
            device,
            "--disable_tensorboard",
            "--run_dir",
            f"/tmp/bp_{device}_{n}_{i}",
        ]
        procs.append((i, subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)))
    for idx, p in procs:
        _, stderr = p.communicate()
        if p.returncode != 0:
            err_msg = stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"parallel job {idx} failed with code {p.returncode}: {err_msg}")
    elapsed = time.perf_counter() - t0
    eps_total = n * (STEPS / EPL) / elapsed
    return elapsed, eps_total


def main():
    print("=" * 60)
    print(f"Parallel Benchmark: K={K} M={M} steps={STEPS} cores={CORES}")
    print("=" * 60)

    # Key insight: threads_per_proc = CORES // n_parallel
    tests = [
        # (label, n_parallel, device, threads_per_proc)
        ("CPU x1 (24 thr)", 1, "cpu", 24),
        ("CPU x4 (6 thr)", 4, "cpu", 6),
        ("CPU x8 (3 thr)", 8, "cpu", 3),
        ("CPU x12 (2 thr)", 12, "cpu", 2),
        ("CPU x24 (1 thr)", 24, "cpu", 1),
        ("CUDA x1 (1 thr)", 1, "cuda", 1),
        ("CUDA x1 (24 thr)", 1, "cuda", 24),
    ]

    results = []
    for label, n, dev, thr in tests:
        print(f"[{len(results)+1}/{len(tests)}] {label}...", end=" ", flush=True)
        t, eps = run_parallel(n, dev, thr)
        results.append((label, n, t, eps))
        print(f"{t:.1f}s  {eps:.2f} eps/s total", flush=True)

    print()
    print("=" * 60)
    fmt = "{:<25} {:>4} {:>8} {:>12}"
    print(fmt.format("Config", "Jobs", "Time(s)", "eps/s total"))
    print("-" * 60)
    for label, n, t, eps in results:
        print(fmt.format(label, str(n), f"{t:.1f}", f"{eps:.2f}"))
    print("=" * 60)
    best = max(results, key=lambda x: x[3])
    print(f"\nHighest throughput: {best[0]} ({best[3]:.2f} eps/s, {best[2]:.1f}s)")


if __name__ == "__main__":
    main()
```

## benchmark_device.py

```python
"""
CPU vs CUDA 基准测试。
用 2400 步 (8 episodes × 300 步) 做快速对比。
"""
import os
import subprocess
import sys
import time

STEPS = 2400
EPL = 300
K, M = 60, 10
SEED = 42


def run_bench(label, device, blas_threads=None):
    env = os.environ.copy()
    if blas_threads:
        env["OMP_NUM_THREADS"] = str(blas_threads)
        env["MKL_NUM_THREADS"] = str(blas_threads)
    run_dir = f"/tmp/bench_{label}"
    cmd = [
        sys.executable,
        "train.py",
        "--algorithm",
        "AB-MAPPO",
        "--num_mus",
        str(K),
        "--num_uavs",
        str(M),
        "--total_steps",
        str(STEPS),
        "--episode_length",
        str(EPL),
        "--seed",
        str(SEED),
        "--device",
        device,
        "--disable_tensorboard",
        "--run_dir",
        run_dir,
    ]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    eps_s = "?"
    for line in r.stdout.strip().split("\n"):
        if "eps/s=" in line:
            eps_s = line.split("eps/s=")[-1].strip()
    return elapsed, eps_s, r.returncode


def run_parallel(n_parallel):
    """Run n_parallel CPU training processes simultaneously."""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    procs = []
    t0 = time.perf_counter()
    for i in range(n_parallel):
        run_dir = f"/tmp/bench_par{n_parallel}_{i}"
        cmd = [
            sys.executable,
            "train.py",
            "--algorithm",
            "AB-MAPPO",
            "--num_mus",
            str(K),
            "--num_uavs",
            str(M),
            "--total_steps",
            str(STEPS),
            "--episode_length",
            str(EPL),
            "--seed",
            str(SEED + i),
            "--device",
            "cpu",
            "--disable_tensorboard",
            "--run_dir",
            run_dir,
        ]
        procs.append((i, subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)))
    for idx, p in procs:
        _, stderr = p.communicate()
        if p.returncode != 0:
            err_msg = stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"parallel job {idx} failed with code {p.returncode}: {err_msg}")
    elapsed = time.perf_counter() - t0
    return elapsed


def main():
    print("=" * 60)
    print(f"AB-MAPPO Benchmark: K={K} M={M} steps={STEPS} epl={EPL}")
    print(f"CPU cores: {os.cpu_count()}")
    print("=" * 60)

    results = []

    print("\n[1/5] CPU single (default threads)...")
    t, eps, rc = run_bench("cpu_default", "cpu")
    results.append(("CPU single (default)", t, eps, rc))

    print("[2/5] CPU single (OMP=1)...")
    t, eps, rc = run_bench("cpu_omp1", "cpu", blas_threads=1)
    results.append(("CPU single (OMP=1)", t, eps, rc))

    print("[3/5] CUDA single...")
    t, eps, rc = run_bench("cuda", "cuda")
    results.append(("CUDA single", t, eps, rc))

    print("[4/5] CUDA single (OMP=1)...")
    t, eps, rc = run_bench("cuda_omp1", "cuda", blas_threads=1)
    results.append(("CUDA single (OMP=1)", t, eps, rc))

    print("[5/5] CPU x8 parallel (OMP=1)...")
    t_par = run_parallel(8)
    results.append(("CPU x8 parallel (OMP=1)", t_par, f"{8*STEPS/EPL/t_par:.2f} total", 0))

    print("\n" + "=" * 60)
    print(f"{'Config':<30} {'Time(s)':>8} {'eps/s':>10} {'rc':>3}")
    print("-" * 60)
    for label, t, eps, rc in results:
        print(f"{label:<30} {t:>8.1f} {str(eps):>10} {rc:>3}")
    print("=" * 60)

    successful = [r for r in results if r[3] == 0]
    if not successful:
        raise RuntimeError("No successful benchmark runs; cannot select best config.")
    best = min(successful, key=lambda x: x[1])
    print(f"\n最快方案: {best[0]} ({best[1]:.1f}s)")


if __name__ == "__main__":
    main()
```

## buffer.py

```python
"""
AB-MAPPO 论文复现 — 经验缓冲区 (Rollout Buffer)
存储一个Episode的轨迹数据, 并计算GAE优势函数
"""

import torch
import numpy as np
import config as cfg

try:
    from numba import njit
except Exception:  # pragma: no cover - optional acceleration path
    njit = None


def _compute_gae_numpy(delta, non_terminal, values, coef):
    size, num_agents = delta.shape
    advantages = np.zeros((size, num_agents), dtype=np.float32)
    returns = np.zeros((size, num_agents), dtype=np.float32)
    gae = np.zeros((num_agents,), dtype=np.float32)

    for t in range(size - 1, -1, -1):
        gae = delta[t] + coef * non_terminal[t] * gae
        advantages[t] = gae
        returns[t] = gae + values[t]
    return advantages, returns


if njit is not None:
    _compute_gae_numba = njit(cache=True, fastmath=True)(_compute_gae_numpy)
else:
    _compute_gae_numba = None


class RolloutBuffer:
    """
    On-policy Rollout Buffer for MAPPO

    存储一个episode的完整数据, episode结束时用于PPO更新
    支持异构智能体 (MU和UAV有不同的obs/action维度)
    """

    def __init__(self, num_agents, obs_dim, action_dim, buffer_size, gamma=0.99, gae_lambda=0.95, state_dim=None):
        self.num_agents = num_agents
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.state_dim = state_dim
        self.pos = 0
        self.full = False

        # 存储数据
        self.observations = np.zeros((buffer_size, num_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, num_agents, action_dim), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.values = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)
        self.states = None
        if self.state_dim is not None:
            self.states = np.zeros((buffer_size, self.state_dim), dtype=np.float32)

        # GAE计算后的数据
        self.advantages = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.returns = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self._compute_gae_impl = _compute_gae_numba or _compute_gae_numpy

    def add(self, obs, action, log_prob, reward, value, done, state=None):
        """添加一步数据"""
        if self.pos >= self.buffer_size:
            raise RuntimeError(
                f"RolloutBuffer overflow: pos={self.pos}, buffer_size={self.buffer_size}. "
                "Call reset() before adding new transitions."
            )
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.log_probs[self.pos] = log_prob
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.dones[self.pos] = done
        if self.states is not None:
            if state is None:
                raise ValueError("state is required when state_dim is set")
            self.states[self.pos] = state
        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True

    def compute_gae(self, last_values):
        """
        计算 GAE (Generalized Advantage Estimation)

        A_t = Σ_{l=0}^{T-t-1} (γλ)^l * δ_{t+l}
        δ_t = r_t + γ * V(s_{t+1}) - V(s_t)

        Args:
            last_values: (num_agents,) 最后一步的Value估计
        """
        size = self.pos if not self.full else self.buffer_size
        if size == 0:
            return

        non_terminal = 1.0 - self.dones[:size]
        next_values = np.empty_like(self.values[:size])
        if size > 1:
            next_values[:-1] = self.values[1:size]
        next_values[-1] = last_values
        delta = (
            self.rewards[:size]
            + self.gamma * next_values * non_terminal[:, None]
            - self.values[:size]
        ).astype(np.float32, copy=False)
        coef = np.float32(self.gamma * self.gae_lambda)
        adv, ret = self._compute_gae_impl(
            delta,
            non_terminal.astype(np.float32, copy=False),
            self.values[:size],
            coef,
        )
        self.advantages[:size] = adv
        self.returns[:size] = ret

    def get_batches(self, batch_size=None):
        """
        生成训练batch

        Returns:
            dict of tensors
        """
        size = self.pos if not self.full else self.buffer_size

        if batch_size is None or batch_size >= size:
            obs = self.observations[:size]
            actions = self.actions[:size]
            log_probs = self.log_probs[:size]
            advantages = self.advantages[:size]
            returns = self.returns[:size]
            values = self.values[:size]
            states = self.states[:size] if self.states is not None else None
        else:
            indices = np.random.choice(size, batch_size, replace=False)
            obs = self.observations[indices]
            actions = self.actions[indices]
            log_probs = self.log_probs[indices]
            advantages = self.advantages[indices]
            returns = self.returns[indices]
            values = self.values[indices]
            states = self.states[indices] if self.states is not None else None

        batch = {
            'observations': torch.from_numpy(obs),
            'actions': torch.from_numpy(actions),
            'log_probs': torch.from_numpy(log_probs),
            'advantages': torch.from_numpy(advantages),
            'returns': torch.from_numpy(returns),
            'values': torch.from_numpy(values),
        }
        if states is not None:
            batch['states'] = torch.from_numpy(states)
        return batch

    def reset(self):
        self.pos = 0
        self.full = False


class MultiAgentRolloutBuffer:
    """
    异构多智能体的Rollout Buffer

    分别管理 MU 和 UAV 的数据 (不同的obs/action维度)
    """

    def __init__(self, num_mus, num_uavs, mu_obs_dim, uav_obs_dim,
                 mu_action_dim, uav_action_dim, buffer_size,
                 gamma_mu=None, gamma_uav=None, state_dim=None):
        self.num_mus = num_mus
        self.num_uavs = num_uavs
        self.buffer_size = buffer_size
        self.pos = 0

        gamma_mu = gamma_mu or cfg.GAMMA_MU
        gamma_uav = gamma_uav or cfg.GAMMA_UAV

        self.mu_buffer = RolloutBuffer(
            num_mus, mu_obs_dim, mu_action_dim,
            buffer_size, gamma=gamma_mu, gae_lambda=cfg.GAE_LAMBDA, state_dim=state_dim
        )
        self.uav_buffer = RolloutBuffer(
            num_uavs, uav_obs_dim, uav_action_dim,
            buffer_size, gamma=gamma_uav, gae_lambda=cfg.GAE_LAMBDA, state_dim=state_dim
        )

    def add(self, mu_obs, mu_action, mu_log_prob, mu_reward, mu_value,
            uav_obs, uav_action, uav_log_prob, uav_reward, uav_value, done, state=None):
        self.mu_buffer.add(mu_obs, mu_action, mu_log_prob, mu_reward, mu_value, done, state=state)
        self.uav_buffer.add(uav_obs, uav_action, uav_log_prob, uav_reward, uav_value, done, state=state)
        self.pos = self.mu_buffer.pos

    def compute_gae(self, mu_last_values, uav_last_values):
        self.mu_buffer.compute_gae(mu_last_values)
        self.uav_buffer.compute_gae(uav_last_values)

    def get_batches(self, batch_size=None):
        return {
            'mu': self.mu_buffer.get_batches(batch_size),
            'uav': self.uav_buffer.get_batches(batch_size),
        }

    def reset(self):
        self.mu_buffer.reset()
        self.uav_buffer.reset()
        self.pos = 0

    @property
    def size(self):
        return self.mu_buffer.pos
```

## channel_model.py

```python
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
```

## colab_env_check.py

```python
"""
Google Colab 环境检测。

使用方式：在 Colab Notebook 的第一个 Cell 中粘贴以下内容并运行：

# ── Cell 1: 克隆项目 ──
# !git clone https://<your-repo-url> AB_MAPPO
# %cd AB_MAPPO

# ── Cell 2: 环境检测 ──
# %run colab_env_check.py

或者直接将下面的代码复制到一个 Cell 中运行。
"""
# fmt: off
# ────────────────────────────────────────────────
#  复制下面所有内容到 Colab Cell 中运行
# ────────────────────────────────────────────────

import os, sys, platform, subprocess

def _cmd(c):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.STDOUT).strip()
    except: return "(N/A)"

print("=" * 50)
print("  AB-MAPPO Colab 环境检测")
print("=" * 50)

# ── 系统 ──
print(f"\n{'─'*20} 系统 {'─'*20}")
print(f"Platform : {platform.platform()}")
print(f"Python   : {sys.version.split()[0]}")
print(f"CPU cores: {os.cpu_count()}")
print(f"Memory   : {_cmd('free -h | grep Mem')}")
print(f"Disk     : {_cmd('df -h / | tail -1')}")

# ── GPU ──
print(f"\n{'─'*20} GPU {'─'*21}")
smi = _cmd("nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader")
if "N/A" in smi or "failed" in smi.lower():
    print("⚠️  未检测到 GPU → Runtime → Change runtime type → GPU")
else:
    print(f"GPU      : {smi}")

try:
    import torch
    print(f"PyTorch  : {torch.__version__}")
    print(f"CUDA     : {torch.cuda.is_available()} (ver {getattr(torch.version, 'cuda', 'N/A')})")
    if torch.cuda.is_available():
        print(f"GPU name : {torch.cuda.get_device_name(0)}")
        gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU mem  : {gb:.1f} GB")
        # Beta 采样兼容性测试（本项目关键 op）
        from torch.distributions import Beta
        s = Beta(torch.tensor([2.0], device="cuda"), torch.tensor([3.0], device="cuda")).sample()
        print(f"Beta采样 : ✅ (CUDA 原生支持)")
except ImportError:
    print("PyTorch  : ❌ 未安装")

# ── 依赖 ──
print(f"\n{'─'*20} 依赖 {'─'*20}")
for pkg in ["torch", "numpy", "matplotlib", "tensorboard"]:
    try:
        m = __import__(pkg); print(f"  ✅ {pkg:15s} {getattr(m, '__version__', '?')}")
    except ImportError:
        print(f"  ❌ {pkg:15s} → pip install {pkg}")

# ── 项目文件 ──
print(f"\n{'─'*20} 项目 {'─'*20}")
root = os.path.dirname(os.path.abspath("__file__")) if os.path.exists("config.py") else "."
for f in ["config.py","train.py","environment.py","mappo.py","networks.py","buffer.py",
          "maddpg.py","channel_model.py","train_sweep.py","generate_figures.py","run_all.py"]:
    print(f"  {'✅' if os.path.exists(os.path.join(root,f)) else '❌'} {f}")

# ── 推荐 ──
print(f"\n{'─'*20} 推荐 {'─'*20}")
try:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
except: dev = "cpu"
cores = os.cpu_count() or 2
par = min(4, max(1, cores // 2))
print(f"设备: {dev} | 核心: {cores} | 建议并行: {par}")
print(f"\n# Smoke 测试")
print(f"!python run_all.py --stage smoke")
print(f"\n# Full Sweep")
print(f"!python run_all.py --stage full --max_parallel {par}")
if dev == "cuda":
    print(f"\n# 单次 GPU 训练")
    print(f"!python train.py --algorithm AB-MAPPO --device cuda --total_steps 80000")
print("\n" + "=" * 50)
print("  检测完成")
print("=" * 50)
```

## config.py

```python
"""
Global configuration for AB-MAPPO paper reproduction experiments.

Default values are aligned to:
Energy Efficient Computation Offloading in Aerial Edge Networks With Multi-Agent Cooperation
"""

from __future__ import annotations

import copy
import os
import numpy as np


# ============================================================
# Experiment meta
# ============================================================
EXPERIMENT_ROOT = os.environ.get("ABMAPPO_EXPERIMENT_ROOT", "experiments")
RESULT_DIR = os.environ.get("ABMAPPO_RESULT_DIR", "results")
DEFAULT_SEEDS = [42, 43, 44]

ALL_ALGOS = ["AB-MAPPO", "B-MAPPO", "AG-MAPPO", "MADDPG", "Randomized"]
CONVERGENCE_ALGOS = ["AB-MAPPO", "B-MAPPO", "AG-MAPPO", "MADDPG"]
COMPARISON_ALGOS = ["AB-MAPPO", "B-MAPPO", "AG-MAPPO", "MADDPG", "Randomized"]


# ============================================================
# Topology (paper defaults for main scenario)
# ============================================================
NUM_MUS = 60
NUM_UAVS = 10
AREA_WIDTH = 1000.0
AREA_HEIGHT = 1000.0
BS_POSITION = np.array([-500.0, 0.0, 10.0], dtype=np.float32)
UAV_HEIGHT = 200.0


# ============================================================
# Time
# ============================================================
TIME_PERIOD = 60.0
TIME_SLOT = 1.0
NUM_SLOTS = int(TIME_PERIOD / TIME_SLOT)  # 60 env steps per episode


# ============================================================
# Channel parameters (Table II)
# ============================================================
BANDWIDTH = 50e6
CHANNEL_POWER_GAIN_DB = -30
CHANNEL_POWER_GAIN = 10 ** (CHANNEL_POWER_GAIN_DB / 10)
ENV_PARAM_A = 15.0
ENV_PARAM_B = 0.5
PATH_LOSS_EXP = 2.2
NLOS_ATTENUATION = 0.2
NOISE_POWER_DENSITY_DBM = -127
NOISE_POWER_DENSITY = 10 ** ((NOISE_POWER_DENSITY_DBM - 30) / 10)

MU_TRANSMIT_POWER = 0.2
UAV_TRANSMIT_POWER = 0.5
UAV_RELAY_BANDWIDTH = 5e6


# ============================================================
# Computing parameters
# ============================================================
EFFECTIVE_CAPACITANCE = 1e-27
MU_MAX_CPU_FREQ = 1e9
UAV_MAX_CPU_FREQ = 10e9

TASK_DATA_MIN = 0.5e6
TASK_DATA_MAX = 2.0e6
TASK_CPU_CYCLES_MIN = 500.0
TASK_CPU_CYCLES_MAX = 1500.0


# ============================================================
# UAV mobility and propulsion
# ============================================================
UAV_MAX_VELOCITY = 30.0
UAV_MAX_ACCELERATION = 5.0

BLADE_PROFILE_POWER = 39.04
INDUCED_POWER = 79.07
TIP_SPEED = 120.0
MEAN_ROTOR_VELOCITY = 3.6
FUSELAGE_DRAG_RATIO = 0.012
ROTOR_SOLIDITY = 0.05
AIR_DENSITY = 1.225
ROTOR_DISC_AREA = 0.5030


# ============================================================
# MU mobility
# ============================================================
MU_MEMORY_FACTOR_V = 0.5
MU_MEMORY_FACTOR_THETA = 0.5
MU_MEAN_VELOCITY = 2.0
MU_MEAN_DIRECTION = 0.0
MU_VELOCITY_STD = 1.0
MU_DIRECTION_STD = 0.5


# ============================================================
# DT settings
# ============================================================
DT_DEVIATION_RATE = 0.0
WEIGHT_FACTOR = 0.001


# ============================================================
# MAPPO hyper-parameters (Table III aligned)
# NOTE:
# - The paper lists episode length 300 for training.
# - Environment still uses NUM_SLOTS=60 for physical episode horizon.
# - Training rollout can span multiple env episodes.
# ============================================================
TOTAL_STEPS = 80_000
EPISODE_LENGTH = 300

ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
GAMMA_UAV = 0.95
GAMMA_MU = 0.8
GAE_LAMBDA = 0.95
PPO_CLIP_EPSILON = 0.2
PPO_EPOCHS = 5
NUM_MINI_BATCHES = 1
HIDDEN_SIZE = 128
NUM_ATTENTION_HEADS = 4
ENTROPY_COEFF = 0.01
VALUE_LOSS_COEFF = 0.5
MAX_GRAD_NORM = 10.0


# ============================================================
# Reward / penalty
# ============================================================
# Paper scale is around 0.1 for penalty factors.
PENALTY_DELAY = 0.1
PENALTY_BOUNDARY = 0.1
PENALTY_COLLISION = 0.1
REWARD_SCALE = 10.0

D_MIN = 50.0
D_TH = 300.0


# ============================================================
# MADDPG hyper-parameters
# ============================================================
MADDPG_ACTOR_LR = 3e-4
MADDPG_CRITIC_LR = 3e-4
MADDPG_GAMMA = 0.95
MADDPG_TAU = 0.005
MADDPG_BUFFER_SIZE = 20_000
MADDPG_BATCH_SIZE = 256
# Unit: replay transitions (not episodes). With EPISODE_LENGTH=300, warmup starts after ~7 episodes.
MADDPG_WARMUP_STEPS = 2_000
MADDPG_UPDATES_PER_STEP = 1
MADDPG_NOISE_STD_INIT = 0.5
MADDPG_NOISE_STD_MIN = 0.05
MADDPG_NOISE_DECAY = 0.99995


# ============================================================
# Logging / checkpoints
# ============================================================
LOG_DIR = "logs"
SAVE_DIR = "checkpoints"
SAVE_INTERVAL = 5_000
LOG_INTERVAL = 100
EVAL_INTERVAL = 2_000


# ============================================================
# Figure grids (paper-aligned)
# ============================================================
FIGURE_SWEEPS = {
    "fig6": {
        "x_name": "num_mus",
        "x_values": [50, 60, 70, 80, 90, 100],
        "fixed": {"num_uavs": 10},
        "algorithms": COMPARISON_ALGOS,
        "metric": "weighted_energy_mu_total",
    },
    "fig7": {
        "x_name": "num_uavs",
        "x_values": [4, 6, 8, 10, 12, 14],
        "fixed": {"num_mus": 70},
        "algorithms": COMPARISON_ALGOS,
        "metric": "weighted_energy_mu_avg",
    },
    "fig8": {
        "x_name": "bandwidth_mhz",
        "x_values": [30, 40, 50, 60, 70],
        "fixed": {"num_mus": 60, "num_uavs": 10},
        "algorithms": COMPARISON_ALGOS,
        "metric": "weighted_energy_mu_avg",
    },
    "fig9": {
        "x_name": "weight_factor",
        "x_values": [0.0, 0.002, 0.004, 0.006, 0.008, 0.009],
        "fixed": {"num_mus": 60, "num_uavs": 10},
        "algorithms": ["AB-MAPPO"],
        "metric": "mu_uav_energy_pair",
    },
    "fig10": {
        "fixed": {"num_uavs": 10, "ks": [60, 80, 100]},
        "mu_cpu_ghz": [0.5, 1.0, 1.5, 2.0, 2.5],
        "uav_cpu_ghz": [2, 4, 6, 8, 10, 12, 14],
        "algorithms": ["AB-MAPPO"],
        "metric": "weighted_energy_mu_avg",
    },
    "fig11": {
        "fixed": {"num_uavs": 10, "ks": [50, 60, 70, 80]},
        "dt_values": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25],
        "algorithms": ["AB-MAPPO"],
        "metric": "weighted_energy_mu_avg",
    },
    "fig12": {
        "fixed": {"num_mus": 60, "num_uavs": 10},
        "task_max_mbits": [1, 1.5, 2, 2.5, 3],
        "bandwidth_mhz": [30, 50, 70],
        "algorithms": ["AB-MAPPO"],
        "metric": "weighted_energy_mu_avg",
    },
}


def paper_profile_dict() -> dict:
    """Return a copy of paper-aligned core defaults for external tools."""
    return {
        "NUM_MUS": NUM_MUS,
        "NUM_UAVS": NUM_UAVS,
        "BANDWIDTH": BANDWIDTH,
        "MU_MAX_CPU_FREQ": MU_MAX_CPU_FREQ,
        "UAV_MAX_CPU_FREQ": UAV_MAX_CPU_FREQ,
        "WEIGHT_FACTOR": WEIGHT_FACTOR,
        "TOTAL_STEPS": TOTAL_STEPS,
        "EPISODE_LENGTH": EPISODE_LENGTH,
    }


def clone_figure_sweeps() -> dict:
    return copy.deepcopy(FIGURE_SWEEPS)
```

## environment.py

```python
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
from channel_model import compute_mu_uav_rate, compute_mu_uav_rate_batch


class UAVMECEnv:
    """Multi-agent UAV-assisted MEC environment."""

    def __init__(
        self,
        num_mus: int | None = None,
        num_uavs: int | None = None,
        dt_deviation_rate: float | None = None,
        wo_dt_noise_mode: bool = False,
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
        self.mu_discrete_dim = self.M + 1
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
        dv = target_v - self.uav_velocities
        dv_norm = np.linalg.norm(dv, axis=1, keepdims=True)
        max_dv = cfg.UAV_MAX_ACCELERATION * cfg.TIME_SLOT
        scale = np.where(dv_norm > max_dv, max_dv / (dv_norm + 1e-8), 1.0)
        self.uav_velocities += dv * scale

        v_norm = np.linalg.norm(self.uav_velocities, axis=1, keepdims=True)
        scale_v = np.where(v_norm > cfg.UAV_MAX_VELOCITY, cfg.UAV_MAX_VELOCITY / (v_norm + 1e-8), 1.0)
        self.uav_velocities *= scale_v

        raw_pos = self.uav_positions + self.uav_velocities * cfg.TIME_SLOT
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

        if self.M > 1:
            other_uav = uav_pos_norm[self._other_uav_col_idx].reshape(self.M, self.M - 1, 2).reshape(self.M, -1)
            uav_obs[:, 2 + self.K * 5 :] = other_uav

        return self._apply_wo_dt_noise({"mu_obs": mu_obs, "uav_obs": uav_obs})

    def step(self, mu_actions: Dict, uav_actions: np.ndarray) -> Tuple[Dict, Dict, bool, Dict]:
        association = np.asarray(mu_actions["association"], dtype=np.int64).copy()
        offload_ratio = np.asarray(mu_actions["offload_ratio"], dtype=np.float32).copy()

        association = np.clip(association, 0, self.M)
        offload_ratio = np.clip(offload_ratio, 0.0, 1.0)

        local_mask = association == 0
        offload_ratio[local_mask] = 0.0

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
        assoc_mus_list = [np.where(association == (m + 1))[0] for m in range(self.M)]

        bandwidth_alloc = self._bw_alloc_cache
        bandwidth_alloc.fill(0.0)
        cpu_alloc = self._cpu_alloc_cache
        cpu_alloc.fill(0.0)
        edge_load = self._edge_load_cache
        edge_load.fill(0.0)

        for m in range(self.M):
            assoc_mus = assoc_mus_list[m]
            if len(assoc_mus) == 0:
                continue

            bw_weights = self._softmax(raw_bw[m, assoc_mus] * 3.0)
            cpu_weights = self._softmax(raw_cpu[m, assoc_mus] * 3.0)

            bandwidth_alloc[assoc_mus] = bw_weights * cfg.BANDWIDTH
            cpu_alloc[assoc_mus] = cpu_weights * cfg.UAV_MAX_CPU_FREQ

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
        uav_comp_energy = self._uav_comp_cache
        uav_comp_energy.fill(0.0)
        offload_mask = (association > 0) & (offload_ratio > 1e-8)
        offload_indices = np.where(offload_mask)[0]
        if offload_indices.size > 0:
            uav_idx = association[offload_indices] - 1
            bw = np.maximum(bandwidth_alloc[offload_indices], 1e3)
            cpu = np.maximum(cpu_alloc[offload_indices], 1e3)

            offload_data = offload_ratio[offload_indices] * l_all[offload_indices]
            rates = compute_mu_uav_rate_batch(
                self.mu_positions[offload_indices],
                self.uav_positions[uav_idx],
                bw,
            ).astype(np.float32)
            rates = np.maximum(rates, 1e3)

            t_off = offload_data / rates
            e_off = cfg.MU_TRANSMIT_POWER * t_off

            y_edge = offload_ratio[offload_indices] * l_all[offload_indices] * c_all[offload_indices]
            f_dev_edge = cpu * self.dt_deviation_rate * self.rng.uniform(-1.0, 1.0, size=offload_indices.size)
            f_actual_edge = np.maximum(cpu + f_dev_edge, cpu * 0.1)
            t_ecmp = y_edge / np.maximum(f_actual_edge, 1e3)
            e_edge = cfg.EFFECTIVE_CAPACITANCE * (f_actual_edge**2) * y_edge

            mu_energy[offload_indices] += e_off.astype(np.float32)
            t_edge_all[offload_indices] = (t_off + t_ecmp).astype(np.float32)
            uav_comp_energy += np.bincount(uav_idx, weights=e_edge, minlength=self.M).astype(np.float32)
            edge_load[uav_idx, offload_indices] = y_edge.astype(np.float32)

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
        assoc_counts = np.bincount(association, minlength=self.M + 1)[1:]
        n_assoc_per_uav = np.maximum(assoc_counts, 1.0).astype(np.float32)

        latency_penalty_all = self._latency_penalty_cache
        latency_penalty_all[:] = (
            cfg.PENALTY_DELAY
            * (np.maximum(t_loc_all - cfg.TIME_SLOT, 0.0) + np.maximum(t_edge_all - cfg.TIME_SLOT, 0.0))
            / cfg.TIME_SLOT
        )

        mu_rewards = self._mu_rewards_cache
        mu_rewards[:] = -mu_energy - latency_penalty_all
        valid_assoc = association > 0
        if np.any(valid_assoc):
            uav_indices = association[valid_assoc] - 1
            mu_rewards[valid_assoc] -= (
                cfg.WEIGHT_FACTOR * uav_total_energy[uav_indices] / n_assoc_per_uav[uav_indices]
            ).astype(np.float32)
        mu_rewards *= cfg.REWARD_SCALE

        uav_rewards = self._uav_rewards_cache
        uav_rewards[:] = -cfg.WEIGHT_FACTOR * uav_total_energy
        collision_penalties = self._collision_penalties()
        boundary_penalties = (cfg.PENALTY_BOUNDARY * self._boundary_violation).astype(np.float32)
        if np.any(valid_assoc):
            assoc_uav = association[valid_assoc] - 1
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

        # Fairness is computed over MU utility (inverse energy cost), not raw cost.
        # This better reflects "service fairness" in paper-style comparisons.
        mu_utility = 1.0 / (mu_energy + 1e-8)
        fairness_denom = float(self.K * np.sum(mu_utility**2) + 1e-8)
        jain_fairness = float((np.sum(mu_utility) ** 2) / fairness_denom) if fairness_denom > 0 else 1.0

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
```

## evaluate.py

```python
"""
Deprecated compatibility wrapper.

Use generate_figures.py as the single authoritative entrypoint.
"""

from __future__ import annotations

import warnings

import sys

from generate_figures import main


if __name__ == "__main__":
    warnings.warn(
        "evaluate.py is deprecated. Use generate_figures.py instead.",
        DeprecationWarning,
        stacklevel=1,
    )
    # Backward compatibility: old script used --fig; new script uses --figs.
    if "--fig" in sys.argv and "--figs" not in sys.argv:
        idx = sys.argv.index("--fig")
        sys.argv[idx] = "--figs"
    main()
```

## generate_figures.py

```python
"""
Single authoritative figure generator for paper Fig.3-Fig.13.

Input:
- Aggregated sweep results in experiments/{fig}/aggregate/*.json
Output:
- PNG figures under results/

Note:
- Fig.13 is a deterministic heuristic trajectory visualization, not a direct
  rendering of aggregate JSON outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from typing import Dict, List


def _is_writable_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        return False

    probe = os.path.join(path, ".write_probe")
    try:
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
    except OSError:
        return False
    finally:
        try:
            if os.path.exists(probe):
                os.remove(probe)
        except OSError:
            pass
    return True


def _ensure_writable_mplconfigdir(candidates: List[str] | None = None) -> str | None:
    existing = os.environ.get("MPLCONFIGDIR")
    if existing:
        return existing

    if candidates is None:
        candidates = [
            os.path.join(tempfile.gettempdir(), "ab_mappo_mplconfig"),
            os.path.join(os.getcwd(), ".mplconfig"),
        ]

    for path in candidates:
        if _is_writable_dir(path):
            os.environ["MPLCONFIGDIR"] = path
            return path
    return None


_ensure_writable_mplconfigdir()

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from environment import UAVMECEnv


COLORS = {
    "AB-MAPPO": "#1f77b4",
    "B-MAPPO": "#ff7f0e",
    "AG-MAPPO": "#2ca02c",
    "MADDPG": "#d62728",
    "Randomized": "#9467bd",
}
MARKERS = {
    "AB-MAPPO": "o",
    "B-MAPPO": "s",
    "AG-MAPPO": "^",
    "MADDPG": "D",
    "Randomized": "x",
}
K_SERIES_PATTERN = re.compile(r"^K=(\d+)$")


def _sort_series_labels_by_k(labels: List[str]) -> List[str]:
    parsed = []
    for label in labels:
        m = K_SERIES_PATTERN.match(label)
        if m is None:
            return sorted(labels)
        parsed.append((int(m.group(1)), label))
    parsed.sort(key=lambda x: x[0])
    return [label for _, label in parsed]


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(fig, output_dir, name):
    _ensure_dir(output_dir)
    out = os.path.join(output_dir, name)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"[saved] {out}")


def _path(exp_root, *parts):
    return os.path.join(exp_root, *parts)


def plot_fig3(exp_root, output_dir):
    p = _path(exp_root, "base", "aggregate", "convergence.json")
    if not os.path.exists(p):
        print(f"[skip] fig3 missing {p}")
        return
    data = _load_json(p)
    steps = np.asarray(data["steps"], dtype=float)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for algo in cfg.CONVERGENCE_ALGOS:
        if algo not in data["algorithms"]:
            continue
        d = data["algorithms"][algo]
        mu = np.asarray(d["mu_reward_mean"], dtype=float)
        mu_std = np.asarray(d["mu_reward_std"])
        uav = np.asarray(d["uav_reward_mean"], dtype=float)
        uav_std = np.asarray(d["uav_reward_std"])
        ax1.plot(steps, mu, label=algo, color=COLORS[algo], linewidth=1.9)
        ax1.fill_between(steps, mu - mu_std, mu + mu_std, color=COLORS[algo], alpha=0.12)
        ax2.plot(steps, uav, label=algo, color=COLORS[algo], linewidth=1.9)
        ax2.fill_between(steps, uav - uav_std, uav + uav_std, color=COLORS[algo], alpha=0.12)
    ax1.set_title("(a) Average reward of MUs")
    ax2.set_title("(b) Average reward of UAVs")
    for ax in (ax1, ax2):
        ax.set_xlabel("Training steps")
        ax.set_ylabel("Average episode reward")
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend()
    _save(fig, output_dir, "fig3_reward_convergence.png")


def plot_fig4(exp_root, output_dir):
    p = _path(exp_root, "base", "aggregate", "convergence.json")
    if not os.path.exists(p):
        print(f"[skip] fig4 missing {p}")
        return
    data = _load_json(p)
    steps = np.asarray(data["steps"], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo in cfg.CONVERGENCE_ALGOS:
        if algo not in data["algorithms"]:
            continue
        d = data["algorithms"][algo]
        y = np.asarray(d["weighted_energy_mean"], dtype=float)
        y_std = np.asarray(d["weighted_energy_std"])
        ax.plot(steps, y, label=algo, color=COLORS[algo], linewidth=1.9)
        ax.fill_between(steps, y - y_std, y + y_std, color=COLORS[algo], alpha=0.12)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Average weighted energy of MUs (J)")
    ax.set_title("Fig.4 Convergence of weighted energy")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    _save(fig, output_dir, "fig4_energy_convergence.png")


def plot_fig5(exp_root, output_dir):
    p = _path(exp_root, "base", "aggregate", "convergence.json")
    if not os.path.exists(p):
        print(f"[skip] fig5 missing {p}")
        return
    data = _load_json(p)
    steps = np.asarray(data["steps"], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo in ["AB-MAPPO", "AG-MAPPO"]:
        if algo not in data["algorithms"]:
            continue
        d = data["algorithms"][algo]
        y = np.asarray(d["jain_fairness_mean"], dtype=float)
        y_std = np.asarray(d["jain_fairness_std"])
        ax.plot(steps, y, label=algo, color=COLORS[algo], linewidth=1.9)
        ax.fill_between(steps, y - y_std, y + y_std, color=COLORS[algo], alpha=0.12)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Jain fairness index")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Fig.5 Fairness of MUs")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    _save(fig, output_dir, "fig5_fairness.png")


def _plot_multi_algo_bar(fig_name, x_label, y_label, title, p, output_dir):
    if not os.path.exists(p):
        print(f"[skip] {fig_name} missing {p}")
        return
    data = _load_json(p)
    x = np.asarray(data["x_values"], dtype=float)
    algorithms = [algo for algo in cfg.COMPARISON_ALGOS if algo in data["algorithms"]]

    series = {}
    stds = {}
    bar_colors = dict(COLORS)
    for algo in algorithms:
        d = data["algorithms"].get(algo)
        y = np.asarray(d["mean"], dtype=float)
        series[algo] = y
        stds[algo] = np.asarray(d["std"], dtype=float)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    x_idx = np.arange(len(x), dtype=float)
    width = min(0.16, 0.82 / max(1, len(algorithms)))
    left = -0.5 * width * (len(algorithms) - 1)

    for i, algo in enumerate(algorithms):
        pos = x_idx + left + i * width
        y = series[algo]
        y_std = stds[algo]
        ax.bar(
            pos,
            y,
            width=width,
            color=bar_colors.get(algo, COLORS[algo]),
            edgecolor="white",
            linewidth=0.6,
            alpha=0.92,
            label=algo,
        )
        if len(y_std) == len(y):
            ax.errorbar(pos, y, yerr=y_std, fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2, alpha=0.35)

    x_labels = []
    for v in x:
        if abs(v - int(v)) < 1e-8:
            x_labels.append(str(int(v)))
        else:
            x_labels.append(f"{v:g}")
    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.20, linestyle="--", axis="y")
    if fig_name == "fig8_vs_bandwidth":
        ax.legend(loc="lower left")
    else:
        ax.legend()
    _save(fig, output_dir, f"{fig_name}.png")


def plot_fig6(exp_root, output_dir):
    _plot_multi_algo_bar(
        "fig6_vs_K",
        "Number of MUs (K)",
        "Weighted energy consumption of MUs (J)",
        "Fig.6 Impact of number of MUs",
        _path(exp_root, "fig6", "aggregate", "results.json"),
        output_dir,
    )


def plot_fig7(exp_root, output_dir):
    _plot_multi_algo_bar(
        "fig7_vs_M",
        "Number of UAVs (M)",
        "Average weighted energy of MUs (J)",
        "Fig.7 Impact of number of UAVs",
        _path(exp_root, "fig7", "aggregate", "results.json"),
        output_dir,
    )


def plot_fig8(exp_root, output_dir):
    _plot_multi_algo_bar(
        "fig8_vs_bandwidth",
        "Bandwidth B (MHz)",
        "Average weighted energy of MUs (J)",
        "Fig.8 Impact of bandwidth",
        _path(exp_root, "fig8", "aggregate", "results.json"),
        output_dir,
    )


def plot_fig9(exp_root, output_dir):
    p = _path(exp_root, "fig9", "aggregate", "results.json")
    if not os.path.exists(p):
        print(f"[skip] fig9 missing {p}")
        return
    data = _load_json(p)
    x = np.asarray(data["x_values"], dtype=float)
    mu = np.asarray(data["mu_energy"]["mean"], dtype=float)
    mu_std = np.asarray(data["mu_energy"]["std"], dtype=float)
    uav = np.asarray(data["uav_energy"]["mean"], dtype=float)
    uav_std = np.asarray(data["uav_energy"]["std"], dtype=float)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    l1 = ax1.plot(x, mu, "o-", color="#1f77b4", label="MU energy", linewidth=2.0)[0]
    ax1.fill_between(x, mu - mu_std, mu + mu_std, color="#1f77b4", alpha=0.12)
    l2 = ax2.plot(x, uav, "s--", color="#d62728", label="UAV energy", linewidth=2.0)[0]
    ax2.fill_between(x, uav - uav_std, uav + uav_std, color="#d62728", alpha=0.12)
    ax1.set_xlabel("Weight factor")
    ax1.set_ylabel("MU energy (J)", color="#1f77b4")
    ax2.set_ylabel("UAV energy (J)", color="#d62728")
    ax1.set_title("Fig.9 Impact of weight factor")
    ax1.grid(alpha=0.25, linestyle="--")
    ax1.legend([l1, l2], ["MU energy", "UAV energy"], loc="best")
    _save(fig, output_dir, "fig9_vs_weight.png")


def plot_fig10(exp_root, output_dir):
    p = _path(exp_root, "fig10", "aggregate", "results.json")
    if not os.path.exists(p):
        print(f"[skip] fig10 missing {p}")
        return
    data = _load_json(p)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x1 = np.asarray(data["mu_cpu"]["x_values"], dtype=float)
    sorted_mu_labels = _sort_series_labels_by_k(list(data["mu_cpu"]["series"].keys()))
    for i, k in enumerate(sorted_mu_labels):
        series = data["mu_cpu"]["series"][k]
        y = np.asarray(series["mean"], dtype=float)
        y_std = np.asarray(series["std"], dtype=float)
        c = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][i % 4]
        ax1.plot(x1, y, marker="o", color=c, linewidth=1.9, label=k)
        ax1.fill_between(x1, y - y_std, y + y_std, color=c, alpha=0.10)
    ax1.set_xlabel("MU computational resource (GHz)")
    ax1.set_ylabel("Average weighted energy of MUs (J)")
    ax1.set_title("(a) Impact of MU computational resource")
    ax1.grid(alpha=0.25, linestyle="--")
    ax1.legend()

    x2 = np.asarray(data["uav_cpu"]["x_values"], dtype=float)
    sorted_uav_labels = _sort_series_labels_by_k(list(data["uav_cpu"]["series"].keys()))
    for i, k in enumerate(sorted_uav_labels):
        series = data["uav_cpu"]["series"][k]
        y = np.asarray(series["mean"], dtype=float)
        y_std = np.asarray(series["std"], dtype=float)
        c = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][i % 4]
        ax2.plot(x2, y, marker="s", color=c, linewidth=1.9, label=k)
        ax2.fill_between(x2, y - y_std, y + y_std, color=c, alpha=0.10)
    ax2.set_xlabel("UAV computational resource (GHz)")
    ax2.set_ylabel("Average weighted energy of MUs (J)")
    ax2.set_title("(b) Impact of UAV computational resource")
    ax2.grid(alpha=0.25, linestyle="--")
    ax2.legend()

    _save(fig, output_dir, "fig10_vs_cpu.png")


def plot_fig11(exp_root, output_dir):
    p = _path(exp_root, "fig11", "aggregate", "results.json")
    if not os.path.exists(p):
        print(f"[skip] fig11 missing {p}")
        return
    data = _load_json(p)
    x = np.asarray(data["x_values"], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_labels = _sort_series_labels_by_k(list(data["series"].keys()))
    for i, label in enumerate(sorted_labels):
        series = data["series"][label]
        y = np.asarray(series["mean"], dtype=float)
        y_std = np.asarray(series["std"], dtype=float)
        c = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][i % 4]
        ax.plot(x, y, marker="o", linewidth=2.0, color=c, label=label)
        ax.fill_between(x, y - y_std, y + y_std, color=c, alpha=0.10)
    ax.set_xlabel("DT deviation rate")
    ax.set_ylabel("Average weighted energy of MUs (J)")
    ax.set_title("Fig.11 Impact of DT deviation")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    _save(fig, output_dir, "fig11_vs_dt.png")


def plot_fig12(exp_root, output_dir):
    p = _path(exp_root, "fig12", "aggregate", "results.json")
    if not os.path.exists(p):
        print(f"[skip] fig12 missing {p}")
        return
    data = _load_json(p)
    x = np.asarray(data["x_values"], dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 5))

    def _series_order(name: str):
        bw = 999
        if "30" in name:
            bw = 30
        elif "50" in name:
            bw = 50
        elif "70" in name:
            bw = 70
        mode = 0 if "with DT" in name else 1
        return (bw, mode)

    for label in sorted(data["series"].keys(), key=_series_order):
        series = data["series"][label]
        y = np.asarray(series["mean"], dtype=float)
        y_std = np.asarray(series["std"], dtype=float)
        linestyle = "--" if "w/o DT" in label else "-"
        if "30" in label:
            color = "#d62728"
            marker = "o"
        elif "50" in label:
            color = "#1f77b4"
            marker = "s"
        else:
            color = "#2ca02c"
            marker = "^"
        ax.plot(x, y, linestyle=linestyle, marker=marker, color=color, linewidth=1.9, label=label)
        ax.fill_between(x, y - y_std, y + y_std, color=color, alpha=0.08)
    ax.set_xlabel("Maximum task size (Mbits)")
    ax.set_ylabel("Average weighted energy of MUs (J)")
    ax.set_title("Fig.12 Impact of maximum task size")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, output_dir, "fig12_vs_tasksize.png")


def _make_corner_positions(width, height, m, rng):
    base = np.array(
        [
            [0.05 * width, 0.05 * height],
            [0.95 * width, 0.05 * height],
            [0.95 * width, 0.95 * height],
            [0.05 * width, 0.95 * height],
        ],
        dtype=np.float32,
    )
    if m <= 4:
        return base[:m].copy()
    reps = int(np.ceil(m / 4))
    arr = np.vstack([base for _ in range(reps)])[:m]
    arr += rng.normal(0.0, 0.03 * min(width, height), size=arr.shape)
    arr[:, 0] = np.clip(arr[:, 0], 0.0, width)
    arr[:, 1] = np.clip(arr[:, 1], 0.0, height)
    return arr


def _run_heuristic_trajectory(scenario, seed):
    rng = np.random.default_rng(seed)
    k = scenario["K"]
    m = scenario["M"]
    w = scenario["W"]
    t_horizon = scenario["T"]
    env = UAVMECEnv(
        num_mus=k,
        num_uavs=m,
        area_width=w,
        area_height=w,
        dt_deviation_rate=0.0,
        wo_dt_noise_mode=False,
        rng=rng,
    )
    env.reset()

    if scenario.get("cluster", False):
        centers = np.array([[0.25 * w, 0.25 * w], [0.75 * w, 0.7 * w], [0.55 * w, 0.2 * w]], dtype=np.float32)
        for i in range(k):
            env.mu_positions[i] = centers[i % len(centers)] + rng.normal(0, 0.04 * w, size=2)
        env.mu_positions[:, 0] = np.clip(env.mu_positions[:, 0], 0.0, w)
        env.mu_positions[:, 1] = np.clip(env.mu_positions[:, 1], 0.0, w)

    if scenario.get("uav_start", "random") == "corner":
        env.uav_positions = _make_corner_positions(w, w, m, rng)
        env.uav_velocities[:] = 0.0

    trajs = [[env.uav_positions[i].copy()] for i in range(m)]
    for _ in range(t_horizon):
        assoc = np.zeros((k,), dtype=np.int64)
        offload = np.full((k,), 0.5, dtype=np.float32)
        for i in range(k):
            d = np.linalg.norm(env.uav_positions - env.mu_positions[i], axis=1)
            assoc[i] = int(np.argmin(d)) + 1

        uav_actions = np.full((m, env.uav_continuous_dim), 0.5, dtype=np.float32)
        for j in range(m):
            users = np.where(assoc == (j + 1))[0]
            if len(users) == 0:
                target = env.mu_positions[rng.integers(0, k)]
            else:
                target = env.mu_positions[users].mean(axis=0)
            delta = target - env.uav_positions[j]
            norm = np.linalg.norm(delta)
            if norm > 1e-4:
                uav_actions[j, 0] = np.clip(0.5 + 0.45 * delta[0] / norm, 0.0, 1.0)
                uav_actions[j, 1] = np.clip(0.5 + 0.45 * delta[1] / norm, 0.0, 1.0)

        env.step({"association": assoc, "offload_ratio": offload}, uav_actions)
        for j in range(m):
            trajs[j].append(env.uav_positions[j].copy())
    return env.mu_positions.copy(), trajs


def plot_fig13(exp_root, output_dir, base_seed):
    _ = exp_root
    scenarios = [
        {"title": "(a) K=12, M=3, W=500m, T=50s", "K": 12, "M": 3, "W": 500, "T": 50, "cluster": True, "uav_start": "random"},
        {"title": "(b) K=12, M=3, W=500m, T=50s", "K": 12, "M": 3, "W": 500, "T": 50, "cluster": False, "uav_start": "corner"},
        {"title": "(c) K=60, M=10, W=1000m, T=60s", "K": 60, "M": 10, "W": 1000, "T": 60, "cluster": False, "uav_start": "random"},
        {"title": "(d) K=60, M=10, W=1000m, T=60s", "K": 60, "M": 10, "W": 1000, "T": 60, "cluster": False, "uav_start": "corner"},
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    cmap = plt.get_cmap("tab10")

    for idx, (ax, sc) in enumerate(zip(axes.flat, scenarios)):
        mu_pos, trajs = _run_heuristic_trajectory(sc, seed=base_seed + idx)
        ax.scatter(mu_pos[:, 0], mu_pos[:, 1], s=14, c="gray", alpha=0.45, label="MU")
        for i, tr in enumerate(trajs):
            arr = np.asarray(tr)
            color = cmap(i % 10)
            ax.plot(arr[:, 0], arr[:, 1], "-", color=color, linewidth=1.8, alpha=0.9)
            ax.scatter(arr[0, 0], arr[0, 1], marker="^", s=70, c=[color], edgecolors="k", linewidths=0.6)
            ax.scatter(arr[-1, 0], arr[-1, 1], marker="*", s=95, c=[color], edgecolors="k", linewidths=0.6)
        ax.set_xlim(-0.05 * sc["W"], 1.05 * sc["W"])
        ax.set_ylim(-0.05 * sc["W"], 1.05 * sc["W"])
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(sc["title"], fontsize=10, fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(alpha=0.22, linestyle="--")
    _save(fig, output_dir, "fig13_trajectory.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Fig.3-13 from aggregated experiment outputs")
    parser.add_argument("--exp_root", type=str, default=cfg.EXPERIMENT_ROOT)
    parser.add_argument("--output_dir", type=str, default=cfg.RESULT_DIR)
    parser.add_argument("--figs", type=str, default="all", help='all or comma list from "3,4,...,13"')
    parser.add_argument("--allow-missing", action="store_true", help="allow missing inputs and skip corresponding figs")
    parser.add_argument("--fig13-seed", type=int, default=20260225, help="base seed for deterministic Fig.13")
    return parser.parse_args()


def _required_inputs(exp_root, fig_number):
    req = {
        3: [_path(exp_root, "base", "aggregate", "convergence.json")],
        4: [_path(exp_root, "base", "aggregate", "convergence.json")],
        5: [_path(exp_root, "base", "aggregate", "convergence.json")],
        6: [_path(exp_root, "fig6", "aggregate", "results.json")],
        7: [_path(exp_root, "fig7", "aggregate", "results.json")],
        8: [_path(exp_root, "fig8", "aggregate", "results.json")],
        9: [_path(exp_root, "fig9", "aggregate", "results.json")],
        10: [_path(exp_root, "fig10", "aggregate", "results.json")],
        11: [_path(exp_root, "fig11", "aggregate", "results.json")],
        12: [_path(exp_root, "fig12", "aggregate", "results.json")],
        13: [],
    }
    return req[fig_number]


def main():
    args = parse_args()
    _ensure_dir(args.output_dir)
    print("=" * 60)
    print("Generating paper figures (Fig.3-Fig.13)")
    print("=" * 60)

    runners = {
        3: lambda: plot_fig3(args.exp_root, args.output_dir),
        4: lambda: plot_fig4(args.exp_root, args.output_dir),
        5: lambda: plot_fig5(args.exp_root, args.output_dir),
        6: lambda: plot_fig6(args.exp_root, args.output_dir),
        7: lambda: plot_fig7(args.exp_root, args.output_dir),
        8: lambda: plot_fig8(args.exp_root, args.output_dir),
        9: lambda: plot_fig9(args.exp_root, args.output_dir),
        10: lambda: plot_fig10(args.exp_root, args.output_dir),
        11: lambda: plot_fig11(args.exp_root, args.output_dir),
        12: lambda: plot_fig12(args.exp_root, args.output_dir),
        13: lambda: plot_fig13(args.exp_root, args.output_dir, args.fig13_seed),
    }

    if args.figs == "all":
        targets = list(range(3, 14))
    else:
        targets = [int(x.strip()) for x in args.figs.split(",") if x.strip()]

    missing = []
    for n in targets:
        if n not in runners:
            raise ValueError(f"Unsupported figure number: {n}")
        for p in _required_inputs(args.exp_root, n):
            if not os.path.exists(p):
                missing.append((n, p))

    if missing and not args.allow_missing:
        print("[error] missing required inputs:")
        for fig_n, p in missing:
            print(f"  Fig.{fig_n}: {p}")
        raise SystemExit(2)

    for n in targets:
        print(f"[run] Fig.{n}")
        runners[n]()

    print(f"done. figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
```

## maddpg.py

```python
"""
MADDPG baseline for the UAV-MEC environment.

This implementation uses:
- shared MU actor
- shared UAV actor
- centralized Q critic
- continuous association relaxation (argmax to discrete association for env step)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

import config as cfg
from environment import UAVMECEnv
from networks import CentralizedQCritic, DeterministicActor


@dataclass
class ReplayBatch:
    state: torch.Tensor
    mu_obs: torch.Tensor
    uav_obs: torch.Tensor
    mu_action_repr: torch.Tensor
    uav_action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    next_mu_obs: torch.Tensor
    next_uav_obs: torch.Tensor
    done: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity, state_dim, num_mus, num_uavs, mu_obs_dim, uav_obs_dim, mu_repr_dim, uav_act_dim):
        self.capacity = int(capacity)
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.mu_obs = np.zeros((capacity, num_mus, mu_obs_dim), dtype=np.float32)
        self.uav_obs = np.zeros((capacity, num_uavs, uav_obs_dim), dtype=np.float32)
        self.mu_action_repr = np.zeros((capacity, num_mus, mu_repr_dim), dtype=np.float32)
        self.uav_action = np.zeros((capacity, num_uavs, uav_act_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_mu_obs = np.zeros((capacity, num_mus, mu_obs_dim), dtype=np.float32)
        self.next_uav_obs = np.zeros((capacity, num_uavs, uav_obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

    def add(
        self,
        state,
        mu_obs,
        uav_obs,
        mu_action_repr,
        uav_action,
        reward,
        next_state,
        next_mu_obs,
        next_uav_obs,
        done,
    ):
        i = self.ptr
        self.state[i] = state
        self.mu_obs[i] = mu_obs
        self.uav_obs[i] = uav_obs
        self.mu_action_repr[i] = mu_action_repr
        self.uav_action[i] = uav_action
        self.reward[i] = reward
        self.next_state[i] = next_state
        self.next_mu_obs[i] = next_mu_obs
        self.next_uav_obs[i] = next_uav_obs
        self.done[i] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device) -> ReplayBatch:
        idx = np.random.randint(0, self.size, size=batch_size)
        return ReplayBatch(
            state=torch.as_tensor(self.state[idx], dtype=torch.float32, device=device),
            mu_obs=torch.as_tensor(self.mu_obs[idx], dtype=torch.float32, device=device),
            uav_obs=torch.as_tensor(self.uav_obs[idx], dtype=torch.float32, device=device),
            mu_action_repr=torch.as_tensor(self.mu_action_repr[idx], dtype=torch.float32, device=device),
            uav_action=torch.as_tensor(self.uav_action[idx], dtype=torch.float32, device=device),
            reward=torch.as_tensor(self.reward[idx], dtype=torch.float32, device=device),
            next_state=torch.as_tensor(self.next_state[idx], dtype=torch.float32, device=device),
            next_mu_obs=torch.as_tensor(self.next_mu_obs[idx], dtype=torch.float32, device=device),
            next_uav_obs=torch.as_tensor(self.next_uav_obs[idx], dtype=torch.float32, device=device),
            done=torch.as_tensor(self.done[idx], dtype=torch.float32, device=device),
        )


class MADDPG:
    def __init__(self, env: UAVMECEnv, device="cpu"):
        self.env = env
        self.device = torch.device(device)
        self.K = env.K
        self.M = env.M

        # MU actor outputs association probabilities and offload ratio.
        self.mu_assoc_dim = self.M + 1
        self.mu_cont_dim = 1
        self.mu_repr_dim = self.mu_assoc_dim + self.mu_cont_dim
        self.uav_action_dim = env.uav_continuous_dim

        self.state_dim = env.state_dim
        self.total_action_dim = self.K * self.mu_repr_dim + self.M * self.uav_action_dim

        self.mu_actor = DeterministicActor(env.mu_obs_dim, action_dim=self.mu_cont_dim, assoc_dim=self.mu_assoc_dim).to(
            self.device
        )
        self.uav_actor = DeterministicActor(env.uav_obs_dim, action_dim=self.uav_action_dim, assoc_dim=0).to(self.device)
        self.mu_actor_target = DeterministicActor(
            env.mu_obs_dim, action_dim=self.mu_cont_dim, assoc_dim=self.mu_assoc_dim
        ).to(self.device)
        self.uav_actor_target = DeterministicActor(env.uav_obs_dim, action_dim=self.uav_action_dim, assoc_dim=0).to(
            self.device
        )
        self.critic = CentralizedQCritic(self.state_dim, self.total_action_dim).to(self.device)
        self.critic_target = CentralizedQCritic(self.state_dim, self.total_action_dim).to(self.device)

        self.mu_actor_target.load_state_dict(self.mu_actor.state_dict())
        self.uav_actor_target.load_state_dict(self.uav_actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.mu_actor_optim = optim.Adam(self.mu_actor.parameters(), lr=cfg.MADDPG_ACTOR_LR)
        self.uav_actor_optim = optim.Adam(self.uav_actor.parameters(), lr=cfg.MADDPG_ACTOR_LR)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=cfg.MADDPG_CRITIC_LR)

        self.replay = ReplayBuffer(
            capacity=cfg.MADDPG_BUFFER_SIZE,
            state_dim=self.state_dim,
            num_mus=self.K,
            num_uavs=self.M,
            mu_obs_dim=env.mu_obs_dim,
            uav_obs_dim=env.uav_obs_dim,
            mu_repr_dim=self.mu_repr_dim,
            uav_act_dim=self.uav_action_dim,
        )
        self.noise_std = cfg.MADDPG_NOISE_STD_INIT

    def _actions_to_env(self, mu_repr, uav_action):
        assoc_prob = mu_repr[:, : self.mu_assoc_dim]
        offload = mu_repr[:, self.mu_assoc_dim]
        association = np.argmax(assoc_prob, axis=1).astype(np.int64)
        offload = np.clip(offload, 0.0, 1.0)
        return {"association": association, "offload_ratio": offload}, np.clip(uav_action, 0.0, 1.0)

    def _stack_joint_action(self, mu_repr_t: torch.Tensor, uav_action_t: torch.Tensor) -> torch.Tensor:
        b = mu_repr_t.shape[0]
        mu_flat = mu_repr_t.reshape(b, -1)
        uav_flat = uav_action_t.reshape(b, -1)
        return torch.cat([mu_flat, uav_flat], dim=-1)

    @torch.no_grad()
    def get_actions(self, observations: Dict, deterministic=False):
        mu_obs = torch.as_tensor(observations["mu_obs"], dtype=torch.float32, device=self.device)
        uav_obs = torch.as_tensor(observations["uav_obs"], dtype=torch.float32, device=self.device)

        mu_cont, mu_assoc = self.mu_actor(mu_obs)
        uav_cont, _ = self.uav_actor(uav_obs)

        mu_cont_np = mu_cont.cpu().numpy()
        mu_assoc_np = mu_assoc.cpu().numpy()
        uav_cont_np = uav_cont.cpu().numpy()

        if not deterministic:
            mu_cont_np = np.clip(mu_cont_np + np.random.normal(0.0, self.noise_std, size=mu_cont_np.shape), 0.0, 1.0)
            uav_cont_np = np.clip(uav_cont_np + np.random.normal(0.0, self.noise_std, size=uav_cont_np.shape), 0.0, 1.0)
            mu_assoc_np = np.clip(
                mu_assoc_np + np.random.normal(0.0, self.noise_std * 0.25, size=mu_assoc_np.shape),
                1e-6,
                1.0,
            )
            mu_assoc_np = mu_assoc_np / (mu_assoc_np.sum(axis=1, keepdims=True) + 1e-8)

        mu_repr = np.concatenate([mu_assoc_np, mu_cont_np], axis=1)
        mu_actions_env, uav_actions_env = self._actions_to_env(mu_repr, uav_cont_np)

        # Keep tuple layout compatible with MAPPO get_actions.
        return (
            mu_actions_env,
            uav_actions_env,
            mu_repr,
            uav_actions_env,
            np.zeros((self.K,), dtype=np.float32),
            np.zeros((self.M,), dtype=np.float32),
        )

    def collect_episode(self):
        obs = self.env.reset()

        ep_info = []
        mu_rewards_all = []
        uav_rewards_all = []

        for _ in range(cfg.EPISODE_LENGTH):
            state = self.env.get_state().copy()
            acts = self.get_actions(obs, deterministic=False)
            next_obs, rewards, done, info = self.env.step(acts[0], acts[1])
            next_state = self.env.get_state().copy()

            # Scalar cooperative reward for centralized critic.
            scalar_reward = float((np.mean(rewards["mu_rewards"]) + np.mean(rewards["uav_rewards"])) / 2.0)

            self.replay.add(
                state=state,
                mu_obs=obs["mu_obs"],
                uav_obs=obs["uav_obs"],
                mu_action_repr=acts[2],
                uav_action=acts[3],
                reward=scalar_reward,
                next_state=next_state,
                next_mu_obs=next_obs["mu_obs"],
                next_uav_obs=next_obs["uav_obs"],
                done=float(done),
            )

            ep_info.append(info)
            mu_rewards_all.append(float(np.mean(rewards["mu_rewards"])))
            uav_rewards_all.append(float(np.mean(rewards["uav_rewards"])))

            obs = next_obs
            if done:
                obs = self.env.reset()

        return {
            "mu_reward": float(np.mean(mu_rewards_all)),
            "uav_reward": float(np.mean(uav_rewards_all)),
            "total_cost": float(-np.mean(mu_rewards_all)),
            "weighted_energy": float(np.mean([i["weighted_energy"] for i in ep_info])),
            "weighted_energy_mu_avg": float(np.mean([i["weighted_energy_mu_avg"] for i in ep_info])),
            "weighted_energy_mu_total": float(np.mean([i["weighted_energy_mu_total"] for i in ep_info])),
            "mu_energy": float(np.mean([i["mu_energy_avg"] for i in ep_info])),
            "uav_energy": float(np.mean([i["uav_energy_avg"] for i in ep_info])),
            "mu_energy_avg": float(np.mean([i["mu_energy_avg"] for i in ep_info])),
            "uav_energy_avg": float(np.mean([i["uav_energy_avg"] for i in ep_info])),
            "jain_fairness": float(np.mean([i["jain_fairness"] for i in ep_info])),
            "delay_violation": float(np.mean([i["delay_violation_rate"] for i in ep_info])),
        }

    def _soft_update(self, src: nn.Module, dst: nn.Module, tau: float):
        for p, tp in zip(src.parameters(), dst.parameters()):
            tp.data.mul_(1.0 - tau)
            tp.data.add_(tau * p.data)

    def update(self):
        # Warmup is counted in replay transitions.
        warmup_transitions = max(cfg.MADDPG_WARMUP_STEPS, cfg.MADDPG_BATCH_SIZE)
        if self.replay.size < warmup_transitions:
            return {}

        updates = max(1, cfg.MADDPG_UPDATES_PER_STEP)
        actor_losses = []
        critic_losses = []

        for _ in range(updates):
            batch = self.replay.sample(cfg.MADDPG_BATCH_SIZE, self.device)

            action_flat = self._stack_joint_action(batch.mu_action_repr, batch.uav_action)
            q = self.critic(batch.state, action_flat)

            with torch.no_grad():
                b = batch.next_state.shape[0]

                nmu_cont, nmu_assoc = self.mu_actor_target(batch.next_mu_obs.reshape(b * self.K, -1))
                nmu_cont = nmu_cont.reshape(b, self.K, self.mu_cont_dim)
                nmu_assoc = nmu_assoc.reshape(b, self.K, self.mu_assoc_dim)
                nmu_repr = torch.cat([nmu_assoc, nmu_cont], dim=-1)

                nuav_cont, _ = self.uav_actor_target(batch.next_uav_obs.reshape(b * self.M, -1))
                nuav_cont = nuav_cont.reshape(b, self.M, self.uav_action_dim)

                next_action_flat = self._stack_joint_action(nmu_repr, nuav_cont)
                q_target_next = self.critic_target(batch.next_state, next_action_flat)
                q_target = batch.reward + cfg.MADDPG_GAMMA * (1.0 - batch.done) * q_target_next

            critic_loss = F.mse_loss(q, q_target)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.MAX_GRAD_NORM)
            self.critic_optim.step()

            b = batch.state.shape[0]
            mu_cont, mu_assoc = self.mu_actor(batch.mu_obs.reshape(b * self.K, -1))
            mu_cont = mu_cont.reshape(b, self.K, self.mu_cont_dim)
            mu_assoc = mu_assoc.reshape(b, self.K, self.mu_assoc_dim)
            mu_repr = torch.cat([mu_assoc, mu_cont], dim=-1)

            uav_cont, _ = self.uav_actor(batch.uav_obs.reshape(b * self.M, -1))
            uav_cont = uav_cont.reshape(b, self.M, self.uav_action_dim)

            actor_action_flat = self._stack_joint_action(mu_repr, uav_cont)
            actor_loss = -self.critic(batch.state, actor_action_flat).mean()

            self.mu_actor_optim.zero_grad()
            self.uav_actor_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.mu_actor.parameters(), cfg.MAX_GRAD_NORM)
            nn.utils.clip_grad_norm_(self.uav_actor.parameters(), cfg.MAX_GRAD_NORM)
            self.mu_actor_optim.step()
            self.uav_actor_optim.step()

            self._soft_update(self.mu_actor, self.mu_actor_target, cfg.MADDPG_TAU)
            self._soft_update(self.uav_actor, self.uav_actor_target, cfg.MADDPG_TAU)
            self._soft_update(self.critic, self.critic_target, cfg.MADDPG_TAU)

            actor_losses.append(float(actor_loss.item()))
            critic_losses.append(float(critic_loss.item()))

        self.noise_std = max(cfg.MADDPG_NOISE_STD_MIN, self.noise_std * cfg.MADDPG_NOISE_DECAY)

        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(self.noise_std),
        }

    def save(self, path):
        torch.save(
            {
                "mu_actor": self.mu_actor.state_dict(),
                "uav_actor": self.uav_actor.state_dict(),
                "critic": self.critic.state_dict(),
                "noise_std": self.noise_std,
            },
            path,
        )

    def load(self, path):
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)
        self.mu_actor.load_state_dict(ckpt["mu_actor"])
        self.uav_actor.load_state_dict(ckpt["uav_actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.mu_actor_target.load_state_dict(self.mu_actor.state_dict())
        self.uav_actor_target.load_state_dict(self.uav_actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.noise_std = float(ckpt.get("noise_std", cfg.MADDPG_NOISE_STD_INIT))
```

## mappo.py

```python
"""
AB-MAPPO 论文复现 — 核心算法 (精确版)
实现 Algorithm 1: AB-MAPPO

精确版改进:
  - UAV动作空间包含资源分配 (带宽+CPU)
  - 异构折扣因子 (γ_MU=0.8, γ_UAV=0.95)
  - 严格的CTDE: 注意力Critic用所有智能体观测
"""

import os
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple

import config as cfg
from environment import UAVMECEnv
from networks import BetaActor, GaussianActor, AttentionCritic, MLPCritic
from buffer import MultiAgentRolloutBuffer


class RunningMeanStd:
    """Running mean/std for reward normalization"""
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        batch_mean = np.mean(x)
        # Per-step normalization: treat the current timestep aggregate as one sample.
        batch_var = 0.0
        batch_count = 1.0
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

class ABMAPPO:
    def __init__(self, env: UAVMECEnv, algorithm='AB-MAPPO', device='cpu'):
        self.env = env
        self.algorithm = algorithm
        self.device = torch.device(device)
        self.K = env.K
        self.M = env.M

        # 动作维度
        self.mu_action_dim = 2   # [discrete(1), offload_ratio(1)]
        self.uav_action_dim = env.uav_continuous_dim  # 2 + 2K

        # 网络选择
        use_beta = algorithm in ['AB-MAPPO', 'B-MAPPO']
        use_attention = algorithm in ['AB-MAPPO', 'AG-MAPPO']
        ActorClass = BetaActor if use_beta else GaussianActor

        # MU Actor
        self.mu_actor = ActorClass(
            obs_dim=env.mu_obs_dim,
            continuous_dim=env.mu_continuous_dim,
            discrete_dim=env.mu_discrete_dim
        ).to(self.device)

        # UAV Actor (大动作空间: 2 + 2K)
        self.uav_actor = ActorClass(
            obs_dim=env.uav_obs_dim,
            continuous_dim=env.uav_continuous_dim,
            discrete_dim=0
        ).to(self.device)

        # Critic
        self.use_attention = use_attention
        if use_attention:
            self.critic_obs_dim = max(env.mu_obs_dim, env.uav_obs_dim)
            self.critic = AttentionCritic(
                obs_dim=self.critic_obs_dim,
                num_agents=self.K + self.M,
                num_heads=cfg.NUM_ATTENTION_HEADS
            ).to(self.device)
        else:
            self.critic_obs_dim = env.state_dim
            self.critic = MLPCritic(obs_dim=self.critic_obs_dim).to(self.device)

        if hasattr(torch, "compile") and self.device.type == "cuda" and os.name != "nt":
            try:
                # Compile critic first: this path is statically shaped and benefits from graph fusion.
                self.critic = torch.compile(self.critic)
            except Exception:
                pass

        self.use_amp = self.device.type == "cuda"

        # 优化器
        if algorithm not in ('Random', 'Randomized'):
            self.mu_actor_optimizer = optim.Adam(self.mu_actor.parameters(), lr=cfg.ACTOR_LR)
            self.uav_actor_optimizer = optim.Adam(self.uav_actor.parameters(), lr=cfg.ACTOR_LR)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.CRITIC_LR)
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except Exception:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # 缓冲区
        self.buffer = MultiAgentRolloutBuffer(
            num_mus=self.K, num_uavs=self.M,
            mu_obs_dim=env.mu_obs_dim, uav_obs_dim=env.uav_obs_dim,
            mu_action_dim=self.mu_action_dim,
            uav_action_dim=self.uav_action_dim,
            buffer_size=cfg.EPISODE_LENGTH,
            state_dim=env.state_dim,
        )

        # 奖励归一化
        self.mu_reward_rms = RunningMeanStd()
        self.uav_reward_rms = RunningMeanStd()
        # Optional expensive consistency check for debugging MLP critic input path.
        self.strict_state_sync_check = os.getenv("ABMAPPO_STRICT_STATE_CHECK", "0").lower() in {"1", "true", "yes"}

    def _autocast_ctx(self):
        if not self.use_amp:
            return nullcontext()
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast(device_type="cuda", enabled=True)
        return torch.cuda.amp.autocast(enabled=True)


    def _obs_to_tensors(self, observations: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        mu_obs = torch.as_tensor(observations['mu_obs'], dtype=torch.float32, device=self.device)
        uav_obs = torch.as_tensor(observations['uav_obs'], dtype=torch.float32, device=self.device)
        return mu_obs, uav_obs

    @torch.no_grad()
    def _actions_from_tensors(
        self, mu_obs: torch.Tensor, uav_obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # MU动作
        mu_actions, mu_log_probs, _ = self.mu_actor.get_action(mu_obs, deterministic)
        mu_actions_np = mu_actions.cpu().numpy()
        mu_log_probs_np = mu_log_probs.cpu().numpy()

        association = mu_actions_np[:, 0].astype(int)
        association = np.clip(association, 0, self.M)
        offload_ratio = np.clip(mu_actions_np[:, 1], 0, 1)

        # UAV动作 (2 + 2K 维)
        uav_actions, uav_log_probs, _ = self.uav_actor.get_action(uav_obs, deterministic)
        uav_actions_np = uav_actions.cpu().numpy()
        uav_log_probs_np = uav_log_probs.cpu().numpy()

        mu_actions_env = {
            'association': association,
            'offload_ratio': offload_ratio,
        }

        return (
            mu_actions_env,
            uav_actions_np,
            mu_actions_np,
            uav_actions_np,
            mu_log_probs_np,
            uav_log_probs_np,
        )

    @torch.no_grad()
    def _values_from_tensors(
        self,
        mu_obs: torch.Tensor,
        uav_obs: torch.Tensor,
        state: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.use_attention:
            pad_mu = self.critic_obs_dim - self.env.mu_obs_dim
            pad_uav = self.critic_obs_dim - self.env.uav_obs_dim
            if pad_mu > 0:
                mu_obs = F.pad(mu_obs, (0, pad_mu))
            if pad_uav > 0:
                uav_obs = F.pad(uav_obs, (0, pad_uav))

            all_obs = torch.cat([mu_obs, uav_obs], dim=0).unsqueeze(0)
            values = self.critic(all_obs).squeeze(0).squeeze(-1).cpu().numpy()
            return values[:self.K], values[self.K:]

        return self._values_from_state(state=state)

    @torch.no_grad()
    def _values_from_state(self, state: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
        state_np = self.env.get_state() if state is None else state
        state_t = torch.as_tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        v = self.critic(state_t).item()
        # MLP critic is a centralized scalar-value baseline; broadcast to all agents by design.
        return np.full(self.K, v), np.full(self.M, v)

    @torch.no_grad()
    def get_actions(self, observations, deterministic=False):
        if self.algorithm in ('Random', 'Randomized'):
            return self._get_random_actions()

        mu_obs, uav_obs = self._obs_to_tensors(observations)
        return self._actions_from_tensors(mu_obs, uav_obs, deterministic=deterministic)

    def _get_random_actions(self):
        association = np.random.randint(0, self.M + 1, size=self.K)
        offload_ratio = np.random.uniform(0, 1, size=self.K)
        uav_actions = np.random.uniform(0, 1, size=(self.M, self.uav_action_dim))
        mu_actions_env = {'association': association, 'offload_ratio': offload_ratio}
        mu_actions_store = np.stack([association.astype(float), offload_ratio], axis=1)
        return (mu_actions_env, uav_actions,
                mu_actions_store, uav_actions,
                np.zeros(self.K), np.zeros(self.M))

    @torch.no_grad()
    def get_values(self, observations):
        if not self.use_attention:
            return self._values_from_state()
        mu_obs, uav_obs = self._obs_to_tensors(observations)
        return self._values_from_tensors(mu_obs, uav_obs)

    @torch.no_grad()
    def get_actions_and_values(self, observations, deterministic=False, state=None):
        if self.algorithm in ('Random', 'Randomized'):
            mu_val, uav_val = self._values_from_state(state=state) if not self.use_attention else self.get_values(observations)
            return (*self._get_random_actions(), mu_val, uav_val)

        mu_obs, uav_obs = self._obs_to_tensors(observations)
        actions = self._actions_from_tensors(mu_obs, uav_obs, deterministic=deterministic)
        mu_val, uav_val = self._values_from_tensors(mu_obs, uav_obs, state=state)
        return (*actions, mu_val, uav_val)

    def collect_episode(self):
        self.buffer.reset()
        obs = self.env.reset()
        steps = 0
        raw_mu_reward_sum = 0.0
        raw_uav_reward_sum = 0.0
        weighted_energy_sum = 0.0
        weighted_energy_mu_avg_sum = 0.0
        weighted_energy_mu_total_sum = 0.0
        mu_energy_avg_sum = 0.0
        uav_energy_avg_sum = 0.0
        jain_fairness_sum = 0.0
        delay_violation_sum = 0.0

        for t in range(cfg.EPISODE_LENGTH):
            state = self.env.get_state()
            (mu_act_env, uav_act_env, mu_act_store, uav_act_store,
             mu_lp, uav_lp, mu_val, uav_val) = self.get_actions_and_values(obs, state=state)

            next_obs, rewards, done, info = self.env.step(mu_act_env, uav_act_env)

            # 归一化奖励
            mu_r = rewards['mu_rewards']
            uav_r = rewards['uav_rewards']
            self.mu_reward_rms.update(mu_r)
            self.uav_reward_rms.update(uav_r)
            mu_r_norm = self.mu_reward_rms.normalize(mu_r)
            uav_r_norm = self.uav_reward_rms.normalize(uav_r)

            self.buffer.add(
                mu_obs=obs['mu_obs'], mu_action=mu_act_store,
                mu_log_prob=mu_lp, mu_reward=mu_r_norm, mu_value=mu_val,
                uav_obs=obs['uav_obs'], uav_action=uav_act_store,
                uav_log_prob=uav_lp, uav_reward=uav_r_norm, uav_value=uav_val,
                done=float(done),
                state=state,
            )
            raw_mu_reward_sum += float(mu_r.mean())
            raw_uav_reward_sum += float(uav_r.mean())
            weighted_energy_sum += float(info['weighted_energy'])
            weighted_energy_mu_avg_sum += float(info['weighted_energy_mu_avg'])
            weighted_energy_mu_total_sum += float(info['weighted_energy_mu_total'])
            mu_energy_avg_sum += float(info['mu_energy_avg'])
            uav_energy_avg_sum += float(info['uav_energy_avg'])
            jain_fairness_sum += float(info['jain_fairness'])
            delay_violation_sum += float(info['delay_violation_rate'])
            steps += 1
            obs = next_obs
            if done:
                obs = self.env.reset()

        mu_last, uav_last = self.get_values(obs)
        self.buffer.compute_gae(mu_last, uav_last)

        denom = max(steps, 1)
        raw_mu_r = raw_mu_reward_sum / denom
        raw_uav_r = raw_uav_reward_sum / denom

        return {
            'mu_reward': raw_mu_r,
            'uav_reward': raw_uav_r,
            'total_cost': -(raw_mu_r + raw_uav_r) / 2,
            'weighted_energy': weighted_energy_sum / denom,
            'weighted_energy_mu_avg': weighted_energy_mu_avg_sum / denom,
            'weighted_energy_mu_total': weighted_energy_mu_total_sum / denom,
            'mu_energy': mu_energy_avg_sum / denom,
            'uav_energy': uav_energy_avg_sum / denom,
            'mu_energy_avg': mu_energy_avg_sum / denom,
            'uav_energy_avg': uav_energy_avg_sum / denom,
            'jain_fairness': jain_fairness_sum / denom,
            'delay_violation': delay_violation_sum / denom,
        }


    def update(self):
        if self.algorithm in ('Random', 'Randomized'):
            return {}

        batches = self.buffer.get_batches()
        mu_data = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batches['mu'].items()}
        uav_data = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batches['uav'].items()}
        all_obs_t = None
        all_ret_t = None
        if self.use_attention:
            mu_obs_t = mu_data['observations']
            uav_obs_t = uav_data['observations']
            t_steps = mu_obs_t.shape[0]
            num_agents = self.K + self.M
            all_obs_t = torch.zeros(t_steps, num_agents, self.critic_obs_dim, device=self.device)
            all_obs_t[:, :self.K, :self.env.mu_obs_dim] = mu_obs_t
            all_obs_t[:, self.K:, :self.env.uav_obs_dim] = uav_obs_t
            all_ret_t = torch.cat([mu_data['returns'], uav_data['returns']], dim=1)

        total_al, total_cl, total_ent = 0, 0, 0

        for epoch in range(cfg.PPO_EPOCHS):
            # ---- MU Actor ----
            mu_obs = mu_data['observations']
            mu_acts = mu_data['actions']
            mu_old_lp = mu_data['log_probs']
            mu_adv = mu_data['advantages']

            with self._autocast_ctx():
                T, K = mu_obs.shape[0], mu_obs.shape[1]
                new_lp, ent = self.mu_actor.evaluate_action(
                    mu_obs.reshape(T * K, -1), mu_acts.reshape(T * K, -1)
                )
                new_lp = new_lp.reshape(T, K)
                ent = ent.reshape(T, K)

                # Baseline choice: normalize advantages jointly across time and agents.
                # This keeps optimization simple; per-agent normalization can be enabled in future ablations.
                adv = (mu_adv - mu_adv.mean()) / (mu_adv.std() + 1e-8)
                ratio = torch.exp(new_lp - mu_old_lp)
                s1 = ratio * adv
                s2 = torch.clamp(ratio, 1 - cfg.PPO_CLIP_EPSILON, 1 + cfg.PPO_CLIP_EPSILON) * adv
                al_mu = -torch.min(s1, s2).mean()
                loss_mu = al_mu - cfg.ENTROPY_COEFF * ent.mean()

            self.mu_actor_optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss_mu).backward()
            self.scaler.unscale_(self.mu_actor_optimizer)
            nn.utils.clip_grad_norm_(self.mu_actor.parameters(), cfg.MAX_GRAD_NORM)
            self.scaler.step(self.mu_actor_optimizer)

            # ---- UAV Actor ----
            uav_obs = uav_data['observations']
            uav_acts = uav_data['actions']
            uav_old_lp = uav_data['log_probs']
            uav_adv = uav_data['advantages']

            with self._autocast_ctx():
                T_u, M = uav_obs.shape[0], uav_obs.shape[1]
                new_lp_u, ent_u = self.uav_actor.evaluate_action(
                    uav_obs.reshape(T_u * M, -1), uav_acts.reshape(T_u * M, -1)
                )
                new_lp_u = new_lp_u.reshape(T_u, M)
                ent_u = ent_u.reshape(T_u, M)

                # Same normalization policy for UAV branch: joint standardization over batch.
                adv_u = (uav_adv - uav_adv.mean()) / (uav_adv.std() + 1e-8)
                ratio_u = torch.exp(new_lp_u - uav_old_lp)
                s1_u = ratio_u * adv_u
                s2_u = torch.clamp(ratio_u, 1 - cfg.PPO_CLIP_EPSILON, 1 + cfg.PPO_CLIP_EPSILON) * adv_u
                al_uav = -torch.min(s1_u, s2_u).mean()
                loss_uav = al_uav - cfg.ENTROPY_COEFF * ent_u.mean()

            self.uav_actor_optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss_uav).backward()
            self.scaler.unscale_(self.uav_actor_optimizer)
            nn.utils.clip_grad_norm_(self.uav_actor.parameters(), cfg.MAX_GRAD_NORM)
            self.scaler.step(self.uav_actor_optimizer)

            # ---- Critic ----
            if self.use_attention:
                cl = self._update_attention_critic(all_obs_t, all_ret_t)
            else:
                cl = self._update_mlp_critic(mu_data, uav_data)

            self.scaler.update()

            total_al += (al_mu.item() + al_uav.item()) / 2
            total_cl += cl
            total_ent += (ent.mean().item() + ent_u.mean().item()) / 2

        n = cfg.PPO_EPOCHS
        return {'actor_loss': total_al/n, 'critic_loss': total_cl/n, 'entropy': total_ent/n}

    def _update_attention_critic(self, all_obs, all_ret):
        with self._autocast_ctx():
            values = self.critic(all_obs).squeeze(-1)
            loss = 0.5 * ((values - all_ret) ** 2).mean()

        self.critic_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.critic_optimizer)
        nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.MAX_GRAD_NORM)
        self.scaler.step(self.critic_optimizer)
        return float(loss.item())

    def _update_mlp_critic(self, mu_data, uav_data):
        mu_ret = mu_data['returns'].mean(dim=1, keepdim=True)
        uav_ret = uav_data['returns'].mean(dim=1, keepdim=True)
        avg_ret = (mu_ret + uav_ret) / 2

        if 'states' not in mu_data or 'states' not in uav_data:
            raise KeyError("MLP critic update requires `states` in both MU and UAV batches")

        mu_states = mu_data['states']
        uav_states = uav_data['states']

        if mu_states.shape != uav_states.shape:
            raise ValueError(f"State batch shape mismatch: MU={mu_states.shape}, UAV={uav_states.shape}")
        if mu_states.shape[1] != self.critic_obs_dim:
            raise ValueError(
                f"State dim mismatch for MLP critic: expected {self.critic_obs_dim}, got {mu_states.shape[1]}"
            )
        if self.strict_state_sync_check and not torch.allclose(mu_states, uav_states):
            raise ValueError("MU and UAV buffered states must be identical for centralized MLP critic")

        state = mu_states

        with self._autocast_ctx():
            values = self.critic(state)
            loss = 0.5 * ((values - avg_ret) ** 2).mean()

        self.critic_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.critic_optimizer)
        nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.MAX_GRAD_NORM)
        self.scaler.step(self.critic_optimizer)
        return float(loss.item())

    def save(self, path):
        torch.save({
            'mu_actor': self.mu_actor.state_dict(),
            'uav_actor': self.uav_actor.state_dict(),
            'critic': self.critic.state_dict(),
            'algorithm': self.algorithm,
        }, path)

    def load(self, path):
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)
        self.mu_actor.load_state_dict(ckpt['mu_actor'])
        self.uav_actor.load_state_dict(ckpt['uav_actor'])
        self.critic.load_state_dict(ckpt['critic'])
```

## networks.py

```python
"""
AB-MAPPO 论文复现 — 神经网络模块
实现 Section III-C:
  - BetaActor: 使用Beta分布输出连续动作 (论文的 "B")
  - AttentionCritic: 使用多头注意力机制的Critic (论文的 "A")
  - GaussianActor: 基线对比用
  - MLPCritic: 基线对比用 (无注意力)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Categorical, Normal
import config as cfg


class BetaActor(nn.Module):
    """
    Beta分布Actor网络 (论文 "B" 部分)

    输出 Beta(α, β) 分布的参数, 动作天然在 [0,1] 区间
    对于离散动作(UAV关联), 使用 Categorical 分布

    网络结构: obs -> MLP(hidden) -> ReLU -> MLP(hidden) -> ReLU -> (α, β) 或 logits
    """

    def __init__(self, obs_dim, continuous_dim, discrete_dim=0):
        super().__init__()
        self.continuous_dim = continuous_dim
        self.discrete_dim = discrete_dim
        hidden = cfg.HIDDEN_SIZE

        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Beta分布参数 (α, β) — 连续动作
        if continuous_dim > 0:
            self.alpha_head = nn.Linear(hidden, continuous_dim)
            self.beta_head = nn.Linear(hidden, continuous_dim)

        # Categorical分布 logits — 离散动作
        if discrete_dim > 0:
            self.discrete_head = nn.Linear(hidden, discrete_dim)

    def forward(self, obs):
        """
        Args:
            obs: (batch, obs_dim)
        Returns:
            distributions dict
        """
        feat = self.feature(obs)
        result = {}

        if self.continuous_dim > 0:
            # α, β > 1 确保单峰分布，使用 softplus + 1
            alpha = F.softplus(self.alpha_head(feat)) + 1.0
            beta = F.softplus(self.beta_head(feat)) + 1.0
            result['continuous_dist'] = Beta(alpha, beta)

        if self.discrete_dim > 0:
            logits = self.discrete_head(feat)
            result['discrete_dist'] = Categorical(logits=logits)

        return result

    def get_action(self, obs, deterministic=False):
        """采样动作并返回log概率"""
        dists = self.forward(obs)
        log_probs = []
        actions = []

        if 'discrete_dist' in dists:
            dist = dists['discrete_dist']
            if deterministic:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()
            log_probs.append(dist.log_prob(action))
            actions.append(action.unsqueeze(-1).float())

        if 'continuous_dist' in dists:
            dist = dists['continuous_dist']
            if deterministic:
                action = dist.mean
            else:
                action = dist.sample()
            # 裁剪避免边界问题
            action = torch.clamp(action, 1e-6, 1 - 1e-6)
            log_probs.append(dist.log_prob(action).sum(dim=-1))
            actions.append(action)

        total_log_prob = sum(log_probs)
        action_concat = torch.cat(actions, dim=-1) if len(actions) > 1 else actions[0]
        return action_concat, total_log_prob, dists

    def evaluate_action(self, obs, action):
        """评估给定观测下动作的log概率和熵"""
        dists = self.forward(obs)
        log_probs = []
        entropies = []

        col = 0
        if 'discrete_dist' in dists:
            dist = dists['discrete_dist']
            discrete_action = action[:, col].long()
            log_probs.append(dist.log_prob(discrete_action))
            entropies.append(dist.entropy())
            col += 1

        if 'continuous_dist' in dists:
            dist = dists['continuous_dist']
            continuous_action = action[:, col:col + self.continuous_dim]
            continuous_action = torch.clamp(continuous_action, 1e-6, 1 - 1e-6)
            log_probs.append(dist.log_prob(continuous_action).sum(dim=-1))
            entropies.append(dist.entropy().sum(dim=-1))

        return sum(log_probs), sum(entropies)


class GaussianActor(nn.Module):
    """
    高斯分布Actor (AG-MAPPO基线)
    输出 N(μ, σ²) 分布, 动作通过 sigmoid 映射到 [0,1]
    """

    def __init__(self, obs_dim, continuous_dim, discrete_dim=0):
        super().__init__()
        self.continuous_dim = continuous_dim
        self.discrete_dim = discrete_dim
        hidden = cfg.HIDDEN_SIZE

        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        if continuous_dim > 0:
            self.mean_head = nn.Linear(hidden, continuous_dim)
            self.log_std = nn.Parameter(torch.zeros(continuous_dim))

        if discrete_dim > 0:
            self.discrete_head = nn.Linear(hidden, discrete_dim)

    def forward(self, obs):
        feat = self.feature(obs)
        result = {}
        if self.continuous_dim > 0:
            mean = self.mean_head(feat)
            std = torch.exp(self.log_std.clamp(-5, 2))
            result['continuous_dist'] = Normal(mean, std)
        if self.discrete_dim > 0:
            logits = self.discrete_head(feat)
            result['discrete_dist'] = Categorical(logits=logits)
        return result

    def get_action(self, obs, deterministic=False):
        dists = self.forward(obs)
        log_probs = []
        actions = []

        if 'discrete_dist' in dists:
            dist = dists['discrete_dist']
            action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
            log_probs.append(dist.log_prob(action))
            actions.append(action.unsqueeze(-1).float())

        if 'continuous_dist' in dists:
            dist = dists['continuous_dist']
            if deterministic:
                raw_action = dist.mean
            else:
                raw_action = dist.rsample()
            action = torch.sigmoid(raw_action)  # 映射到 [0,1]
            # 修正log_prob (Jacobian)
            lp = dist.log_prob(raw_action).sum(dim=-1)
            lp -= torch.log(action * (1 - action) + 1e-8).sum(dim=-1)
            log_probs.append(lp)
            actions.append(action)

        total_log_prob = sum(log_probs)
        action_concat = torch.cat(actions, dim=-1) if len(actions) > 1 else actions[0]
        return action_concat, total_log_prob, dists

    def evaluate_action(self, obs, action):
        dists = self.forward(obs)
        log_probs = []
        entropies = []

        col = 0
        if 'discrete_dist' in dists:
            dist = dists['discrete_dist']
            discrete_action = action[:, col].long()
            log_probs.append(dist.log_prob(discrete_action))
            entropies.append(dist.entropy())
            col += 1

        if 'continuous_dist' in dists:
            dist = dists['continuous_dist']
            cont_action = action[:, col:col + self.continuous_dim].clamp(1e-6, 1 - 1e-6)
            raw_action = torch.logit(cont_action)
            lp = dist.log_prob(raw_action).sum(dim=-1)
            lp -= torch.log(cont_action * (1 - cont_action) + 1e-8).sum(dim=-1)
            log_probs.append(lp)
            entropies.append(dist.entropy().sum(dim=-1))

        return sum(log_probs), sum(entropies)


class AttentionCritic(nn.Module):
    """
    注意力机制Critic (论文 "A" 部分, 公式49-50)

    结构:
      1. 每个智能体的观测 → Encoder MLP → 特征向量 e_i
      2. 多头注意力: Q=W_q*e_i, K=W_key*e_j, V=W_v*e_j
         α_{i,j} = softmax(K^T * Q / √d_key)
         x_i = Σ α_{i,j} * V_j
      3. [x_i, o_i] → MLP → V(s)
    """

    def __init__(self, obs_dim, num_agents, num_heads=None):
        super().__init__()
        hidden = cfg.HIDDEN_SIZE
        self.num_agents = num_agents
        self.num_heads = num_heads or cfg.NUM_ATTENTION_HEADS
        self.head_dim = hidden // self.num_heads

        # 观测编码器
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # 注意力 Q, K, V 变换 (公式49-50)
        self.W_query = nn.Linear(hidden, hidden, bias=False)
        self.W_key = nn.Linear(hidden, hidden, bias=False)
        self.W_value = nn.Linear(hidden, hidden, bias=False)

        # 输出MLP: [attention_output + obs_feature] -> V(s)
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, all_obs, agent_idx=None):
        """
        Args:
            all_obs: (batch, num_agents, obs_dim) 所有智能体的观测
            agent_idx: 如果指定, 只返回该智能体的Value; 否则返回所有

        Returns:
            values: (batch, num_agents, 1) 或 (batch, 1)
        """
        batch_size = all_obs.shape[0]
        N = self.num_agents

        # 编码所有观测
        obs_flat = all_obs.reshape(batch_size * N, -1)
        features = self.encoder(obs_flat)  # (B*N, hidden)
        features = features.reshape(batch_size, N, -1)  # (B, N, hidden)

        # 多头注意力
        Q = self.W_query(features)  # (B, N, hidden)
        K = self.W_key(features)
        V = self.W_value(features)

        # 重塑为多头 (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        Q = Q.reshape(batch_size, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(batch_size, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(batch_size, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 优先使用 PyTorch 融合注意力算子；旧版本回退到手写实现。
        if hasattr(F, "scaled_dot_product_attention"):
            attn_output = F.scaled_dot_product_attention(
                Q.contiguous(),
                K.contiguous(),
                V.contiguous(),
                dropout_p=0.0,
            )
        else:
            scale = float(self.head_dim) ** 0.5
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, heads, N, N)
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_weights, V)  # (B, heads, N, head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, N, -1)  # (B, N, hidden)

        # 拼接注意力输出和原始特征
        combined = torch.cat([attn_output, features], dim=-1)  # (B, N, hidden*2)

        # 输出Value
        values = self.output_mlp(combined)  # (B, N, 1)

        if agent_idx is not None:
            return values[:, agent_idx, :]  # (B, 1)

        return values


class MLPCritic(nn.Module):
    """无注意力的普通Critic (B-MAPPO基线)"""

    def __init__(self, obs_dim, num_agents=None):
        super().__init__()
        hidden = cfg.HIDDEN_SIZE
        # 输入: 全局状态或拼接观测
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, agent_idx=None):
        """
        Args:
            state: (batch, state_dim) 全局状态
        Returns:
            value: (batch, 1)
        """
        return self.net(state)


class DeterministicActor(nn.Module):
    """
    Deterministic actor for MADDPG.

    Outputs:
    - optional association logits (for MU actor)
    - bounded continuous actions in [0, 1]
    """

    def __init__(self, obs_dim, action_dim, assoc_dim=0):
        super().__init__()
        hidden = cfg.HIDDEN_SIZE
        self.action_dim = action_dim
        self.assoc_dim = assoc_dim

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden, action_dim)
        self.assoc_head = nn.Linear(hidden, assoc_dim) if assoc_dim > 0 else None

    def forward(self, obs):
        feat = self.backbone(obs)
        continuous = torch.sigmoid(self.action_head(feat))
        if self.assoc_head is None:
            return continuous, None
        assoc_logits = self.assoc_head(feat)
        assoc_prob = F.softmax(assoc_logits, dim=-1)
        return continuous, assoc_prob


class CentralizedQCritic(nn.Module):
    """Centralized Q critic used by MADDPG."""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        hidden = max(256, cfg.HIDDEN_SIZE * 2)
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
```

## run_all.py

```python
"""
Staged pipeline runner:
- smoke: fast correctness pass
- full: paper-scale full sweep
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import subprocess
import sys
import time
import uuid

from train_sweep import RunOptions, build_run_specs


FULL_SEEDS = "42,43,44"
FULL_TOTAL_STEPS = 80000
FULL_EPISODE_LENGTH = 300


def _run(cmd):
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _safe_job_name(name):
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(name))


def _build_full_log_dir(base_dir="run_logs"):
    os.makedirs(base_dir, exist_ok=True)
    run_id = f"full_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    log_dir = os.path.join(base_dir, run_id)
    os.makedirs(log_dir, exist_ok=False)
    return log_dir


def _spawn_logged_process(job_name, cmd, log_dir, threads_per_proc=None):
    safe_name = _safe_job_name(job_name)
    log_file = os.path.join(log_dir, f"{safe_name}.log")
    print("[cmd]", " ".join(cmd))
    print("[log]", log_file)
    stream = open(log_file, "w", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if threads_per_proc is not None:
        thread_env = str(int(max(1, threads_per_proc)))
        env["OMP_NUM_THREADS"] = thread_env
        env["MKL_NUM_THREADS"] = thread_env
        env["OPENBLAS_NUM_THREADS"] = thread_env
        env["NUMEXPR_NUM_THREADS"] = thread_env
    process = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, env=env)
    return {
        "name": str(job_name),
        "cmd": list(cmd),
        "log_file": log_file,
        "stream": stream,
        "process": process,
        "start_time": time.monotonic(),
    }


def _close_job_stream(job):
    stream = job.get("stream")
    if stream is not None and not stream.closed:
        stream.close()


def _terminate_job(job, grace_sec=3.0):
    process = job["process"]
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=grace_sec)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _shutdown_active_jobs(active_jobs):
    for job in list(active_jobs):
        try:
            _terminate_job(job)
        finally:
            _close_job_stream(job)


def _run_parallel_commands(
    job_specs,
    max_parallel,
    log_dir,
    job_timeout_sec=0.0,
    poll_interval_sec=0.2,
    threads_per_proc=None,
):
    pending = list(job_specs)
    active = []

    while pending or active:
        while pending and len(active) < max_parallel:
            name, cmd = pending.pop(0)
            active.append(_spawn_logged_process(name, cmd, log_dir, threads_per_proc=threads_per_proc))

        if not active:
            continue

        time.sleep(max(0.01, poll_interval_sec))

        for job in list(active):
            process = job["process"]
            elapsed = time.monotonic() - job["start_time"]

            if job_timeout_sec > 0 and process.poll() is None and elapsed > job_timeout_sec:
                _terminate_job(job)
                _close_job_stream(job)
                active.remove(job)
                _shutdown_active_jobs(active)
                raise RuntimeError(
                    f"Job timeout after {job_timeout_sec}s: {job['name']}; "
                    f"log={job['log_file']}"
                )

            return_code = process.poll()
            if return_code is None:
                continue

            _close_job_stream(job)
            active.remove(job)

            if return_code != 0:
                _shutdown_active_jobs(active)
                raise RuntimeError(
                    f"Job failed with code {return_code}: {job['name']} "
                    f"({' '.join(job['cmd'])}); log={job['log_file']}"
                )


def run_smoke():
    _run(
        [
            sys.executable,
            "train_sweep.py",
            "--fig",
            "all",
            "--seeds",
            "42",
            "--total_steps",
            "4000",
            "--episode_length",
            "120",
            "--smoke",
            "--disable_tensorboard",
        ]
    )
    _run([sys.executable, "generate_figures.py", "--figs", "all"])


def _build_full_cmd(fig, full_device):
    return [
        sys.executable,
        "train_sweep.py",
        "--fig",
        fig,
        "--seeds",
        "42,43,44",
        "--total_steps",
        "80000",
        "--episode_length",
        "300",
        "--resume",
        "--disable_tensorboard",
        "--device",
        full_device,
    ]


def _is_cuda_device(device):
    return "cuda" in str(device).lower()


def _effective_parallel(max_parallel, full_device):
    return 1 if _is_cuda_device(full_device) else max(1, int(max_parallel))


def _summary_matches(summary_path, total_steps, episode_length):
    if not os.path.exists(summary_path):
        return False
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception:
        return False
    same_steps = int(summary.get("total_steps", -1)) == int(total_steps)
    same_epl = int(summary.get("episode_length", -1)) == int(episode_length)
    return same_steps and same_epl


def _append_override_arg(cmd, key, value):
    if value is None:
        return
    if key == "wo_dt_noise_mode":
        if bool(value):
            cmd.append("--wo_dt_noise_mode")
        return
    cmd.extend([f"--{key}", str(value)])


def _build_train_cmd_from_spec(run_spec, full_device, total_steps, episode_length):
    cmd = [
        sys.executable,
        "train.py",
        "--algorithm",
        str(run_spec["algorithm"]),
        "--seed",
        str(run_spec["seed"]),
        "--device",
        str(full_device),
        "--total_steps",
        str(total_steps),
        "--episode_length",
        str(episode_length),
        "--run_dir",
        str(run_spec["run_dir"]),
        "--disable_tensorboard",
    ]
    for key, value in run_spec.get("cli_overrides", {}).items():
        _append_override_arg(cmd, key, value)
    return cmd


def _build_run_job_specs(full_device, total_steps, episode_length):
    options = RunOptions(
        seeds=[42, 43, 44],
        device=full_device,
        total_steps=total_steps,
        episode_length=episode_length,
        resume=True,
        skip_existing=False,
        disable_tensorboard=True,
        smoke=False,
    )
    all_specs = build_run_specs("all", options)
    pending_specs = []

    for spec in all_specs:
        if _summary_matches(spec["summary_path"], total_steps, episode_length):
            print(
                f"[skip] fig={spec['fig']} algo={spec['algorithm']} "
                f"{spec['setting_name']} seed={spec['seed']}"
            )
            continue
        pending_specs.append(spec)

    pending_specs.sort(
        key=lambda s: (int(s.get("num_mus", 0)) * int(s.get("num_uavs", 0))),
        reverse=True,
    )

    job_specs = []
    for spec in pending_specs:
        name = f"fig{spec['fig']}_{spec['algorithm']}_{spec['setting_name']}_seed{spec['seed']}"
        cmd = _build_train_cmd_from_spec(spec, full_device, total_steps, episode_length)
        job_specs.append((name, cmd))
    return job_specs


def run_full(max_parallel=3, full_device="cpu", job_timeout_sec=0.0):
    log_dir = _build_full_log_dir("run_logs")
    effective_parallel = _effective_parallel(max_parallel, full_device)
    cpu_count = os.cpu_count() or 1
    threads_per_proc = max(1, cpu_count // max(1, effective_parallel))
    job_specs = _build_run_job_specs(
        full_device=full_device,
        total_steps=FULL_TOTAL_STEPS,
        episode_length=FULL_EPISODE_LENGTH,
    )

    print("[run_logs]", log_dir)
    print(
        f"[full] device={full_device} requested_parallel={max_parallel} "
        f"effective_parallel={effective_parallel} threads_per_proc={threads_per_proc} "
        f"pending_jobs={len(job_specs)}"
    )
    if job_specs:
        _run_parallel_commands(
            job_specs,
            max_parallel=effective_parallel,
            log_dir=log_dir,
            job_timeout_sec=job_timeout_sec,
            threads_per_proc=threads_per_proc,
        )
    else:
        print("[full] no pending training jobs, continue to aggregation")

    _run(
        [
            sys.executable,
            "train_sweep.py",
            "--fig",
            "all",
            "--seeds",
            FULL_SEEDS,
            "--total_steps",
            str(FULL_TOTAL_STEPS),
            "--episode_length",
            str(FULL_EPISODE_LENGTH),
            "--resume",
            "--skip_existing",
            "--disable_tensorboard",
            "--device",
            full_device,
        ]
    )

    _run([sys.executable, "generate_figures.py", "--figs", "all"])


def parse_args():
    p = argparse.ArgumentParser(description="Run full paper reproduction pipeline")
    p.add_argument("--stage", choices=["smoke", "full", "all"], default="smoke")
    p.add_argument("--max_parallel", type=int, default=3, help="parallel training jobs in full stage")
    p.add_argument("--full-device", type=str, default="cpu", help="device for full stage train_sweep runs")
    p.add_argument(
        "--job-timeout-sec",
        type=float,
        default=0.0,
        help="timeout per full-stage job in seconds (0 disables timeout)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.stage == "smoke":
        run_smoke()
    elif args.stage == "full":
        run_full(
            max_parallel=max(1, args.max_parallel),
            full_device=args.full_device,
            job_timeout_sec=max(0.0, float(args.job_timeout_sec)),
        )
    else:
        run_smoke()
        run_full(
            max_parallel=max(1, args.max_parallel),
            full_device=args.full_device,
            job_timeout_sec=max(0.0, float(args.job_timeout_sec)),
        )


if __name__ == "__main__":
    main()
```

## train.py

```python
"""
Unified training entrypoint for AB-MAPPO paper reproduction.

Supported algorithms:
- AB-MAPPO
- B-MAPPO
- AG-MAPPO
- MADDPG
- Randomized
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import torch

import config as cfg
from environment import UAVMECEnv
from maddpg import MADDPG
from mappo import ABMAPPO


def parse_args():
    parser = argparse.ArgumentParser(description="AB-MAPPO Paper Reproduction Training")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="AB-MAPPO",
        choices=["AB-MAPPO", "B-MAPPO", "AG-MAPPO", "MADDPG", "Randomized", "Random"],
    )
    parser.add_argument("--num_mus", type=int, default=cfg.NUM_MUS)
    parser.add_argument("--num_uavs", type=int, default=cfg.NUM_UAVS)
    parser.add_argument("--total_steps", type=int, default=cfg.TOTAL_STEPS)
    parser.add_argument("--episode_length", type=int, default=cfg.EPISODE_LENGTH)
    parser.add_argument("--device", type=str, default="auto", help="cpu / cuda / auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dt_deviation", type=float, default=cfg.DT_DEVIATION_RATE)
    parser.add_argument("--wo_dt_noise_mode", action="store_true")

    # optional config overrides
    parser.add_argument("--bandwidth_mhz", type=float, default=None)
    parser.add_argument("--weight_factor", type=float, default=None)
    parser.add_argument("--mu_max_cpu_ghz", type=float, default=None)
    parser.add_argument("--uav_max_cpu_ghz", type=float, default=None)
    parser.add_argument("--task_data_min_mbits", type=float, default=None)
    parser.add_argument("--task_data_max_mbits", type=float, default=None)
    parser.add_argument("--area_width", type=float, default=None)
    parser.add_argument("--area_height", type=float, default=None)

    parser.add_argument("--save_dir", type=str, default=cfg.SAVE_DIR)
    parser.add_argument("--log_dir", type=str, default=cfg.LOG_DIR)
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Per-run output directory. If set, writes history.npz + summary.json + checkpoints/ here.",
    )
    parser.add_argument("--disable_tensorboard", action="store_true")
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


def setup_device(device_str):
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def _canonical_algo_name(algo):
    return "Randomized" if algo == "Random" else algo


def _set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_cfg_overrides(args):
    overrides = {"EPISODE_LENGTH": args.episode_length}
    if args.bandwidth_mhz is not None:
        overrides["BANDWIDTH"] = args.bandwidth_mhz * 1e6
    if args.weight_factor is not None:
        overrides["WEIGHT_FACTOR"] = args.weight_factor
    if args.mu_max_cpu_ghz is not None:
        overrides["MU_MAX_CPU_FREQ"] = args.mu_max_cpu_ghz * 1e9
    if args.uav_max_cpu_ghz is not None:
        overrides["UAV_MAX_CPU_FREQ"] = args.uav_max_cpu_ghz * 1e9
    if args.task_data_min_mbits is not None:
        overrides["TASK_DATA_MIN"] = args.task_data_min_mbits * 1e6
    if args.task_data_max_mbits is not None:
        overrides["TASK_DATA_MAX"] = args.task_data_max_mbits * 1e6
    if args.area_width is not None:
        overrides["AREA_WIDTH"] = args.area_width
    if args.area_height is not None:
        overrides["AREA_HEIGHT"] = args.area_height
    return overrides


@contextmanager
def _temporary_cfg(overrides):
    old = {}
    for key, value in overrides.items():
        old[key] = getattr(cfg, key)
        setattr(cfg, key, value)
    try:
        yield
    finally:
        for key, value in old.items():
            setattr(cfg, key, value)


def _make_agent(env, algorithm, device):
    if algorithm == "MADDPG":
        return MADDPG(env, device=device)
    if algorithm == "Randomized":
        return ABMAPPO(env, algorithm="Randomized", device=device)
    return ABMAPPO(env, algorithm=algorithm, device=device)


def _history_schema():
    return {
        "episode": [],
        "step": [],
        "mu_reward": [],
        "uav_reward": [],
        "total_cost": [],
        "weighted_energy": [],
        "weighted_energy_mu_avg": [],
        "weighted_energy_mu_total": [],
        "mu_energy": [],
        "uav_energy": [],
        "jain_fairness": [],
        "delay_violation": [],
        "actor_loss": [],
        "critic_loss": [],
        "entropy": [],
    }


def _to_numpy_history(history):
    return {k: np.asarray(v, dtype=np.float32) for k, v in history.items()}


def _tail_mean(arr, n=20):
    if len(arr) == 0:
        return 0.0
    tail = arr[-min(n, len(arr)) :]
    return float(np.mean(tail))


def _build_summary(args, history):
    metrics = [
        "mu_reward",
        "uav_reward",
        "total_cost",
        "weighted_energy_mu_avg",
        "weighted_energy_mu_total",
        "mu_energy",
        "uav_energy",
        "jain_fairness",
        "delay_violation",
    ]
    tail = {m: _tail_mean(history[m]) for m in metrics}
    return {
        "algorithm": args.algorithm,
        "seed": int(args.seed),
        "num_mus": int(args.num_mus),
        "num_uavs": int(args.num_uavs),
        "total_steps": int(args.total_steps),
        "episode_length": int(args.episode_length),
        "tail_metrics": tail,
        "num_episodes": int(len(history["episode"])),
        "tag": args.tag,
    }


def train(args):
    args.algorithm = _canonical_algo_name(args.algorithm)
    _set_random_seed(args.seed)
    device = setup_device(args.device)

    overrides = _build_cfg_overrides(args)

    # output layout
    if args.run_dir:
        run_dir = args.run_dir
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        history_path = os.path.join(run_dir, "history.npz")
        summary_path = os.path.join(run_dir, "summary.json")
        best_ckpt_path = os.path.join(ckpt_dir, "best.pt")
        final_ckpt_path = os.path.join(ckpt_dir, "final.pt")
    else:
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        history_path = os.path.join(args.log_dir, f"{args.algorithm}_history.npz")
        summary_path = os.path.join(args.log_dir, f"{args.algorithm}_summary.json")
        best_ckpt_path = os.path.join(args.save_dir, f"{args.algorithm}_best.pt")
        final_ckpt_path = os.path.join(args.save_dir, f"{args.algorithm}_final.pt")

    print("=" * 70)
    print("AB-MAPPO Paper Reproduction Training")
    print(f"algorithm={args.algorithm} device={device} seed={args.seed}")
    print(f"K={args.num_mus} M={args.num_uavs} total_steps={args.total_steps} episode_length={args.episode_length}")
    if args.run_dir:
        print(f"run_dir={args.run_dir}")
    print("=" * 70)

    with _temporary_cfg(overrides):
        env = UAVMECEnv(
            num_mus=args.num_mus,
            num_uavs=args.num_uavs,
            dt_deviation_rate=args.dt_deviation,
            wo_dt_noise_mode=args.wo_dt_noise_mode,
            area_width=args.area_width,
            area_height=args.area_height,
            seed=args.seed,
        )
        agent = _make_agent(env, args.algorithm, device)

        writer = None
        if not args.disable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_name = f"{args.algorithm}_K{args.num_mus}_M{args.num_uavs}_seed{args.seed}"
                writer = SummaryWriter(os.path.join(args.log_dir, tb_name))
            except Exception:
                writer = None

        total_episodes = max(1, args.total_steps // args.episode_length)
        history = _history_schema()

        best_metric = float("inf")
        start_time = time.time()

        for episode in range(1, total_episodes + 1):
            episode_stats = agent.collect_episode()
            update_stats = agent.update()
            step = episode * args.episode_length

            history["episode"].append(episode)
            history["step"].append(step)
            history["mu_reward"].append(float(episode_stats.get("mu_reward", 0.0)))
            history["uav_reward"].append(float(episode_stats.get("uav_reward", 0.0)))
            history["total_cost"].append(float(episode_stats.get("total_cost", 0.0)))
            history["weighted_energy"].append(float(episode_stats.get("weighted_energy", 0.0)))
            history["weighted_energy_mu_avg"].append(float(episode_stats.get("weighted_energy_mu_avg", 0.0)))
            history["weighted_energy_mu_total"].append(float(episode_stats.get("weighted_energy_mu_total", 0.0)))
            history["mu_energy"].append(float(episode_stats.get("mu_energy", 0.0)))
            history["uav_energy"].append(float(episode_stats.get("uav_energy", 0.0)))
            history["jain_fairness"].append(float(episode_stats.get("jain_fairness", 0.0)))
            history["delay_violation"].append(float(episode_stats.get("delay_violation", 0.0)))
            history["actor_loss"].append(float(update_stats.get("actor_loss", 0.0)))
            history["critic_loss"].append(float(update_stats.get("critic_loss", 0.0)))
            history["entropy"].append(float(update_stats.get("entropy", 0.0)))

            if writer is not None:
                writer.add_scalar("Reward/MU", history["mu_reward"][-1], step)
                writer.add_scalar("Reward/UAV", history["uav_reward"][-1], step)
                writer.add_scalar("Energy/WeightedAvg", history["weighted_energy_mu_avg"][-1], step)
                writer.add_scalar("Energy/MUAvg", history["mu_energy"][-1], step)
                writer.add_scalar("Energy/UAVAvg", history["uav_energy"][-1], step)
                writer.add_scalar("Fairness/Jain", history["jain_fairness"][-1], step)
                writer.add_scalar("Violation/Delay", history["delay_violation"][-1], step)
                writer.add_scalar("Loss/Actor", history["actor_loss"][-1], step)
                writer.add_scalar("Loss/Critic", history["critic_loss"][-1], step)
                writer.add_scalar("Exploration/EntropyOrNoise", history["entropy"][-1], step)

            should_log = (episode == 1) or (episode % max(1, total_episodes // 20) == 0)
            if should_log:
                elapsed = max(1e-6, time.time() - start_time)
                eps_per_sec = episode / elapsed
                print(
                    f"ep={episode:4d}/{total_episodes} step={step:7d} "
                    f"mu_r={history['mu_reward'][-1]:8.4f} uav_r={history['uav_reward'][-1]:8.4f} "
                    f"wE={history['weighted_energy_mu_avg'][-1]:8.4f} fairness={history['jain_fairness'][-1]:6.3f} "
                    f"viol={history['delay_violation'][-1]:6.3f} eps/s={eps_per_sec:5.2f}"
                )

            metric = history["weighted_energy_mu_avg"][-1]
            if metric < best_metric and hasattr(agent, "save"):
                best_metric = metric
                agent.save(best_ckpt_path)

            if step % cfg.SAVE_INTERVAL == 0 and hasattr(agent, "save"):
                if args.run_dir:
                    interval_path = os.path.join(ckpt_dir, f"step{step}.pt")
                else:
                    interval_path = os.path.join(args.save_dir, f"{args.algorithm}_step{step}.pt")
                agent.save(interval_path)

        if hasattr(agent, "save"):
            agent.save(final_ckpt_path)
        if writer is not None:
            writer.close()

    history_np = _to_numpy_history(history)
    np.savez(history_path, **history_np)
    summary = _build_summary(args, history_np)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"history saved: {history_path}")
    print(f"summary saved: {summary_path}")
    return history_np


def namespace_from_kwargs(**kwargs):
    base = {
        "algorithm": "AB-MAPPO",
        "num_mus": cfg.NUM_MUS,
        "num_uavs": cfg.NUM_UAVS,
        "total_steps": cfg.TOTAL_STEPS,
        "episode_length": cfg.EPISODE_LENGTH,
        "device": "auto",
        "seed": 42,
        "dt_deviation": cfg.DT_DEVIATION_RATE,
        "wo_dt_noise_mode": False,
        "bandwidth_mhz": None,
        "weight_factor": None,
        "mu_max_cpu_ghz": None,
        "uav_max_cpu_ghz": None,
        "task_data_min_mbits": None,
        "task_data_max_mbits": None,
        "area_width": None,
        "area_height": None,
        "save_dir": cfg.SAVE_DIR,
        "log_dir": cfg.LOG_DIR,
        "run_dir": None,
        "disable_tensorboard": False,
        "tag": "",
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


if __name__ == "__main__":
    args = parse_args()
    train(args)
```

## train_sweep.py

```python
"""
Paper-aligned sweep executor for Fig.3-13 reproduction.

Outputs:
- Per-run artifacts in: experiments/{fig}/{algo}/{setting}/seed_{seed}/
  - history.npz
  - summary.json
  - checkpoints/*
- Aggregates in: experiments/{fig}/aggregate/*.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

import config as cfg
from train import namespace_from_kwargs, train


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _seed_list(seeds_str):
    if isinstance(seeds_str, (list, tuple)):
        return [int(s) for s in seeds_str]
    if not seeds_str:
        return list(cfg.DEFAULT_SEEDS)
    return [int(x.strip()) for x in str(seeds_str).split(",") if x.strip()]


def _fmt_value(v):
    if isinstance(v, float):
        s = f"{v:.4f}".rstrip("0").rstrip(".")
        return s.replace(".", "p")
    return str(v)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


@dataclass
class RunOptions:
    seeds: List[int]
    device: str
    total_steps: int
    episode_length: int
    resume: bool
    skip_existing: bool
    disable_tensorboard: bool
    smoke: bool


def _build_run_specs_for_setting(fig, algorithm, setting_name, base_kwargs, options: RunOptions):
    specs = []
    fig_dir = "base" if fig == "base" else f"fig{fig}"
    num_mus = int(base_kwargs.get("num_mus", cfg.NUM_MUS))
    num_uavs = int(base_kwargs.get("num_uavs", cfg.NUM_UAVS))
    for seed in options.seeds:
        run_dir = os.path.join(cfg.EXPERIMENT_ROOT, fig_dir, algorithm, setting_name, f"seed_{seed}")
        summary_path = os.path.join(run_dir, "summary.json")
        specs.append(
            {
                "fig": fig,
                "fig_dir": fig_dir,
                "algorithm": algorithm,
                "setting_name": setting_name,
                "seed": int(seed),
                "run_dir": run_dir,
                "summary_path": summary_path,
                "cli_overrides": dict(base_kwargs),
                "num_mus": num_mus,
                "num_uavs": num_uavs,
            }
        )
    return specs


def build_run_specs(fig, options: RunOptions):
    runners = {
        "base": _build_specs_base,
        "6": _build_specs_fig6,
        "7": _build_specs_fig7,
        "8": _build_specs_fig8,
        "9": _build_specs_fig9,
        "10": _build_specs_fig10,
        "11": _build_specs_fig11,
        "12": _build_specs_fig12,
    }
    targets = list(runners.keys()) if fig == "all" else [fig]
    specs = []
    for target in targets:
        if target not in runners:
            raise ValueError(f"Unsupported fig selector: {target}")
        specs.extend(runners[target](options))
    return specs


def _build_specs_base(options: RunOptions):
    specs = []
    for algo in cfg.CONVERGENCE_ALGOS:
        setting_name = f"K{cfg.NUM_MUS}_M{cfg.NUM_UAVS}"
        kwargs = {"num_mus": cfg.NUM_MUS, "num_uavs": cfg.NUM_UAVS}
        specs.extend(_build_run_specs_for_setting("base", algo, setting_name, kwargs, options))
    return specs


def _build_specs_fig6(options: RunOptions):
    spec = cfg.clone_figure_sweeps()["fig6"]
    xs = spec["x_values"] if not options.smoke else _values_for_smoke(spec["x_values"])
    specs = []
    for algo in spec["algorithms"]:
        for x in xs:
            setting_name = f"K{_fmt_value(x)}_M{spec['fixed']['num_uavs']}"
            kwargs = {"num_mus": int(x), "num_uavs": int(spec["fixed"]["num_uavs"])}
            specs.extend(_build_run_specs_for_setting("6", algo, setting_name, kwargs, options))
    return specs


def _build_specs_fig7(options: RunOptions):
    spec = cfg.clone_figure_sweeps()["fig7"]
    xs = spec["x_values"] if not options.smoke else _values_for_smoke(spec["x_values"])
    specs = []
    for algo in spec["algorithms"]:
        for x in xs:
            setting_name = f"K{spec['fixed']['num_mus']}_M{_fmt_value(x)}"
            kwargs = {"num_mus": int(spec["fixed"]["num_mus"]), "num_uavs": int(x)}
            specs.extend(_build_run_specs_for_setting("7", algo, setting_name, kwargs, options))
    return specs


def _build_specs_fig8(options: RunOptions):
    spec = cfg.clone_figure_sweeps()["fig8"]
    xs = spec["x_values"] if not options.smoke else _values_for_smoke(spec["x_values"])
    specs = []
    for algo in spec["algorithms"]:
        for x in xs:
            setting_name = f"K{spec['fixed']['num_mus']}_M{spec['fixed']['num_uavs']}_BW{_fmt_value(x)}"
            kwargs = {
                "num_mus": int(spec["fixed"]["num_mus"]),
                "num_uavs": int(spec["fixed"]["num_uavs"]),
                "bandwidth_mhz": float(x),
            }
            specs.extend(_build_run_specs_for_setting("8", algo, setting_name, kwargs, options))
    return specs


def _build_specs_fig9(options: RunOptions):
    spec = cfg.clone_figure_sweeps()["fig9"]
    xs = spec["x_values"] if not options.smoke else _values_for_smoke(spec["x_values"])
    specs = []
    for x in xs:
        setting_name = f"K{spec['fixed']['num_mus']}_M{spec['fixed']['num_uavs']}_w{_fmt_value(x)}"
        kwargs = {
            "num_mus": int(spec["fixed"]["num_mus"]),
            "num_uavs": int(spec["fixed"]["num_uavs"]),
            "weight_factor": float(x),
        }
        specs.extend(_build_run_specs_for_setting("9", "AB-MAPPO", setting_name, kwargs, options))
    return specs


def _build_specs_fig10(options: RunOptions):
    spec = cfg.clone_figure_sweeps()["fig10"]
    ks = spec["fixed"]["ks"] if not options.smoke else _values_for_smoke(spec["fixed"]["ks"])
    mu_cpus = spec["mu_cpu_ghz"] if not options.smoke else _values_for_smoke(spec["mu_cpu_ghz"])
    uav_cpus = spec["uav_cpu_ghz"] if not options.smoke else _values_for_smoke(spec["uav_cpu_ghz"])

    specs = []
    for k in ks:
        for cpu in mu_cpus:
            setting_name = f"K{k}_M{spec['fixed']['num_uavs']}_muCPU{_fmt_value(cpu)}"
            kwargs = {"num_mus": int(k), "num_uavs": int(spec["fixed"]["num_uavs"]), "mu_max_cpu_ghz": float(cpu)}
            specs.extend(_build_run_specs_for_setting("10", "AB-MAPPO", setting_name, kwargs, options))

    for k in ks:
        for cpu in uav_cpus:
            setting_name = f"K{k}_M{spec['fixed']['num_uavs']}_uavCPU{_fmt_value(cpu)}"
            kwargs = {"num_mus": int(k), "num_uavs": int(spec["fixed"]["num_uavs"]), "uav_max_cpu_ghz": float(cpu)}
            specs.extend(_build_run_specs_for_setting("10", "AB-MAPPO", setting_name, kwargs, options))
    return specs


def _build_specs_fig11(options: RunOptions):
    spec = cfg.clone_figure_sweeps()["fig11"]
    ks = spec["fixed"]["ks"] if not options.smoke else _values_for_smoke(spec["fixed"]["ks"])
    dts = spec["dt_values"] if not options.smoke else _values_for_smoke(spec["dt_values"])
    specs = []
    for k in ks:
        for dt in dts:
            setting_name = f"K{k}_M{spec['fixed']['num_uavs']}_dt{_fmt_value(dt)}"
            kwargs = {"num_mus": int(k), "num_uavs": int(spec["fixed"]["num_uavs"]), "dt_deviation": float(dt)}
            specs.extend(_build_run_specs_for_setting("11", "AB-MAPPO", setting_name, kwargs, options))
    return specs


def _build_specs_fig12(options: RunOptions):
    spec = cfg.clone_figure_sweeps()["fig12"]
    dmaxs = spec["task_max_mbits"] if not options.smoke else _values_for_smoke(spec["task_max_mbits"])
    bws = spec["bandwidth_mhz"] if not options.smoke else _values_for_smoke(spec["bandwidth_mhz"])
    specs = []
    for bw in bws:
        for use_wo_dt in (False, True):
            for dmax in dmaxs:
                setting_name = (
                    f"K{spec['fixed']['num_mus']}_M{spec['fixed']['num_uavs']}_BW{_fmt_value(bw)}_"
                    f"Dmax{_fmt_value(dmax)}_{'woDT' if use_wo_dt else 'withDT'}"
                )
                kwargs = {
                    "num_mus": int(spec["fixed"]["num_mus"]),
                    "num_uavs": int(spec["fixed"]["num_uavs"]),
                    "bandwidth_mhz": float(bw),
                    "task_data_min_mbits": 0.5,
                    "task_data_max_mbits": float(dmax),
                    "wo_dt_noise_mode": bool(use_wo_dt),
                    "dt_deviation": 0.0,
                }
                specs.extend(_build_run_specs_for_setting("12", "AB-MAPPO", setting_name, kwargs, options))
    return specs


def _run_one(fig, algorithm, setting_name, base_kwargs, options: RunOptions):
    seed_results = []
    for seed in options.seeds:
        run_dir = os.path.join(cfg.EXPERIMENT_ROOT, fig, algorithm, setting_name, f"seed_{seed}")
        summary_path = os.path.join(run_dir, "summary.json")
        should_skip = (options.skip_existing or options.resume) and os.path.exists(summary_path)

        if should_skip:
            summary = _load_json(summary_path)
            same_steps = int(summary.get("total_steps", -1)) == int(options.total_steps)
            same_epl = int(summary.get("episode_length", -1)) == int(options.episode_length)
            if not (same_steps and same_epl):
                should_skip = False
            else:
                print(f"[skip] {fig} {algorithm} {setting_name} seed={seed}")
        if not should_skip:
            _ensure_dir(run_dir)
            args = namespace_from_kwargs(
                algorithm=algorithm,
                seed=seed,
                device=options.device,
                total_steps=options.total_steps,
                episode_length=options.episode_length,
                run_dir=run_dir,
                disable_tensorboard=options.disable_tensorboard,
                **base_kwargs,
            )
            train(args)
            summary = _load_json(summary_path)

        seed_results.append(summary)
    return seed_results


def _aggregate_scalar(seed_summaries, metric):
    vals = [float(s["tail_metrics"][metric]) for s in seed_summaries]
    return float(np.mean(vals)), float(np.std(vals)), vals


def _aggregate_history(algo_dirs, out_path):
    aggregate = {"algorithms": {}}
    for algo, run_dirs in algo_dirs.items():
        histories = []
        for p in run_dirs:
            h_path = os.path.join(p, "history.npz")
            if os.path.exists(h_path):
                with np.load(h_path) as h:
                    histories.append({k: h[k] for k in h.files})
        if not histories:
            continue

        min_len = min(len(h["episode"]) for h in histories)
        steps = histories[0]["step"][:min_len].astype(float)

        def stack(key):
            return np.stack([h[key][:min_len].astype(float) for h in histories], axis=0)

        mu_reward = stack("mu_reward")
        uav_reward = stack("uav_reward")
        weighted = stack("weighted_energy_mu_avg")
        fairness = stack("jain_fairness")

        aggregate["steps"] = steps.tolist()
        aggregate["algorithms"][algo] = {
            "mu_reward_mean": mu_reward.mean(axis=0).tolist(),
            "mu_reward_std": mu_reward.std(axis=0).tolist(),
            "uav_reward_mean": uav_reward.mean(axis=0).tolist(),
            "uav_reward_std": uav_reward.std(axis=0).tolist(),
            "weighted_energy_mean": weighted.mean(axis=0).tolist(),
            "weighted_energy_std": weighted.std(axis=0).tolist(),
            "jain_fairness_mean": fairness.mean(axis=0).tolist(),
            "jain_fairness_std": fairness.std(axis=0).tolist(),
        }
    _save_json(out_path, aggregate)


def run_base_training(options: RunOptions):
    fig = "base"
    _ensure_dir(os.path.join(cfg.EXPERIMENT_ROOT, fig, "aggregate"))

    algos = cfg.CONVERGENCE_ALGOS
    algo_dirs = {}
    for algo in algos:
        setting_name = f"K{cfg.NUM_MUS}_M{cfg.NUM_UAVS}"
        kwargs = {"num_mus": cfg.NUM_MUS, "num_uavs": cfg.NUM_UAVS}
        seeds = _run_one(fig, algo, setting_name, kwargs, options)
        _ = seeds
        run_dirs = [
            os.path.join(cfg.EXPERIMENT_ROOT, fig, algo, setting_name, f"seed_{seed}") for seed in options.seeds
        ]
        algo_dirs[algo] = run_dirs

    out_path = os.path.join(cfg.EXPERIMENT_ROOT, fig, "aggregate", "convergence.json")
    _aggregate_history(algo_dirs, out_path)
    print(f"[saved] {out_path}")


def _values_for_smoke(values):
    if len(values) <= 2:
        return values
    return values[:2]


def run_fig6(options: RunOptions):
    spec = cfg.clone_figure_sweeps()["fig6"]
    xs = spec["x_values"] if not options.smoke else _values_for_smoke(spec["x_values"])
    metric = spec["metric"]

    out = {"figure": "fig6", "x_name": spec["x_name"], "x_values": xs, "metric": metric, "algorithms": {}}
    for algo in spec["algorithms"]:
        means, stds, raws = [], [], []
        for x in xs:
            setting_name = f"K{_fmt_value(x)}_M{spec['fixed']['num_uavs']}"
            kwargs = {"num_mus": int(x), "num_uavs": int(spec["fixed"]["num_uavs"])}
            seed_summaries = _run_one("fig6", algo, setting_name, kwargs, options)
            mean, std, raw = _aggregate_scalar(seed_summaries, metric)
            means.append(mean)
            stds.append(std)
            raws.append(raw)
        out["algorithms"][algo] = {"mean": means, "std": stds, "raw": raws}

    path = os.path.join(cfg.EXPERIMENT_ROOT, "fig6", "aggregate", "results.json")
    _ensure_dir(os.path.dirname(path))
    _save_json(path, out)
    print(f"[saved] {path}")


def run_fig7(options: RunOptions):
    spec = cfg.clone_figure_sweeps()["fig7"]
    xs = spec["x_values"] if not options.smoke else _values_for_smoke(spec["x_values"])
    metric = spec["metric"]
    out = {"figure": "fig7", "x_name": spec["x_name"], "x_values": xs, "metric": metric, "algorithms": {}}

    for algo in spec["algorithms"]:
        means, stds, raws = [], [], []
        for x in xs:
            setting_name = f"K{spec['fixed']['num_mus']}_M{_fmt_value(x)}"
            kwargs = {"num_mus": int(spec["fixed"]["num_mus"]), "num_uavs": int(x)}
            seed_summaries = _run_one("fig7", algo, setting_name, kwargs, options)
            mean, std, raw = _aggregate_scalar(seed_summaries, metric)
            means.append(mean)
            stds.append(std)
            raws.append(raw)
        out["algorithms"][algo] = {"mean": means, "std": stds, "raw": raws}

    path = os.path.join(cfg.EXPERIMENT_ROOT, "fig7", "aggregate", "results.json")
    _ensure_dir(os.path.dirname(path))
    _save_json(path, out)
    print(f"[saved] {path}")


def run_fig8(options: RunOptions):
    spec = cfg.clone_figure_sweeps()["fig8"]
    xs = spec["x_values"] if not options.smoke else _values_for_smoke(spec["x_values"])
    metric = spec["metric"]
    out = {"figure": "fig8", "x_name": spec["x_name"], "x_values": xs, "metric": metric, "algorithms": {}}

    for algo in spec["algorithms"]:
        means, stds, raws = [], [], []
        for x in xs:
            setting_name = f"K{spec['fixed']['num_mus']}_M{spec['fixed']['num_uavs']}_BW{_fmt_value(x)}"
            kwargs = {
                "num_mus": int(spec["fixed"]["num_mus"]),
                "num_uavs": int(spec["fixed"]["num_uavs"]),
                "bandwidth_mhz": float(x),
            }
            seed_summaries = _run_one("fig8", algo, setting_name, kwargs, options)
            mean, std, raw = _aggregate_scalar(seed_summaries, metric)
            means.append(mean)
            stds.append(std)
            raws.append(raw)
        out["algorithms"][algo] = {"mean": means, "std": stds, "raw": raws}

    path = os.path.join(cfg.EXPERIMENT_ROOT, "fig8", "aggregate", "results.json")
    _ensure_dir(os.path.dirname(path))
    _save_json(path, out)
    print(f"[saved] {path}")


def run_fig9(options: RunOptions):
    spec = cfg.clone_figure_sweeps()["fig9"]
    xs = spec["x_values"] if not options.smoke else _values_for_smoke(spec["x_values"])

    mu_mean, mu_std, mu_raw = [], [], []
    uav_mean, uav_std, uav_raw = [], [], []

    for x in xs:
        setting_name = f"K{spec['fixed']['num_mus']}_M{spec['fixed']['num_uavs']}_w{_fmt_value(x)}"
        kwargs = {
            "num_mus": int(spec["fixed"]["num_mus"]),
            "num_uavs": int(spec["fixed"]["num_uavs"]),
            "weight_factor": float(x),
        }
        seeds = _run_one("fig9", "AB-MAPPO", setting_name, kwargs, options)

        m, s, raw = _aggregate_scalar(seeds, "mu_energy")
        mu_mean.append(m)
        mu_std.append(s)
        mu_raw.append(raw)

        m, s, raw = _aggregate_scalar(seeds, "uav_energy")
        uav_mean.append(m)
        uav_std.append(s)
        uav_raw.append(raw)

    out = {
        "figure": "fig9",
        "x_name": spec["x_name"],
        "x_values": xs,
        "mu_energy": {"mean": mu_mean, "std": mu_std, "raw": mu_raw},
        "uav_energy": {"mean": uav_mean, "std": uav_std, "raw": uav_raw},
    }
    path = os.path.join(cfg.EXPERIMENT_ROOT, "fig9", "aggregate", "results.json")
    _ensure_dir(os.path.dirname(path))
    _save_json(path, out)
    print(f"[saved] {path}")


def run_fig10(options: RunOptions):
    spec = cfg.clone_figure_sweeps()["fig10"]
    ks = spec["fixed"]["ks"] if not options.smoke else _values_for_smoke(spec["fixed"]["ks"])
    mu_cpus = spec["mu_cpu_ghz"] if not options.smoke else _values_for_smoke(spec["mu_cpu_ghz"])
    uav_cpus = spec["uav_cpu_ghz"] if not options.smoke else _values_for_smoke(spec["uav_cpu_ghz"])

    out = {"figure": "fig10", "mu_cpu": {"x_values": mu_cpus, "series": {}}, "uav_cpu": {"x_values": uav_cpus, "series": {}}}

    for k in ks:
        key = f"K={k}"
        means, stds = [], []
        for cpu in mu_cpus:
            setting_name = f"K{k}_M{spec['fixed']['num_uavs']}_muCPU{_fmt_value(cpu)}"
            kwargs = {"num_mus": int(k), "num_uavs": int(spec["fixed"]["num_uavs"]), "mu_max_cpu_ghz": float(cpu)}
            seeds = _run_one("fig10", "AB-MAPPO", setting_name, kwargs, options)
            m, s, _ = _aggregate_scalar(seeds, "weighted_energy_mu_avg")
            means.append(m)
            stds.append(s)
        out["mu_cpu"]["series"][key] = {"mean": means, "std": stds}

    for k in ks:
        key = f"K={k}"
        means, stds = [], []
        for cpu in uav_cpus:
            setting_name = f"K{k}_M{spec['fixed']['num_uavs']}_uavCPU{_fmt_value(cpu)}"
            kwargs = {"num_mus": int(k), "num_uavs": int(spec["fixed"]["num_uavs"]), "uav_max_cpu_ghz": float(cpu)}
            seeds = _run_one("fig10", "AB-MAPPO", setting_name, kwargs, options)
            m, s, _ = _aggregate_scalar(seeds, "weighted_energy_mu_avg")
            means.append(m)
            stds.append(s)
        out["uav_cpu"]["series"][key] = {"mean": means, "std": stds}

    path = os.path.join(cfg.EXPERIMENT_ROOT, "fig10", "aggregate", "results.json")
    _ensure_dir(os.path.dirname(path))
    _save_json(path, out)
    print(f"[saved] {path}")


def run_fig11(options: RunOptions):
    spec = cfg.clone_figure_sweeps()["fig11"]
    ks = spec["fixed"]["ks"] if not options.smoke else _values_for_smoke(spec["fixed"]["ks"])
    dts = spec["dt_values"] if not options.smoke else _values_for_smoke(spec["dt_values"])

    out = {"figure": "fig11", "x_name": "dt_deviation", "x_values": dts, "series": {}}
    for k in ks:
        key = f"K={k}"
        means, stds = [], []
        for dt in dts:
            setting_name = f"K{k}_M{spec['fixed']['num_uavs']}_dt{_fmt_value(dt)}"
            kwargs = {"num_mus": int(k), "num_uavs": int(spec["fixed"]["num_uavs"]), "dt_deviation": float(dt)}
            seeds = _run_one("fig11", "AB-MAPPO", setting_name, kwargs, options)
            m, s, _ = _aggregate_scalar(seeds, "weighted_energy_mu_avg")
            means.append(m)
            stds.append(s)
        out["series"][key] = {"mean": means, "std": stds}

    path = os.path.join(cfg.EXPERIMENT_ROOT, "fig11", "aggregate", "results.json")
    _ensure_dir(os.path.dirname(path))
    _save_json(path, out)
    print(f"[saved] {path}")


def run_fig12(options: RunOptions):
    spec = cfg.clone_figure_sweeps()["fig12"]
    dmaxs = spec["task_max_mbits"] if not options.smoke else _values_for_smoke(spec["task_max_mbits"])
    bws = spec["bandwidth_mhz"] if not options.smoke else _values_for_smoke(spec["bandwidth_mhz"])

    out = {"figure": "fig12", "x_name": "task_max_mbits", "x_values": dmaxs, "series": {}}

    for bw in bws:
        for use_wo_dt in (False, True):
            label = f"B={bw}MHz {'w/o DT' if use_wo_dt else 'with DT'}"
            means, stds = [], []
            for dmax in dmaxs:
                setting_name = f"K{spec['fixed']['num_mus']}_M{spec['fixed']['num_uavs']}_BW{_fmt_value(bw)}_Dmax{_fmt_value(dmax)}_{'woDT' if use_wo_dt else 'withDT'}"
                kwargs = {
                    "num_mus": int(spec["fixed"]["num_mus"]),
                    "num_uavs": int(spec["fixed"]["num_uavs"]),
                    "bandwidth_mhz": float(bw),
                    "task_data_min_mbits": 0.5,
                    "task_data_max_mbits": float(dmax),
                    "wo_dt_noise_mode": bool(use_wo_dt),
                    "dt_deviation": 0.0,
                }
                seeds = _run_one("fig12", "AB-MAPPO", setting_name, kwargs, options)
                m, s, _ = _aggregate_scalar(seeds, "weighted_energy_mu_avg")
                means.append(m)
                stds.append(s)
            out["series"][label] = {"mean": means, "std": stds}

    path = os.path.join(cfg.EXPERIMENT_ROOT, "fig12", "aggregate", "results.json")
    _ensure_dir(os.path.dirname(path))
    _save_json(path, out)
    print(f"[saved] {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Paper sweep runner")
    parser.add_argument("--fig", type=str, default="all", help="all | base | 6 | 7 | 8 | 9 | 10 | 11 | 12")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--total_steps", type=int, default=cfg.TOTAL_STEPS)
    parser.add_argument("--episode_length", type=int, default=cfg.EPISODE_LENGTH)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--disable_tensorboard", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="small subset for fast pipeline validation")
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = _seed_list(args.seeds)
    if args.smoke and len(seeds) > 1:
        seeds = [seeds[0]]

    options = RunOptions(
        seeds=seeds,
        device=args.device,
        total_steps=args.total_steps,
        episode_length=args.episode_length,
        resume=args.resume,
        skip_existing=args.skip_existing,
        disable_tensorboard=args.disable_tensorboard,
        smoke=args.smoke,
    )

    runners = {
        "base": run_base_training,
        "6": run_fig6,
        "7": run_fig7,
        "8": run_fig8,
        "9": run_fig9,
        "10": run_fig10,
        "11": run_fig11,
        "12": run_fig12,
    }

    if args.fig == "all":
        run_base_training(options)
        run_fig6(options)
        run_fig7(options)
        run_fig8(options)
        run_fig9(options)
        run_fig10(options)
        run_fig11(options)
        run_fig12(options)
        return

    if args.fig not in runners:
        raise ValueError(f"Unsupported fig selector: {args.fig}")
    runners[args.fig](options)


if __name__ == "__main__":
    main()
```

