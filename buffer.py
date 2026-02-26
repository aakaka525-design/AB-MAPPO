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
