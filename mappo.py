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
