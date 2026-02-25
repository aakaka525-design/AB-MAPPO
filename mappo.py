"""
AB-MAPPO 论文复现 — 核心算法 (精确版)
实现 Algorithm 1: AB-MAPPO

精确版改进:
  - UAV动作空间包含资源分配 (带宽+CPU)
  - 异构折扣因子 (γ_MU=0.8, γ_UAV=0.95)
  - 严格的CTDE: 注意力Critic用所有智能体观测
"""

import torch
import torch.nn as nn
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
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = x.size if hasattr(x, 'size') else 1
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

        # 优化器
        if algorithm not in ('Random', 'Randomized'):
            self.mu_actor_optimizer = optim.Adam(self.mu_actor.parameters(), lr=cfg.ACTOR_LR)
            self.uav_actor_optimizer = optim.Adam(self.uav_actor.parameters(), lr=cfg.ACTOR_LR)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.CRITIC_LR)

        # 缓冲区
        self.buffer = MultiAgentRolloutBuffer(
            num_mus=self.K, num_uavs=self.M,
            mu_obs_dim=env.mu_obs_dim, uav_obs_dim=env.uav_obs_dim,
            mu_action_dim=self.mu_action_dim,
            uav_action_dim=self.uav_action_dim,
            buffer_size=cfg.EPISODE_LENGTH,
        )

        # 奖励归一化
        self.mu_reward_rms = RunningMeanStd()
        self.uav_reward_rms = RunningMeanStd()


    @torch.no_grad()
    def get_actions(self, observations, deterministic=False):
        if self.algorithm in ('Random', 'Randomized'):
            return self._get_random_actions()

        mu_obs = torch.FloatTensor(observations['mu_obs']).to(self.device)
        uav_obs = torch.FloatTensor(observations['uav_obs']).to(self.device)

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

        return (mu_actions_env, uav_actions_np,
                mu_actions_np, uav_actions_np,
                mu_log_probs_np, uav_log_probs_np)

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
        if self.use_attention:
            all_obs = self._build_attention_input(observations)
            all_obs = torch.FloatTensor(all_obs).unsqueeze(0).to(self.device)
            values = self.critic(all_obs).squeeze(0).squeeze(-1).cpu().numpy()
            return values[:self.K], values[self.K:]
        else:
            state = torch.FloatTensor(self.env.get_state()).unsqueeze(0).to(self.device)
            v = self.critic(state).item()
            return np.full(self.K, v), np.full(self.M, v)

    def _build_attention_input(self, observations):
        N = self.K + self.M
        all_obs = np.zeros((N, self.critic_obs_dim), dtype=np.float32)
        all_obs[:self.K, :self.env.mu_obs_dim] = observations['mu_obs']
        all_obs[self.K:, :self.env.uav_obs_dim] = observations['uav_obs']
        return all_obs

    def collect_episode(self):
        self.buffer.reset()
        obs = self.env.reset()
        ep_info = []

        for t in range(cfg.EPISODE_LENGTH):
            (mu_act_env, uav_act_env, mu_act_store, uav_act_store,
             mu_lp, uav_lp) = self.get_actions(obs)
            mu_val, uav_val = self.get_values(obs)

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
            )
            ep_info.append(info)
            ep_info[-1]['raw_mu_reward'] = mu_r.copy()
            ep_info[-1]['raw_uav_reward'] = uav_r.copy()
            obs = next_obs
            if done:
                obs = self.env.reset()

        mu_last, uav_last = self.get_values(obs)
        self.buffer.compute_gae(mu_last, uav_last)

        raw_mu_r = np.mean([info['raw_mu_reward'].mean() for info in ep_info])
        raw_uav_r = np.mean([info['raw_uav_reward'].mean() for info in ep_info])

        return {
            'mu_reward': raw_mu_r,
            'uav_reward': raw_uav_r,
            'total_cost': -(raw_mu_r + raw_uav_r) / 2,
            'weighted_energy': np.mean([info['weighted_energy'] for info in ep_info]),
            'weighted_energy_mu_avg': np.mean([info['weighted_energy_mu_avg'] for info in ep_info]),
            'weighted_energy_mu_total': np.mean([info['weighted_energy_mu_total'] for info in ep_info]),
            'mu_energy': np.mean([info['mu_energy_avg'] for info in ep_info]),
            'uav_energy': np.mean([info['uav_energy_avg'] for info in ep_info]),
            'mu_energy_avg': np.mean([info['mu_energy_avg'] for info in ep_info]),
            'uav_energy_avg': np.mean([info['uav_energy_avg'] for info in ep_info]),
            'jain_fairness': np.mean([info['jain_fairness'] for info in ep_info]),
            'delay_violation': np.mean([info['delay_violation_rate'] for info in ep_info]),
        }


    def update(self):
        if self.algorithm in ('Random', 'Randomized'):
            return {}

        batches = self.buffer.get_batches()
        mu_data, uav_data = batches['mu'], batches['uav']

        total_al, total_cl, total_ent = 0, 0, 0

        for epoch in range(cfg.PPO_EPOCHS):
            # ---- MU Actor ----
            mu_obs = mu_data['observations'].to(self.device)
            mu_acts = mu_data['actions'].to(self.device)
            mu_old_lp = mu_data['log_probs'].to(self.device)
            mu_adv = mu_data['advantages'].to(self.device)
            mu_ret = mu_data['returns'].to(self.device)

            T, K = mu_obs.shape[0], mu_obs.shape[1]
            new_lp, ent = self.mu_actor.evaluate_action(
                mu_obs.reshape(T*K, -1), mu_acts.reshape(T*K, -1))
            new_lp = new_lp.reshape(T, K)
            ent = ent.reshape(T, K)

            adv = (mu_adv - mu_adv.mean()) / (mu_adv.std() + 1e-8)
            ratio = torch.exp(new_lp - mu_old_lp)
            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1-cfg.PPO_CLIP_EPSILON, 1+cfg.PPO_CLIP_EPSILON) * adv
            al_mu = -torch.min(s1, s2).mean()

            self.mu_actor_optimizer.zero_grad()
            (al_mu - cfg.ENTROPY_COEFF * ent.mean()).backward()
            nn.utils.clip_grad_norm_(self.mu_actor.parameters(), cfg.MAX_GRAD_NORM)
            self.mu_actor_optimizer.step()

            # ---- UAV Actor ----
            uav_obs = uav_data['observations'].to(self.device)
            uav_acts = uav_data['actions'].to(self.device)
            uav_old_lp = uav_data['log_probs'].to(self.device)
            uav_adv = uav_data['advantages'].to(self.device)

            T_u, M = uav_obs.shape[0], uav_obs.shape[1]
            new_lp_u, ent_u = self.uav_actor.evaluate_action(
                uav_obs.reshape(T_u*M, -1), uav_acts.reshape(T_u*M, -1))
            new_lp_u = new_lp_u.reshape(T_u, M)
            ent_u = ent_u.reshape(T_u, M)

            adv_u = (uav_adv - uav_adv.mean()) / (uav_adv.std() + 1e-8)
            ratio_u = torch.exp(new_lp_u - uav_old_lp)
            s1_u = ratio_u * adv_u
            s2_u = torch.clamp(ratio_u, 1-cfg.PPO_CLIP_EPSILON, 1+cfg.PPO_CLIP_EPSILON) * adv_u
            al_uav = -torch.min(s1_u, s2_u).mean()

            self.uav_actor_optimizer.zero_grad()
            (al_uav - cfg.ENTROPY_COEFF * ent_u.mean()).backward()
            nn.utils.clip_grad_norm_(self.uav_actor.parameters(), cfg.MAX_GRAD_NORM)
            self.uav_actor_optimizer.step()

            # ---- Critic ----
            if self.use_attention:
                cl = self._update_attention_critic(mu_data, uav_data)
            else:
                cl = self._update_mlp_critic(mu_data, uav_data)

            total_al += (al_mu.item() + al_uav.item()) / 2
            total_cl += cl
            total_ent += (ent.mean().item() + ent_u.mean().item()) / 2

        n = cfg.PPO_EPOCHS
        return {'actor_loss': total_al/n, 'critic_loss': total_cl/n, 'entropy': total_ent/n}

    def _update_attention_critic(self, mu_data, uav_data):
        mu_obs = mu_data['observations'].to(self.device)
        uav_obs = uav_data['observations'].to(self.device)
        mu_ret = mu_data['returns'].to(self.device)
        uav_ret = uav_data['returns'].to(self.device)
        T = mu_obs.shape[0]
        N = self.K + self.M

        all_obs = torch.zeros(T, N, self.critic_obs_dim, device=self.device)
        all_obs[:, :self.K, :self.env.mu_obs_dim] = mu_obs
        all_obs[:, self.K:, :self.env.uav_obs_dim] = uav_obs

        values = self.critic(all_obs).squeeze(-1)
        all_ret = torch.cat([mu_ret, uav_ret], dim=1)
        loss = 0.5 * ((values - all_ret) ** 2).mean()

        self.critic_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.MAX_GRAD_NORM)
        self.critic_optimizer.step()
        return loss.item()

    def _update_mlp_critic(self, mu_data, uav_data):
        mu_ret = mu_data['returns'].to(self.device).mean(dim=1, keepdim=True)
        uav_ret = uav_data['returns'].to(self.device).mean(dim=1, keepdim=True)
        avg_ret = (mu_ret + uav_ret) / 2

        mu_obs = mu_data['observations'].to(self.device)
        uav_obs = uav_data['observations'].to(self.device)
        state = torch.cat([mu_obs.mean(dim=1), uav_obs.mean(dim=1)], dim=1)

        if state.shape[1] < self.critic_obs_dim:
            pad = torch.zeros(state.shape[0], self.critic_obs_dim - state.shape[1], device=self.device)
            state = torch.cat([state, pad], dim=1)
        elif state.shape[1] > self.critic_obs_dim:
            state = state[:, :self.critic_obs_dim]

        values = self.critic(state)
        loss = 0.5 * ((values - avg_ret) ** 2).mean()

        self.critic_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.MAX_GRAD_NORM)
        self.critic_optimizer.step()
        return loss.item()

    def save(self, path):
        torch.save({
            'mu_actor': self.mu_actor.state_dict(),
            'uav_actor': self.uav_actor.state_dict(),
            'critic': self.critic.state_dict(),
            'algorithm': self.algorithm,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.mu_actor.load_state_dict(ckpt['mu_actor'])
        self.uav_actor.load_state_dict(ckpt['uav_actor'])
        self.critic.load_state_dict(ckpt['critic'])
