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
