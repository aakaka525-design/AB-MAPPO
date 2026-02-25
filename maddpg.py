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
