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
    parser.add_argument(
        "--paper_mode",
        type=str,
        default=cfg.PAPER_MODE,
        choices=["on", "off"],
        help="Enable paper-aligned preset overrides for observation/reward/rollout settings.",
    )
    parser.add_argument(
        "--normalize_reward",
        type=str,
        default="on" if cfg.NORMALIZE_REWARD else "off",
        choices=["on", "off"],
        help="Enable/disable running reward normalization.",
    )
    parser.add_argument(
        "--reward_scale",
        type=float,
        default=cfg.REWARD_SCALE,
        help="Override global REWARD_SCALE without changing source config.",
    )
    parser.add_argument(
        "--uav_obs_mask_mode",
        type=str,
        default=cfg.UAV_OBS_MASK_MODE,
        choices=["none", "prev_assoc"],
        help="UAV observation mask mode.",
    )
    parser.add_argument(
        "--rollout_mode",
        type=str,
        default=cfg.ROLLOUT_MODE,
        choices=["fixed", "env_episode"],
        help="Rollout collection mode for MAPPO-family agents.",
    )
    parser.add_argument(
        "--bs_relay_policy",
        type=str,
        default=cfg.BS_RELAY_POLICY,
        choices=["nearest", "best_snr", "min_load"],
        help="Relay-UAV selection policy when MU chooses BS relay action (M+1).",
    )

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
    overrides["REWARD_SCALE"] = float(args.reward_scale)
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


def _normalize_reward_flag(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"on", "true", "1", "yes"}


def _paper_mode_flag(value):
    return str(value).strip().lower() in {"on", "true", "1", "yes"}


def _apply_paper_mode_preset(args):
    """Mutate args in-place with paper-aligned defaults when paper_mode is enabled."""
    enabled = _paper_mode_flag(getattr(args, "paper_mode", "off"))
    args.paper_mode = "on" if enabled else "off"
    if not enabled:
        return args

    args.normalize_reward = "on" if bool(cfg.PAPER_PROFILE_NORMALIZE_REWARD) else "off"
    args.reward_scale = float(cfg.PAPER_PROFILE_REWARD_SCALE)
    args.uav_obs_mask_mode = str(cfg.PAPER_PROFILE_UAV_OBS_MASK_MODE)
    args.rollout_mode = str(cfg.PAPER_PROFILE_ROLLOUT_MODE)
    args.bs_relay_policy = str(cfg.PAPER_PROFILE_BS_RELAY_POLICY)
    return args


def _make_agent(env, algorithm, device, normalize_reward=True, rollout_mode=cfg.ROLLOUT_MODE):
    if algorithm == "MADDPG":
        return MADDPG(env, device=device)
    if algorithm == "Randomized":
        return ABMAPPO(
            env,
            algorithm="Randomized",
            device=device,
            normalize_reward=normalize_reward,
            rollout_mode=rollout_mode,
        )
    return ABMAPPO(
        env,
        algorithm=algorithm,
        device=device,
        normalize_reward=normalize_reward,
        rollout_mode=rollout_mode,
    )


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
        "paper_mode": bool(_paper_mode_flag(args.paper_mode)),
        "normalize_reward": bool(_normalize_reward_flag(args.normalize_reward)),
        "reward_scale": float(args.reward_scale),
        "uav_obs_mask_mode": str(args.uav_obs_mask_mode),
        "rollout_mode": str(args.rollout_mode),
        "bs_relay_policy": str(args.bs_relay_policy),
        "tail_metrics": tail,
        "num_episodes": int(len(history["episode"])),
        "tag": args.tag,
    }


def train(args):
    args.algorithm = _canonical_algo_name(args.algorithm)
    args = _apply_paper_mode_preset(args)
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
            uav_obs_mask_mode=args.uav_obs_mask_mode,
            bs_relay_policy=args.bs_relay_policy,
            area_width=args.area_width,
            area_height=args.area_height,
            seed=args.seed,
        )
        agent = _make_agent(
            env,
            args.algorithm,
            device,
            normalize_reward=_normalize_reward_flag(args.normalize_reward),
            rollout_mode=args.rollout_mode,
        )

        writer = None
        if not args.disable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_name = f"{args.algorithm}_K{args.num_mus}_M{args.num_uavs}_seed{args.seed}"
                writer = SummaryWriter(os.path.join(args.log_dir, tb_name))
            except Exception:
                writer = None

        target_steps = max(1, int(args.total_steps))
        nominal_episode_len = max(1, int(args.episode_length))
        estimated_total_episodes = max(1, int(np.ceil(target_steps / float(nominal_episode_len))))
        history = _history_schema()

        best_metric = float("inf")
        start_time = time.time()
        episode = 0
        step = 0
        last_log_step = -1

        while step < target_steps:
            episode += 1
            episode_stats = agent.collect_episode()
            update_stats = agent.update()
            collected_steps = int(episode_stats.get("collected_steps", nominal_episode_len))
            if collected_steps <= 0:
                collected_steps = nominal_episode_len
            step = min(target_steps, step + collected_steps)

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

            log_stride = max(1, target_steps // 20)
            should_log = (episode == 1) or (step >= target_steps) or (step - last_log_step >= log_stride)
            if should_log:
                last_log_step = step
                elapsed = max(1e-6, time.time() - start_time)
                eps_per_sec = episode / elapsed
                print(
                    f"ep={episode:4d}/{estimated_total_episodes} step={step:7d}/{target_steps} "
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
        "paper_mode": cfg.PAPER_MODE,
        "normalize_reward": "on" if cfg.NORMALIZE_REWARD else "off",
        "reward_scale": cfg.REWARD_SCALE,
        "uav_obs_mask_mode": cfg.UAV_OBS_MASK_MODE,
        "rollout_mode": cfg.ROLLOUT_MODE,
        "bs_relay_policy": cfg.BS_RELAY_POLICY,
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
