"""
Experiment output consistency checks for full sweep pipelines.

This module verifies:
1) expected run summaries exist and match requested params
2) aggregate JSON files are present and newer than source summaries
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List

import config as cfg
from train_sweep import RunOptions, build_run_specs


def _canonical_algo_name(name: str) -> str:
    return "Randomized" if str(name) == "Random" else str(name)


def _spec_id(spec: dict) -> str:
    return (
        f"fig={spec.get('fig')} algo={spec.get('algorithm')} "
        f"setting={spec.get('setting_name')} seed={spec.get('seed')}"
    )


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_run_summaries(specs: Iterable[dict], total_steps: int, episode_length: int) -> List[str]:
    errors: List[str] = []
    for spec in specs:
        summary_path = str(spec["summary_path"])
        sid = _spec_id(spec)
        if not os.path.exists(summary_path):
            errors.append(f"{sid}: missing summary: {summary_path}")
            continue

        try:
            summary = _load_json(summary_path)
        except Exception as exc:
            errors.append(f"{sid}: failed to load summary ({summary_path}): {exc}")
            continue

        checks = [
            ("algorithm", _canonical_algo_name(spec.get("algorithm")), _canonical_algo_name(summary.get("algorithm"))),
            ("seed", int(spec.get("seed")), int(summary.get("seed", -1))),
            ("num_mus", int(spec.get("num_mus", -1)), int(summary.get("num_mus", -1))),
            ("num_uavs", int(spec.get("num_uavs", -1)), int(summary.get("num_uavs", -1))),
            ("total_steps", int(total_steps), int(summary.get("total_steps", -1))),
            ("episode_length", int(episode_length), int(summary.get("episode_length", -1))),
        ]
        for key, expected, actual in checks:
            if expected != actual:
                errors.append(f"{sid}: {key} mismatch (expected={expected}, actual={actual})")
    return errors


def default_aggregate_paths(experiment_root: str | None = None) -> Dict[str, str]:
    root = experiment_root or cfg.EXPERIMENT_ROOT
    paths = {"base": os.path.join(root, "base", "aggregate", "convergence.json")}
    for fig in ("6", "7", "8", "9", "10", "11", "12"):
        paths[fig] = os.path.join(root, f"fig{fig}", "aggregate", "results.json")
    return paths


def validate_aggregate_freshness(specs: Iterable[dict], aggregate_paths: Dict[str, str]) -> List[str]:
    errors: List[str] = []
    latest_summary_mtime: Dict[str, float] = {}
    for spec in specs:
        fig = str(spec.get("fig"))
        summary_path = str(spec.get("summary_path"))
        if not os.path.exists(summary_path):
            continue
        try:
            mt = os.path.getmtime(summary_path)
        except OSError:
            continue
        latest_summary_mtime[fig] = max(latest_summary_mtime.get(fig, 0.0), mt)

    for fig, latest_mt in latest_summary_mtime.items():
        aggregate_path = aggregate_paths.get(fig)
        if not aggregate_path:
            errors.append(f"fig={fig}: missing aggregate path mapping")
            continue
        if not os.path.exists(aggregate_path):
            errors.append(f"fig={fig}: missing aggregate: {aggregate_path}")
            continue
        try:
            agg_mt = os.path.getmtime(aggregate_path)
        except OSError as exc:
            errors.append(f"fig={fig}: aggregate stat failed ({aggregate_path}): {exc}")
            continue
        if agg_mt + 1e-6 < latest_mt:
            errors.append(
                f"fig={fig}: stale aggregate ({aggregate_path}) older than latest summary mtime "
                f"(aggregate={agg_mt:.3f}, latest_summary={latest_mt:.3f})"
            )

        # Ensure aggregate is valid JSON as a final gate.
        try:
            _load_json(aggregate_path)
        except Exception as exc:
            errors.append(f"fig={fig}: aggregate JSON parse failed ({aggregate_path}): {exc}")
    return errors


def validate_paper_profile(specs: Iterable[dict]) -> List[str]:
    errors: List[str] = []
    expected = {
        "paper_mode": True,
        "normalize_reward": bool(cfg.PAPER_PROFILE_NORMALIZE_REWARD),
        "uav_obs_mask_mode": str(cfg.PAPER_PROFILE_UAV_OBS_MASK_MODE),
        "rollout_mode": str(cfg.PAPER_PROFILE_ROLLOUT_MODE),
        "bs_relay_policy": str(cfg.PAPER_PROFILE_BS_RELAY_POLICY),
    }
    expected_reward_scale = float(cfg.PAPER_PROFILE_REWARD_SCALE)

    for spec in specs:
        summary_path = str(spec["summary_path"])
        sid = _spec_id(spec)
        if not os.path.exists(summary_path):
            continue
        try:
            summary = _load_json(summary_path)
        except Exception:
            continue

        for key, expected_value in expected.items():
            actual = summary.get(key)
            if actual != expected_value:
                errors.append(f"{sid}: {key} mismatch (expected={expected_value}, actual={actual})")

        actual_scale = float(summary.get("reward_scale", float("nan")))
        if abs(actual_scale - expected_reward_scale) > 1e-8:
            errors.append(
                f"{sid}: reward_scale mismatch (expected={expected_reward_scale}, actual={actual_scale})"
            )
    return errors


def validate_experiment_outputs(
    total_steps: int,
    episode_length: int,
    seeds: list[int],
    fig: str = "all",
    experiment_root: str | None = None,
    paper_mode: bool = False,
) -> None:
    options = RunOptions(
        seeds=[int(s) for s in seeds],
        device="cpu",
        total_steps=int(total_steps),
        episode_length=int(episode_length),
        resume=True,
        skip_existing=False,
        disable_tensorboard=True,
        smoke=False,
        paper_mode=bool(paper_mode),
    )
    specs = build_run_specs(fig, options)
    errors = validate_run_summaries(specs, total_steps=int(total_steps), episode_length=int(episode_length))
    if paper_mode:
        errors.extend(validate_paper_profile(specs))
    errors.extend(validate_aggregate_freshness(specs, default_aggregate_paths(experiment_root)))
    if errors:
        detail = "\n".join(f"- {e}" for e in errors[:100])
        extra = "" if len(errors) <= 100 else f"\n- ... and {len(errors) - 100} more"
        raise RuntimeError(f"Experiment validation failed with {len(errors)} issue(s):\n{detail}{extra}")
