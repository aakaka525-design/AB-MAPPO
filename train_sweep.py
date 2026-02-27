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
    paper_mode: bool = False


def _build_run_specs_for_setting(fig, algorithm, setting_name, base_kwargs, options: RunOptions):
    specs = []
    fig_dir = "base" if fig == "base" else f"fig{fig}"
    num_mus = int(base_kwargs.get("num_mus", cfg.NUM_MUS))
    num_uavs = int(base_kwargs.get("num_uavs", cfg.NUM_UAVS))
    for seed in options.seeds:
        run_dir = os.path.join(cfg.EXPERIMENT_ROOT, fig_dir, algorithm, setting_name, f"seed_{seed}")
        summary_path = os.path.join(run_dir, "summary.json")
        cli_overrides = dict(base_kwargs)
        if bool(options.paper_mode):
            cli_overrides["paper_mode"] = "on"
        specs.append(
            {
                "fig": fig,
                "fig_dir": fig_dir,
                "algorithm": algorithm,
                "setting_name": setting_name,
                "seed": int(seed),
                "run_dir": run_dir,
                "summary_path": summary_path,
                "cli_overrides": cli_overrides,
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
    parser.add_argument(
        "--paper_mode",
        action="store_true",
        help="Enable paper-aligned train preset (normalize_reward=off, reward_scale=1.0, prev_assoc, env_episode, best_snr).",
    )
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
        paper_mode=bool(args.paper_mode),
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
