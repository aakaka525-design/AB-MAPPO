"""
Single authoritative figure generator for paper Fig.3-Fig.13.

Input:
- Aggregated sweep results in experiments/{fig}/aggregate/*.json
Output:
- PNG figures under results/
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

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
    for i, (k, series) in enumerate(sorted(data["mu_cpu"]["series"].items())):
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
    for i, (k, series) in enumerate(sorted(data["uav_cpu"]["series"].items())):
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
    for i, (label, series) in enumerate(sorted(data["series"].items())):
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


def _make_corner_positions(width, height, m):
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
    arr += np.random.normal(0.0, 0.03 * min(width, height), size=arr.shape)
    arr[:, 0] = np.clip(arr[:, 0], 0.0, width)
    arr[:, 1] = np.clip(arr[:, 1], 0.0, height)
    return arr


def _run_heuristic_trajectory(scenario):
    k = scenario["K"]
    m = scenario["M"]
    w = scenario["W"]
    t_horizon = scenario["T"]
    env = UAVMECEnv(num_mus=k, num_uavs=m, area_width=w, area_height=w, dt_deviation_rate=0.0, wo_dt_noise_mode=False)
    env.reset()

    if scenario.get("cluster", False):
        centers = np.array([[0.25 * w, 0.25 * w], [0.75 * w, 0.7 * w], [0.55 * w, 0.2 * w]], dtype=np.float32)
        for i in range(k):
            env.mu_positions[i] = centers[i % len(centers)] + np.random.normal(0, 0.04 * w, size=2)
        env.mu_positions[:, 0] = np.clip(env.mu_positions[:, 0], 0.0, w)
        env.mu_positions[:, 1] = np.clip(env.mu_positions[:, 1], 0.0, w)

    if scenario.get("uav_start", "random") == "corner":
        env.uav_positions = _make_corner_positions(w, w, m)
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
                target = env.mu_positions[np.random.randint(0, k)]
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


def plot_fig13(exp_root, output_dir):
    _ = exp_root
    scenarios = [
        {"title": "(a) K=12, M=3, W=500m, T=50s", "K": 12, "M": 3, "W": 500, "T": 50, "cluster": True, "uav_start": "random"},
        {"title": "(b) K=12, M=3, W=500m, T=50s", "K": 12, "M": 3, "W": 500, "T": 50, "cluster": False, "uav_start": "corner"},
        {"title": "(c) K=60, M=10, W=1000m, T=60s", "K": 60, "M": 10, "W": 1000, "T": 60, "cluster": False, "uav_start": "random"},
        {"title": "(d) K=60, M=10, W=1000m, T=60s", "K": 60, "M": 10, "W": 1000, "T": 60, "cluster": False, "uav_start": "corner"},
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    cmap = plt.get_cmap("tab10")

    for ax, sc in zip(axes.flat, scenarios):
        mu_pos, trajs = _run_heuristic_trajectory(sc)
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
    return parser.parse_args()


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
        13: lambda: plot_fig13(args.exp_root, args.output_dir),
    }

    if args.figs == "all":
        targets = list(range(3, 14))
    else:
        targets = [int(x.strip()) for x in args.figs.split(",") if x.strip()]

    for n in targets:
        if n not in runners:
            raise ValueError(f"Unsupported figure number: {n}")
        print(f"[run] Fig.{n}")
        runners[n]()

    print(f"done. figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
