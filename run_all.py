"""
Staged pipeline runner:
- smoke: fast correctness pass
- full: paper-scale full sweep
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


def _run(cmd):
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _run_logged(cmd, log_file):
    print("[cmd]", " ".join(cmd))
    print("[log]", log_file)
    with open(log_file, "w", encoding="utf-8") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed with code {p.returncode}: {' '.join(cmd)}")


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


def run_full(max_parallel=3):
    os.makedirs("run_logs", exist_ok=True)
    fig_jobs = ["base", "6", "7", "8", "9", "10", "11", "12"]

    def cmd_for(fig):
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
        ]

    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        futures = set()
        pending = list(fig_jobs)
        while pending or futures:
            while pending and len(futures) < max_parallel:
                fig = pending.pop(0)
                log = os.path.join("run_logs", f"full_fig{fig}.log")
                futures.add(ex.submit(_run_logged, cmd_for(fig), log))

            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                fut.result()

    _run([sys.executable, "generate_figures.py", "--figs", "all"])


def parse_args():
    p = argparse.ArgumentParser(description="Run full paper reproduction pipeline")
    p.add_argument("--stage", choices=["smoke", "full", "all"], default="smoke")
    p.add_argument("--max_parallel", type=int, default=3, help="parallel figure jobs in full stage")
    return p.parse_args()


def main():
    args = parse_args()
    if args.stage == "smoke":
        run_smoke()
    elif args.stage == "full":
        run_full(max_parallel=max(1, args.max_parallel))
    else:
        run_smoke()
        run_full(max_parallel=max(1, args.max_parallel))


if __name__ == "__main__":
    main()
