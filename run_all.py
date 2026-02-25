"""
Staged pipeline runner:
- smoke: fast correctness pass
- full: paper-scale full sweep
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
import subprocess
import sys
import time
import uuid


def _run(cmd):
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _safe_job_name(name):
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(name))


def _build_full_log_dir(base_dir="run_logs"):
    os.makedirs(base_dir, exist_ok=True)
    run_id = f"full_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    log_dir = os.path.join(base_dir, run_id)
    os.makedirs(log_dir, exist_ok=False)
    return log_dir


def _spawn_logged_process(job_name, cmd, log_dir):
    safe_name = _safe_job_name(job_name)
    log_file = os.path.join(log_dir, f"{safe_name}.log")
    print("[cmd]", " ".join(cmd))
    print("[log]", log_file)
    stream = open(log_file, "w", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, env=env)
    return {
        "name": str(job_name),
        "cmd": list(cmd),
        "log_file": log_file,
        "stream": stream,
        "process": process,
        "start_time": time.monotonic(),
    }


def _close_job_stream(job):
    stream = job.get("stream")
    if stream is not None and not stream.closed:
        stream.close()


def _terminate_job(job, grace_sec=3.0):
    process = job["process"]
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=grace_sec)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _shutdown_active_jobs(active_jobs):
    for job in list(active_jobs):
        try:
            _terminate_job(job)
        finally:
            _close_job_stream(job)


def _run_parallel_commands(job_specs, max_parallel, log_dir, job_timeout_sec=0.0, poll_interval_sec=0.2):
    pending = list(job_specs)
    active = []

    while pending or active:
        while pending and len(active) < max_parallel:
            name, cmd = pending.pop(0)
            active.append(_spawn_logged_process(name, cmd, log_dir))

        if not active:
            continue

        time.sleep(max(0.01, poll_interval_sec))

        for job in list(active):
            process = job["process"]
            elapsed = time.monotonic() - job["start_time"]

            if job_timeout_sec > 0 and process.poll() is None and elapsed > job_timeout_sec:
                _terminate_job(job)
                _close_job_stream(job)
                active.remove(job)
                _shutdown_active_jobs(active)
                raise RuntimeError(
                    f"Job timeout after {job_timeout_sec}s: {job['name']}; "
                    f"log={job['log_file']}"
                )

            return_code = process.poll()
            if return_code is None:
                continue

            _close_job_stream(job)
            active.remove(job)

            if return_code != 0:
                _shutdown_active_jobs(active)
                raise RuntimeError(
                    f"Job failed with code {return_code}: {job['name']} "
                    f"({' '.join(job['cmd'])}); log={job['log_file']}"
                )


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


def _build_full_cmd(fig, full_device):
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
        "--device",
        full_device,
    ]


def run_full(max_parallel=3, full_device="cpu", job_timeout_sec=0.0):
    fig_jobs = ["base", "6", "7", "8", "9", "10", "11", "12"]
    log_dir = _build_full_log_dir("run_logs")
    print("[run_logs]", log_dir)
    job_specs = [(f"fig{fig}", _build_full_cmd(fig, full_device)) for fig in fig_jobs]
    _run_parallel_commands(
        job_specs,
        max_parallel=max_parallel,
        log_dir=log_dir,
        job_timeout_sec=job_timeout_sec,
    )

    _run([sys.executable, "generate_figures.py", "--figs", "all"])


def parse_args():
    p = argparse.ArgumentParser(description="Run full paper reproduction pipeline")
    p.add_argument("--stage", choices=["smoke", "full", "all"], default="smoke")
    p.add_argument("--max_parallel", type=int, default=3, help="parallel figure jobs in full stage")
    p.add_argument("--full-device", type=str, default="cpu", help="device for full stage train_sweep runs")
    p.add_argument(
        "--job-timeout-sec",
        type=float,
        default=0.0,
        help="timeout per full-stage job in seconds (0 disables timeout)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.stage == "smoke":
        run_smoke()
    elif args.stage == "full":
        run_full(
            max_parallel=max(1, args.max_parallel),
            full_device=args.full_device,
            job_timeout_sec=max(0.0, float(args.job_timeout_sec)),
        )
    else:
        run_smoke()
        run_full(
            max_parallel=max(1, args.max_parallel),
            full_device=args.full_device,
            job_timeout_sec=max(0.0, float(args.job_timeout_sec)),
        )


if __name__ == "__main__":
    main()
