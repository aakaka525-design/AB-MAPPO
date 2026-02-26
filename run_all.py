"""
Staged pipeline runner:
- smoke: fast correctness pass
- full: paper-scale full sweep
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid

from train_sweep import RunOptions, build_run_specs


FULL_SEEDS = "42,43,44"
FULL_TOTAL_STEPS = 80000
FULL_EPISODE_LENGTH = 300
CUDA_PARALLEL_HARD_CAP = 12
CUDA_RESERVE_GB = 1.5
CUDA_MEM_PER_JOB_GB = 0.35
CUDA_OOM_KEYWORDS = (
    "cuda out of memory",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
)


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


def _spawn_logged_process(job_name, cmd, log_dir, threads_per_proc=None):
    safe_name = _safe_job_name(job_name)
    log_file = os.path.join(log_dir, f"{safe_name}.log")
    print("[cmd]", " ".join(cmd))
    print("[log]", log_file)
    stream = open(log_file, "w", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if threads_per_proc is not None:
        thread_env = str(int(max(1, threads_per_proc)))
        env["OMP_NUM_THREADS"] = thread_env
        env["MKL_NUM_THREADS"] = thread_env
        env["OPENBLAS_NUM_THREADS"] = thread_env
        env["NUMEXPR_NUM_THREADS"] = thread_env
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


def _run_parallel_commands(
    job_specs,
    max_parallel,
    log_dir,
    job_timeout_sec=0.0,
    poll_interval_sec=0.2,
    threads_per_proc=None,
):
    pending = list(job_specs)
    active = []

    while pending or active:
        while pending and len(active) < max_parallel:
            name, cmd = pending.pop(0)
            active.append(_spawn_logged_process(name, cmd, log_dir, threads_per_proc=threads_per_proc))

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


def _is_cuda_device(device):
    return "cuda" in str(device).lower()


def _resolve_cuda_device_for_query(full_device):
    device_str = str(full_device).lower()
    if ":" in device_str:
        suffix = device_str.split(":", 1)[1].strip()
        if suffix.isdigit():
            return int(suffix)
    return None


def _estimate_cuda_parallel(max_parallel, full_device):
    try:
        import torch
    except Exception:
        return 1, None, None

    try:
        if not torch.cuda.is_available():
            return 1, None, None

        device_idx = _resolve_cuda_device_for_query(full_device)
        if device_idx is None:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        else:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)

        free_gb = float(free_bytes) / float(1024**3)
        total_gb = float(total_bytes) / float(1024**3)
        available_gb = max(0.0, free_gb - float(CUDA_RESERVE_GB))
        mem_jobs = int(math.floor(available_gb / float(CUDA_MEM_PER_JOB_GB)))
        effective = max(
            1,
            min(
                int(max_parallel),
                int(CUDA_PARALLEL_HARD_CAP),
                mem_jobs,
            ),
        )
        return effective, free_gb, total_gb
    except Exception:
        return 1, None, None


def _effective_parallel(max_parallel, full_device):
    if _is_cuda_device(full_device):
        effective, _, _ = _estimate_cuda_parallel(max_parallel, full_device)
        return effective
    return max(1, int(max_parallel))


def _extract_log_path(error_text):
    match = re.search(r"log=([^\s]+)", str(error_text))
    if not match:
        return None
    return match.group(1)


def _read_log_tail(log_path, max_bytes=65536):
    if not log_path or not os.path.exists(log_path):
        return ""
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            seek_pos = max(0, size - max_bytes)
            f.seek(seek_pos, os.SEEK_SET)
            return f.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _is_oom_failure(error_text):
    combined = str(error_text).lower()
    # Linux OOM killer sends SIGKILL (-9)
    if "failed with code -9" in combined:
        return True
    log_path = _extract_log_path(error_text)
    if log_path:
        combined += "\n" + _read_log_tail(log_path).lower()
    return any(keyword in combined for keyword in CUDA_OOM_KEYWORDS)


def _summary_matches(summary_path, total_steps, episode_length):
    if not os.path.exists(summary_path):
        return False
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception:
        return False
    same_steps = int(summary.get("total_steps", -1)) == int(total_steps)
    same_epl = int(summary.get("episode_length", -1)) == int(episode_length)
    return same_steps and same_epl


def _append_override_arg(cmd, key, value):
    if value is None:
        return
    if key == "wo_dt_noise_mode":
        if bool(value):
            cmd.append("--wo_dt_noise_mode")
        return
    cmd.extend([f"--{key}", str(value)])


def _build_train_cmd_from_spec(run_spec, full_device, total_steps, episode_length):
    cmd = [
        sys.executable,
        "train.py",
        "--algorithm",
        str(run_spec["algorithm"]),
        "--seed",
        str(run_spec["seed"]),
        "--device",
        str(full_device),
        "--total_steps",
        str(total_steps),
        "--episode_length",
        str(episode_length),
        "--run_dir",
        str(run_spec["run_dir"]),
        "--disable_tensorboard",
    ]
    for key, value in run_spec.get("cli_overrides", {}).items():
        _append_override_arg(cmd, key, value)
    return cmd


def _build_run_job_specs(full_device, total_steps, episode_length):
    options = RunOptions(
        seeds=[42, 43, 44],
        device=full_device,
        total_steps=total_steps,
        episode_length=episode_length,
        resume=True,
        skip_existing=False,
        disable_tensorboard=True,
        smoke=False,
    )
    all_specs = build_run_specs("all", options)
    pending_specs = []

    for spec in all_specs:
        if _summary_matches(spec["summary_path"], total_steps, episode_length):
            print(
                f"[skip] fig={spec['fig']} algo={spec['algorithm']} "
                f"{spec['setting_name']} seed={spec['seed']}"
            )
            continue
        pending_specs.append(spec)

    pending_specs.sort(
        key=lambda s: (int(s.get("num_mus", 0)) * int(s.get("num_uavs", 0))),
        reverse=True,
    )

    job_specs = []
    for spec in pending_specs:
        name = f"fig{spec['fig']}_{spec['algorithm']}_{spec['setting_name']}_seed{spec['seed']}"
        cmd = _build_train_cmd_from_spec(spec, full_device, total_steps, episode_length)
        job_specs.append((name, cmd))
    return job_specs


def run_full(max_parallel=3, full_device="cpu", job_timeout_sec=0.0):
    log_dir = _build_full_log_dir("run_logs")
    cpu_count = os.cpu_count() or 1
    forced_parallel = None
    attempt = 0

    print("[run_logs]", log_dir)
    while True:
        attempt += 1

        if _is_cuda_device(full_device):
            suggested_parallel, gpu_free_gb, gpu_total_gb = _estimate_cuda_parallel(max_parallel, full_device)
            if forced_parallel is None:
                effective_parallel = suggested_parallel
            else:
                effective_parallel = max(1, min(suggested_parallel, forced_parallel))
        else:
            effective_parallel = _effective_parallel(max_parallel, full_device)
            gpu_free_gb = None
            gpu_total_gb = None

        threads_per_proc = max(1, cpu_count // max(1, effective_parallel))
        job_specs = _build_run_job_specs(
            full_device=full_device,
            total_steps=FULL_TOTAL_STEPS,
            episode_length=FULL_EPISODE_LENGTH,
        )

        gpu_free_text = f"{gpu_free_gb:.2f}" if gpu_free_gb is not None else "n/a"
        gpu_total_text = f"{gpu_total_gb:.2f}" if gpu_total_gb is not None else "n/a"
        print(
            f"[full] attempt={attempt} device={full_device} requested_parallel={max_parallel} "
            f"effective_parallel={effective_parallel} threads_per_proc={threads_per_proc} "
            f"pending_jobs={len(job_specs)} gpu_free_gb={gpu_free_text} gpu_total_gb={gpu_total_text}"
        )

        if not job_specs:
            print("[full] no pending training jobs, continue to aggregation")
            break

        try:
            _run_parallel_commands(
                job_specs,
                max_parallel=effective_parallel,
                log_dir=log_dir,
                job_timeout_sec=job_timeout_sec,
                threads_per_proc=threads_per_proc,
            )
            break
        except RuntimeError as exc:
            if effective_parallel > 1 and _is_oom_failure(str(exc)):
                next_parallel = max(1, effective_parallel // 2)
                if next_parallel == effective_parallel:
                    next_parallel = max(1, effective_parallel - 1)
                forced_parallel = next_parallel
                print(
                    f"[full][retry] OOM detected (code -9 or CUDA OOM), reducing parallel "
                    f"{effective_parallel} -> {forced_parallel}"
                )
                continue
            raise

    _run(
        [
            sys.executable,
            "train_sweep.py",
            "--fig",
            "all",
            "--seeds",
            FULL_SEEDS,
            "--total_steps",
            str(FULL_TOTAL_STEPS),
            "--episode_length",
            str(FULL_EPISODE_LENGTH),
            "--resume",
            "--skip_existing",
            "--disable_tensorboard",
            "--device",
            full_device,
        ]
    )

    _run([sys.executable, "generate_figures.py", "--figs", "all"])


def parse_args():
    p = argparse.ArgumentParser(description="Run full paper reproduction pipeline")
    p.add_argument("--stage", choices=["smoke", "full", "all"], default="smoke")
    p.add_argument("--max_parallel", type=int, default=3, help="parallel training jobs in full stage")
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
