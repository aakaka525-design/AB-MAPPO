"""
Fast parallel benchmark: CPU vs CUDA, proper thread pinning.
Each test: 600 steps (2 episodes x 300), short sanity benchmark.
"""
import os
import subprocess
import sys
import time

STEPS, EPL, K, M, SEED = 600, 300, 60, 10, 42
CORES = os.cpu_count() or 24


def run_parallel(n, device="cpu", threads_per_proc=None):
    env = os.environ.copy()
    if threads_per_proc:
        env["OMP_NUM_THREADS"] = str(threads_per_proc)
        env["MKL_NUM_THREADS"] = str(threads_per_proc)
    procs = []
    t0 = time.perf_counter()
    for i in range(n):
        cmd = [
            sys.executable,
            "train.py",
            "--algorithm",
            "AB-MAPPO",
            "--num_mus",
            str(K),
            "--num_uavs",
            str(M),
            "--total_steps",
            str(STEPS),
            "--episode_length",
            str(EPL),
            "--seed",
            str(SEED + i),
            "--device",
            device,
            "--disable_tensorboard",
            "--run_dir",
            f"/tmp/bp_{device}_{n}_{i}",
        ]
        procs.append((i, subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)))
    for idx, p in procs:
        _, stderr = p.communicate()
        if p.returncode != 0:
            err_msg = stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"parallel job {idx} failed with code {p.returncode}: {err_msg}")
    elapsed = time.perf_counter() - t0
    eps_total = n * (STEPS / EPL) / elapsed
    return elapsed, eps_total


def main():
    print("=" * 60)
    print(f"Parallel Benchmark: K={K} M={M} steps={STEPS} cores={CORES}")
    print("=" * 60)

    # Key insight: threads_per_proc = CORES // n_parallel
    tests = [
        # (label, n_parallel, device, threads_per_proc)
        ("CPU x1 (24 thr)", 1, "cpu", 24),
        ("CPU x4 (6 thr)", 4, "cpu", 6),
        ("CPU x8 (3 thr)", 8, "cpu", 3),
        ("CPU x12 (2 thr)", 12, "cpu", 2),
        ("CPU x24 (1 thr)", 24, "cpu", 1),
        ("CUDA x1 (1 thr)", 1, "cuda", 1),
        ("CUDA x1 (24 thr)", 1, "cuda", 24),
    ]

    results = []
    for label, n, dev, thr in tests:
        print(f"[{len(results)+1}/{len(tests)}] {label}...", end=" ", flush=True)
        t, eps = run_parallel(n, dev, thr)
        results.append((label, n, t, eps))
        print(f"{t:.1f}s  {eps:.2f} eps/s total", flush=True)

    print()
    print("=" * 60)
    fmt = "{:<25} {:>4} {:>8} {:>12}"
    print(fmt.format("Config", "Jobs", "Time(s)", "eps/s total"))
    print("-" * 60)
    for label, n, t, eps in results:
        print(fmt.format(label, str(n), f"{t:.1f}", f"{eps:.2f}"))
    print("=" * 60)
    best = max(results, key=lambda x: x[3])
    print(f"\nHighest throughput: {best[0]} ({best[3]:.2f} eps/s, {best[2]:.1f}s)")


if __name__ == "__main__":
    main()
