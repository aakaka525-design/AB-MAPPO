"""
CPU vs CUDA 基准测试。
用 2400 步 (8 episodes × 300 步) 做快速对比。
"""
import os
import subprocess
import sys
import time

STEPS = 2400
EPL = 300
K, M = 60, 10
SEED = 42


def run_bench(label, device, blas_threads=None):
    env = os.environ.copy()
    if blas_threads:
        env["OMP_NUM_THREADS"] = str(blas_threads)
        env["MKL_NUM_THREADS"] = str(blas_threads)
    run_dir = f"/tmp/bench_{label}"
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
        str(SEED),
        "--device",
        device,
        "--disable_tensorboard",
        "--run_dir",
        run_dir,
    ]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    eps_s = "?"
    for line in r.stdout.strip().split("\n"):
        if "eps/s=" in line:
            eps_s = line.split("eps/s=")[-1].strip()
    return elapsed, eps_s, r.returncode


def run_parallel(n_parallel):
    """Run n_parallel CPU training processes simultaneously."""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    procs = []
    t0 = time.perf_counter()
    for i in range(n_parallel):
        run_dir = f"/tmp/bench_par{n_parallel}_{i}"
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
            "cpu",
            "--disable_tensorboard",
            "--run_dir",
            run_dir,
        ]
        procs.append((i, subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)))
    for idx, p in procs:
        _, stderr = p.communicate()
        if p.returncode != 0:
            err_msg = stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"parallel job {idx} failed with code {p.returncode}: {err_msg}")
    elapsed = time.perf_counter() - t0
    return elapsed


def main():
    print("=" * 60)
    print(f"AB-MAPPO Benchmark: K={K} M={M} steps={STEPS} epl={EPL}")
    print(f"CPU cores: {os.cpu_count()}")
    print("=" * 60)

    results = []

    print("\n[1/5] CPU single (default threads)...")
    t, eps, rc = run_bench("cpu_default", "cpu")
    results.append(("CPU single (default)", t, eps, rc))

    print("[2/5] CPU single (OMP=1)...")
    t, eps, rc = run_bench("cpu_omp1", "cpu", blas_threads=1)
    results.append(("CPU single (OMP=1)", t, eps, rc))

    print("[3/5] CUDA single...")
    t, eps, rc = run_bench("cuda", "cuda")
    results.append(("CUDA single", t, eps, rc))

    print("[4/5] CUDA single (OMP=1)...")
    t, eps, rc = run_bench("cuda_omp1", "cuda", blas_threads=1)
    results.append(("CUDA single (OMP=1)", t, eps, rc))

    print("[5/5] CPU x8 parallel (OMP=1)...")
    t_par = run_parallel(8)
    results.append(("CPU x8 parallel (OMP=1)", t_par, f"{8*STEPS/EPL/t_par:.2f} total", 0))

    print("\n" + "=" * 60)
    print(f"{'Config':<30} {'Time(s)':>8} {'eps/s':>10} {'rc':>3}")
    print("-" * 60)
    for label, t, eps, rc in results:
        print(f"{label:<30} {t:>8.1f} {str(eps):>10} {rc:>3}")
    print("=" * 60)

    successful = [r for r in results if r[3] == 0]
    if not successful:
        raise RuntimeError("No successful benchmark runs; cannot select best config.")
    best = min(successful, key=lambda x: x[1])
    print(f"\n最快方案: {best[0]} ({best[1]:.1f}s)")


if __name__ == "__main__":
    main()
