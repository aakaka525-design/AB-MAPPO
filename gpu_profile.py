"""
GPU vs CPU 深度 profiling — 更多数据点，更长测试
用法: python gpu_profile.py
"""
import os, sys, time, subprocess
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from environment import UAVMECEnv
from mappo import ABMAPPO
import config as cfg


def profile_device(device_str, num_episodes=20, k=60, m=10):
    print(f"\n{'='*60}")
    print(f"device={device_str} K={k} M={m} episodes={num_episodes}")
    print(f"{'='*60}")

    env = UAVMECEnv(num_mus=k, num_uavs=m, seed=42)
    agent = ABMAPPO(env, algorithm="AB-MAPPO", device=device_str)

    # Warmup 2 episodes
    for _ in range(2):
        agent.collect_episode()
        agent.update()
    if "cuda" in device_str:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t_collect = 0.0
    t_update = 0.0
    t_total = time.perf_counter()

    for ep in range(num_episodes):
        if "cuda" in device_str:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        agent.collect_episode()
        if "cuda" in device_str:
            torch.cuda.synchronize()
        t_collect += time.perf_counter() - t0

        t0 = time.perf_counter()
        agent.update()
        if "cuda" in device_str:
            torch.cuda.synchronize()
        t_update += time.perf_counter() - t0

    total = time.perf_counter() - t_total
    eps_s = num_episodes / total

    print(f"  总耗时:     {total:.2f}s | eps/s: {eps_s:.3f}")
    print(f"  collect:    {t_collect:.2f}s ({t_collect/total*100:.1f}%) per_ep={t_collect/num_episodes:.3f}s")
    print(f"  update:     {t_update:.2f}s ({t_update/total*100:.1f}%) per_ep={t_update/num_episodes:.3f}s")
    if "cuda" in device_str and torch.cuda.is_available():
        print(f"  GPU peak:   {torch.cuda.max_memory_allocated()/1024/1024:.1f} MB")
    return eps_s


def run_parallel_gpu(num_jobs, num_episodes=15, k=60, m=10):
    """多进程 GPU 并行：每个进程独立完整测试"""
    print(f"\n{'='*60}")
    print(f"CUDA x{num_jobs} 并行 | K={k} M={m} | {num_episodes} eps/job")
    print(f"{'='*60}")

    script = f'''
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, ".")
import torch
from environment import UAVMECEnv
from mappo import ABMAPPO
env = UAVMECEnv(num_mus={k}, num_uavs={m}, seed=int(sys.argv[1]))
agent = ABMAPPO(env, algorithm="AB-MAPPO", device="cuda")
# warmup 2 episodes
for _ in range(2):
    agent.collect_episode(); agent.update()
torch.cuda.synchronize()
t0 = time.time()
for _ in range({num_episodes}):
    agent.collect_episode(); agent.update()
torch.cuda.synchronize()
elapsed = time.time() - t0
print(f"seed={{sys.argv[1]}} eps/s={{{num_episodes}/elapsed:.3f}} time={{elapsed:.1f}}s")
'''
    procs = []
    t0 = time.perf_counter()
    for i in range(num_jobs):
        p = subprocess.Popen(
            [sys.executable, "-c", script, str(42 + i)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env={**os.environ, "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
        )
        procs.append(p)

    results = []
    for p in procs:
        stdout, stderr = p.communicate()
        out = stdout.decode().strip()
        if out:
            print(f"  {out}")
            # parse eps/s
            for part in out.split():
                if part.startswith("eps/s="):
                    results.append(float(part.split("=")[1]))
        if p.returncode != 0:
            err = stderr.decode().strip()[-300:]
            print(f"  ERROR: {err}")

    wall = time.perf_counter() - t0
    total_eps = num_jobs * num_episodes
    wall_throughput = total_eps / wall
    sum_throughput = sum(results) if results else 0

    print(f"  ---")
    print(f"  wall: {wall:.1f}s | 各job eps/s之和: {sum_throughput:.2f} | wall吞吐: {wall_throughput:.2f} eps/s")
    if results:
        print(f"  per-job平均: {np.mean(results):.3f} eps/s | min: {min(results):.3f} | max: {max(results):.3f}")
    return wall_throughput, sum_throughput


if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info()
        print(f"VRAM: {free/1024**3:.1f}GB free / {total/1024**3:.1f}GB total")
    print(f"CPU cores: {os.cpu_count()}")

    # ===== 1. 单 job 对比 (20 episodes 足够稳定) =====
    cpu60 = profile_device("cpu", num_episodes=20, k=60)
    gpu60 = None
    if torch.cuda.is_available():
        gpu60 = profile_device("cuda", num_episodes=20, k=60)
        print(f"\n  >>> GPU/CPU = {gpu60/cpu60:.2f}x")

    # ===== 2. K=100 单 job =====
    cpu100 = profile_device("cpu", num_episodes=10, k=100)
    gpu100 = None
    if torch.cuda.is_available():
        gpu100 = profile_device("cuda", num_episodes=10, k=100)
        print(f"\n  >>> GPU/CPU = {gpu100/cpu100:.2f}x")

    # ===== 3. GPU 并行扩展 (1,2,3,4,6,8,10,12) =====
    if torch.cuda.is_available():
        print(f"\n\n{'#'*60}")
        print(f"GPU 并行扩展测试 (K=60, 15 eps/job)")
        print(f"{'#'*60}")
        parallel_results = {}
        for n in [1, 2, 3, 4, 6, 8, 10, 12]:
            try:
                wall_tp, sum_tp = run_parallel_gpu(n, num_episodes=15, k=60)
                parallel_results[n] = (wall_tp, sum_tp)
            except Exception as e:
                print(f"  x{n} failed: {e}")

        # 汇总表
        print(f"\n{'='*60}")
        print(f"GPU 并行扩展汇总 (K=60)")
        print(f"{'='*60}")
        print(f"{'并行数':>6} | {'wall吞吐':>10} | {'各job之和':>10} | {'vs单GPU':>8}")
        print("-" * 45)
        base = parallel_results.get(1, (1, 1))[0]
        for n, (wall_tp, sum_tp) in sorted(parallel_results.items()):
            print(f"{n:>6} | {wall_tp:>9.2f} | {sum_tp:>9.2f} | {wall_tp/base:>7.2f}x")

    # ===== 4. CPU 并行扩展 =====
    print(f"\n\n{'#'*60}")
    print(f"CPU 并行扩展测试 (K=60, 10 eps/job)")
    print(f"{'#'*60}")
    cpu_parallel_results = {}
    for n in [1, 5, 10, 15, 20, 25, 30]:
        print(f"\n--- CPU x{n} ---")
        script = f'''
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, ".")
from environment import UAVMECEnv
from mappo import ABMAPPO
env = UAVMECEnv(num_mus=60, num_uavs=10, seed=int(sys.argv[1]))
agent = ABMAPPO(env, algorithm="AB-MAPPO", device="cpu")
agent.collect_episode(); agent.update()
t0 = time.time()
for _ in range(10):
    agent.collect_episode(); agent.update()
print(f"seed={{sys.argv[1]}} eps/s={{10/(time.time()-t0):.3f}}")
'''
        procs = []
        t0 = time.perf_counter()
        for i in range(n):
            p = subprocess.Popen(
                [sys.executable, "-c", script, str(42 + i)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env={**os.environ, "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
            )
            procs.append(p)

        results = []
        for p in procs:
            stdout, _ = p.communicate()
            out = stdout.decode().strip()
            if out:
                for part in out.split():
                    if part.startswith("eps/s="):
                        results.append(float(part.split("=")[1]))

        wall = time.perf_counter() - t0
        total_eps = n * 10
        wall_tp = total_eps / wall
        sum_tp = sum(results) if results else 0
        cpu_parallel_results[n] = (wall_tp, sum_tp)
        print(f"  wall: {wall:.1f}s | wall吞吐: {wall_tp:.2f} | per-job avg: {np.mean(results):.3f}")

    print(f"\n{'='*60}")
    print(f"CPU 并行扩展汇总")
    print(f"{'='*60}")
    print(f"{'并行数':>6} | {'wall吞吐':>10} | {'per-job avg':>12} | {'vs x1':>7}")
    print("-" * 50)
    cpu_base = cpu_parallel_results.get(1, (1, 1))[0]
    for n, (wall_tp, sum_tp) in sorted(cpu_parallel_results.items()):
        avg_per = sum_tp / n if n > 0 else 0
        print(f"{n:>6} | {wall_tp:>9.2f} | {avg_per:>11.3f} | {wall_tp/cpu_base:>6.2f}x")
