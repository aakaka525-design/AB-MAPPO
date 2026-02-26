"""
GPU vs CPU 对比 profiling — 在服务器上直接运行
用法: python gpu_profile.py
"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from environment import UAVMECEnv
from mappo import ABMAPPO
import config as cfg


def profile_device(device_str, num_episodes=5, k=60, m=10):
    print(f"\n{'='*60}")
    print(f"Profiling: device={device_str} K={k} M={m} episodes={num_episodes}")
    print(f"{'='*60}")

    env = UAVMECEnv(num_mus=k, num_uavs=m, seed=42)
    agent = ABMAPPO(env, algorithm="AB-MAPPO", device=device_str)

    # Warmup 1 episode (includes torch.compile if CUDA)
    print("Warmup...")
    t_warmup = time.perf_counter()
    agent.collect_episode()
    agent.update()
    warmup_time = time.perf_counter() - t_warmup
    print(f"  Warmup: {warmup_time:.2f}s (includes compilation if any)")

    # 分模块计时
    t_collect = 0.0
    t_update = 0.0
    t_total = time.perf_counter()

    for ep in range(num_episodes):
        t0 = time.perf_counter()
        stats = agent.collect_episode()
        t_collect += time.perf_counter() - t0

        t0 = time.perf_counter()
        agent.update()
        t_update += time.perf_counter() - t0

    total = time.perf_counter() - t_total
    eps_s = num_episodes / total

    print(f"\n  结果 ({num_episodes} episodes, 不含 warmup):")
    print(f"    总耗时:     {total:.2f}s")
    print(f"    eps/s:      {eps_s:.2f}")
    print(f"    collect:    {t_collect:.2f}s ({t_collect/total*100:.1f}%)")
    print(f"    update:     {t_update:.2f}s ({t_update/total*100:.1f}%)")
    print(f"    每episode:  collect={t_collect/num_episodes:.3f}s + update={t_update/num_episodes:.3f}s")

    if "cuda" in device_str and torch.cuda.is_available():
        print(f"    GPU 显存峰值: {torch.cuda.max_memory_allocated()/1024/1024:.1f} MB")
        torch.cuda.reset_peak_memory_stats()

    return eps_s


def profile_parallel_gpu(num_jobs, num_episodes=3):
    """模拟多个 GPU job 并行（用 torch.multiprocessing）"""
    import subprocess
    print(f"\n{'='*60}")
    print(f"模拟 {num_jobs} 个 CUDA job 并行")
    print(f"{'='*60}")

    script = f'''
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, ".")
from environment import UAVMECEnv
from mappo import ABMAPPO
env = UAVMECEnv(num_mus=60, num_uavs=10, seed=int(sys.argv[1]))
agent = ABMAPPO(env, algorithm="AB-MAPPO", device="cuda")
agent.collect_episode(); agent.update()  # warmup
t0 = time.time()
for _ in range({num_episodes}):
    agent.collect_episode(); agent.update()
print(f"seed={{sys.argv[1]}} eps/s={{{num_episodes}/(time.time()-t0):.2f}}")
'''

    procs = []
    t0 = time.perf_counter()
    for i in range(num_jobs):
        p = subprocess.Popen(
            [sys.executable, "-c", script, str(42 + i)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env={**os.environ, "OMP_NUM_THREADS": "1"}
        )
        procs.append(p)

    for p in procs:
        stdout, stderr = p.communicate()
        out = stdout.decode().strip()
        if out:
            print(f"  {out}")
        if p.returncode != 0:
            err = stderr.decode().strip()[-200:]
            print(f"  ERROR: {err}")

    wall = time.perf_counter() - t0
    print(f"  {num_jobs} 并行总耗时: {wall:.1f}s | 总吞吐: {num_jobs * num_episodes / wall:.2f} eps/s")


if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info()
        print(f"VRAM: {free/1024**3:.1f}GB free / {total/1024**3:.1f}GB total")

    # 1. CPU 基准
    cpu_eps = profile_device("cpu", num_episodes=5, k=60, m=10)

    # 2. GPU 基准
    if torch.cuda.is_available():
        gpu_eps = profile_device("cuda", num_episodes=5, k=60, m=10)
        print(f"\n  GPU/CPU 加速比: {gpu_eps/cpu_eps:.2f}x")

        # 3. GPU 多进程并行
        for n in [2, 4, 8]:
            try:
                profile_parallel_gpu(n, num_episodes=3)
            except Exception as e:
                print(f"  {n} 并行失败: {e}")

    # 4. K=100 大配置
    print("\n\n=== K=100 大配置 ===")
    profile_device("cpu", num_episodes=3, k=100, m=10)
    if torch.cuda.is_available():
        profile_device("cuda", num_episodes=3, k=100, m=10)
