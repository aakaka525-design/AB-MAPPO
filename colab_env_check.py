"""
Google Colab 环境检测。

使用方式：在 Colab Notebook 的第一个 Cell 中粘贴以下内容并运行：

# ── Cell 1: 克隆项目 ──
# !git clone https://<your-repo-url> AB_MAPPO
# %cd AB_MAPPO

# ── Cell 2: 环境检测 ──
# %run colab_env_check.py

或者直接将下面的代码复制到一个 Cell 中运行。
"""
# fmt: off
# ────────────────────────────────────────────────
#  复制下面所有内容到 Colab Cell 中运行
# ────────────────────────────────────────────────

import os, sys, platform, subprocess

def _cmd(c):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.STDOUT).strip()
    except: return "(N/A)"

print("=" * 50)
print("  AB-MAPPO Colab 环境检测")
print("=" * 50)

# ── 系统 ──
print(f"\n{'─'*20} 系统 {'─'*20}")
print(f"Platform : {platform.platform()}")
print(f"Python   : {sys.version.split()[0]}")
print(f"CPU cores: {os.cpu_count()}")
print(f"Memory   : {_cmd('free -h | grep Mem')}")
print(f"Disk     : {_cmd('df -h / | tail -1')}")

# ── GPU ──
print(f"\n{'─'*20} GPU {'─'*21}")
smi = _cmd("nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader")
if "N/A" in smi or "failed" in smi.lower():
    print("⚠️  未检测到 GPU → Runtime → Change runtime type → GPU")
else:
    print(f"GPU      : {smi}")

try:
    import torch
    print(f"PyTorch  : {torch.__version__}")
    print(f"CUDA     : {torch.cuda.is_available()} (ver {getattr(torch.version, 'cuda', 'N/A')})")
    if torch.cuda.is_available():
        print(f"GPU name : {torch.cuda.get_device_name(0)}")
        gb = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"GPU mem  : {gb:.1f} GB")
        # Beta 采样兼容性测试（本项目关键 op）
        from torch.distributions import Beta
        s = Beta(torch.tensor([2.0], device="cuda"), torch.tensor([3.0], device="cuda")).sample()
        print(f"Beta采样 : ✅ (CUDA 原生支持)")
except ImportError:
    print("PyTorch  : ❌ 未安装")

# ── 依赖 ──
print(f"\n{'─'*20} 依赖 {'─'*20}")
for pkg in ["torch", "numpy", "matplotlib", "tensorboard"]:
    try:
        m = __import__(pkg); print(f"  ✅ {pkg:15s} {getattr(m, '__version__', '?')}")
    except ImportError:
        print(f"  ❌ {pkg:15s} → pip install {pkg}")

# ── 项目文件 ──
print(f"\n{'─'*20} 项目 {'─'*20}")
root = os.path.dirname(os.path.abspath("__file__")) if os.path.exists("config.py") else "."
for f in ["config.py","train.py","environment.py","mappo.py","networks.py","buffer.py",
          "maddpg.py","channel_model.py","train_sweep.py","generate_figures.py","run_all.py"]:
    print(f"  {'✅' if os.path.exists(os.path.join(root,f)) else '❌'} {f}")

# ── 推荐 ──
print(f"\n{'─'*20} 推荐 {'─'*20}")
try:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
except: dev = "cpu"
cores = os.cpu_count() or 2
par = min(4, max(1, cores // 2))
print(f"设备: {dev} | 核心: {cores} | 建议并行: {par}")
print(f"\n# Smoke 测试")
print(f"!python run_all.py --stage smoke")
print(f"\n# Full Sweep")
print(f"!python run_all.py --stage full --max_parallel {par}")
if dev == "cuda":
    print(f"\n# 单次 GPU 训练")
    print(f"!python train.py --algorithm AB-MAPPO --device cuda --total_steps 80000")
print("\n" + "=" * 50)
print("  检测完成")
print("=" * 50)
