"""
Global configuration for AB-MAPPO paper reproduction experiments.

Default values are aligned to:
Energy Efficient Computation Offloading in Aerial Edge Networks With Multi-Agent Cooperation
"""

from __future__ import annotations

import copy
import os
import numpy as np


# ============================================================
# Experiment meta
# ============================================================
EXPERIMENT_ROOT = os.environ.get("ABMAPPO_EXPERIMENT_ROOT", "experiments")
RESULT_DIR = os.environ.get("ABMAPPO_RESULT_DIR", "results")
DEFAULT_SEEDS = [42, 43, 44]

ALL_ALGOS = ["AB-MAPPO", "B-MAPPO", "AG-MAPPO", "MADDPG", "Randomized"]
CONVERGENCE_ALGOS = ["AB-MAPPO", "B-MAPPO", "AG-MAPPO", "MADDPG"]
COMPARISON_ALGOS = ["AB-MAPPO", "B-MAPPO", "AG-MAPPO", "MADDPG", "Randomized"]


# ============================================================
# Topology (paper defaults for main scenario)
# ============================================================
NUM_MUS = 60
NUM_UAVS = 10
AREA_WIDTH = 1000.0
AREA_HEIGHT = 1000.0
BS_POSITION = np.array([-500.0, 0.0, 10.0], dtype=np.float32)
UAV_HEIGHT = 200.0


# ============================================================
# Time
# ============================================================
TIME_PERIOD = 60.0
TIME_SLOT = 1.0
NUM_SLOTS = int(TIME_PERIOD / TIME_SLOT)  # 60 env steps per episode


# ============================================================
# Channel parameters (Table II)
# ============================================================
BANDWIDTH = 50e6
CHANNEL_POWER_GAIN_DB = -30
CHANNEL_POWER_GAIN = 10 ** (CHANNEL_POWER_GAIN_DB / 10)
ENV_PARAM_A = 15.0
ENV_PARAM_B = 0.5
PATH_LOSS_EXP = 2.2
NLOS_ATTENUATION = 0.2
NOISE_POWER_DENSITY_DBM = -127
NOISE_POWER_DENSITY = 10 ** ((NOISE_POWER_DENSITY_DBM - 30) / 10)

MU_TRANSMIT_POWER = 0.2
UAV_TRANSMIT_POWER = 0.5
UAV_RELAY_BANDWIDTH = 5e6


# ============================================================
# Computing parameters
# ============================================================
EFFECTIVE_CAPACITANCE = 1e-27
MU_MAX_CPU_FREQ = 1e9
UAV_MAX_CPU_FREQ = 10e9

TASK_DATA_MIN = 0.5e6
TASK_DATA_MAX = 2.0e6
TASK_CPU_CYCLES_MIN = 500.0
TASK_CPU_CYCLES_MAX = 1500.0


# ============================================================
# UAV mobility and propulsion
# ============================================================
UAV_MAX_VELOCITY = 30.0
UAV_MAX_ACCELERATION = 5.0

BLADE_PROFILE_POWER = 39.04
INDUCED_POWER = 79.07
TIP_SPEED = 120.0
MEAN_ROTOR_VELOCITY = 3.6
FUSELAGE_DRAG_RATIO = 0.012
ROTOR_SOLIDITY = 0.05
AIR_DENSITY = 1.225
ROTOR_DISC_AREA = 0.5030


# ============================================================
# MU mobility
# ============================================================
MU_MEMORY_FACTOR_V = 0.5
MU_MEMORY_FACTOR_THETA = 0.5
MU_MEAN_VELOCITY = 2.0
MU_MEAN_DIRECTION = 0.0
MU_VELOCITY_STD = 1.0
MU_DIRECTION_STD = 0.5


# ============================================================
# DT settings
# ============================================================
DT_DEVIATION_RATE = 0.0
WEIGHT_FACTOR = 0.001


# ============================================================
# MAPPO hyper-parameters (Table III aligned)
# NOTE:
# - The paper lists episode length 300 for training.
# - Environment still uses NUM_SLOTS=60 for physical episode horizon.
# - Training rollout can span multiple env episodes.
# ============================================================
TOTAL_STEPS = 80_000
EPISODE_LENGTH = 300

ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
GAMMA_UAV = 0.95
GAMMA_MU = 0.8
GAE_LAMBDA = 0.95
PPO_CLIP_EPSILON = 0.2
PPO_EPOCHS = 5
NUM_MINI_BATCHES = 1
HIDDEN_SIZE = 128
NUM_ATTENTION_HEADS = 4
ENTROPY_COEFF = 0.01
VALUE_LOSS_COEFF = 0.5
MAX_GRAD_NORM = 10.0


# ============================================================
# Reward / penalty
# ============================================================
# Paper scale is around 0.1 for penalty factors.
PENALTY_DELAY = 0.1
PENALTY_BOUNDARY = 0.1
PENALTY_COLLISION = 0.1
REWARD_SCALE = 10.0

D_MIN = 50.0
D_TH = 300.0


# ============================================================
# MADDPG hyper-parameters
# ============================================================
MADDPG_ACTOR_LR = 3e-4
MADDPG_CRITIC_LR = 3e-4
MADDPG_GAMMA = 0.95
MADDPG_TAU = 0.005
MADDPG_BUFFER_SIZE = 20_000
MADDPG_BATCH_SIZE = 256
MADDPG_WARMUP_STEPS = 2_000
MADDPG_UPDATES_PER_STEP = 1
MADDPG_NOISE_STD_INIT = 0.5
MADDPG_NOISE_STD_MIN = 0.05
MADDPG_NOISE_DECAY = 0.99995


# ============================================================
# Logging / checkpoints
# ============================================================
LOG_DIR = "logs"
SAVE_DIR = "checkpoints"
SAVE_INTERVAL = 5_000
LOG_INTERVAL = 100
EVAL_INTERVAL = 2_000


# ============================================================
# Figure grids (paper-aligned)
# ============================================================
FIGURE_SWEEPS = {
    "fig6": {
        "x_name": "num_mus",
        "x_values": [50, 60, 70, 80, 90, 100],
        "fixed": {"num_uavs": 10},
        "algorithms": COMPARISON_ALGOS,
        "metric": "weighted_energy_mu_total",
    },
    "fig7": {
        "x_name": "num_uavs",
        "x_values": [4, 6, 8, 10, 12, 14],
        "fixed": {"num_mus": 70},
        "algorithms": COMPARISON_ALGOS,
        "metric": "weighted_energy_mu_avg",
    },
    "fig8": {
        "x_name": "bandwidth_mhz",
        "x_values": [30, 40, 50, 60, 70],
        "fixed": {"num_mus": 60, "num_uavs": 10},
        "algorithms": COMPARISON_ALGOS,
        "metric": "weighted_energy_mu_avg",
    },
    "fig9": {
        "x_name": "weight_factor",
        "x_values": [0.0, 0.002, 0.004, 0.006, 0.008, 0.009],
        "fixed": {"num_mus": 60, "num_uavs": 10},
        "algorithms": ["AB-MAPPO"],
        "metric": "mu_uav_energy_pair",
    },
    "fig10": {
        "fixed": {"num_uavs": 10, "ks": [60, 80, 100]},
        "mu_cpu_ghz": [0.5, 1.0, 1.5, 2.0, 2.5],
        "uav_cpu_ghz": [2, 4, 6, 8, 10, 12, 14],
        "algorithms": ["AB-MAPPO"],
        "metric": "weighted_energy_mu_avg",
    },
    "fig11": {
        "fixed": {"num_uavs": 10, "ks": [50, 60, 70, 80]},
        "dt_values": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25],
        "algorithms": ["AB-MAPPO"],
        "metric": "weighted_energy_mu_avg",
    },
    "fig12": {
        "fixed": {"num_mus": 60, "num_uavs": 10},
        "task_max_mbits": [1, 1.5, 2, 2.5, 3],
        "bandwidth_mhz": [30, 50, 70],
        "algorithms": ["AB-MAPPO"],
        "metric": "weighted_energy_mu_avg",
    },
}


def paper_profile_dict() -> dict:
    """Return a copy of paper-aligned core defaults for external tools."""
    return {
        "NUM_MUS": NUM_MUS,
        "NUM_UAVS": NUM_UAVS,
        "BANDWIDTH": BANDWIDTH,
        "MU_MAX_CPU_FREQ": MU_MAX_CPU_FREQ,
        "UAV_MAX_CPU_FREQ": UAV_MAX_CPU_FREQ,
        "WEIGHT_FACTOR": WEIGHT_FACTOR,
        "TOTAL_STEPS": TOTAL_STEPS,
        "EPISODE_LENGTH": EPISODE_LENGTH,
    }


def clone_figure_sweeps() -> dict:
    return copy.deepcopy(FIGURE_SWEEPS)
