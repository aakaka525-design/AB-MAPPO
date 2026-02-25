# 工作日志

## 2026-02-25

- 已按修复计划先补充测试用例，覆盖 MU 反弹方向、GAE done-mask、MAPPO critic 状态一致性、`torch.load` 兼容回退。
- 用户要求已确认：依赖安装与测试执行均在项目虚拟环境中进行，不使用系统 Python 直接安装。
- 已创建项目虚拟环境 `.venv`，并在其中安装依赖与 `pytest`。
- 已先执行新增/修改测试并观察到预期失败：
  - `test_mu_direction_bounce_on_y_boundary` 失败（对应 MU 上下边界反弹方向问题）。
  - `test_mlp_critic_uses_buffered_state` 失败（对应 MLP critic 状态未入 buffer）。
  - 两个 checkpoint 兼容测试报错（对应 `torch.load` 无 `weights_only` 回退逻辑）。
- 已完成代码修复：
  - `environment.py`：MU x/y 边界反弹方向分别处理。
  - `buffer.py` + `mappo.py`：rollout 增加全局 `state` 存储与读取，MLP critic 训练使用 buffer state，并加一致性校验。
  - `mappo.py` + `maddpg.py`：`torch.load` 增加 `weights_only=True` 优先加载及 `TypeError` 回退。
  - `networks.py`：`GaussianActor` 逆 sigmoid 改为 `torch.logit(cont_action.clamp(...))`。
  - `requirements.txt`：补充主版本上限约束。
- 回归结果（均在 `.venv`）：
  - `python -m unittest discover -s tests -p "test_*.py" -v`：12/12 通过。
  - `python -m pytest -q`：12/12 通过。
- 已按新依赖上限重新执行 `python -m pip install -r requirements.txt`，`numpy` 已从 `2.4.2` 降级至 `1.26.4`，并再次通过 `unittest` 全量回归（12/12）。
- 第二轮“剩余问题”修复已开始，先新增测试并确认失败（TDD 红灯）：
  - `RunningMeanStd` 向量奖励输入时计数按步更新（此前 `count` 按元素数量累计）。
  - 环境批量速率 API `compute_mu_uav_rate_batch`（此前不存在）。
  - 环境实例级随机数（`seed`/`rng`）与可复现 reset（此前 `UAVMECEnv.__init__` 不支持 `seed`）。
  - 碰撞惩罚向量化接口 `_collision_penalties`（此前不存在）。
  - `step()` 对边缘卸载路径使用批量速率 API（此前仍走逐个 MU 的标量 API）。
- 第二轮代码修复内容：
  - `channel_model.py`：新增 `compute_mu_uav_rate_batch` 并做输入形状校验。
  - `environment.py`：
    - 引入 `seed`/`rng`，内部随机采样统一使用实例 `self.rng`。
    - `_get_observations` 改为向量化构建（减少 K×M 双层循环）。
    - 新增 `_collision_penalties` 并让 `_penalty_collision` 复用。
    - `step` 的能量/时延主路径改为批量计算，调用 `compute_mu_uav_rate_batch`。
  - `mappo.py`：`RunningMeanStd.update` 改为按时间步计数（每次更新 `count += 1`）。
  - `train.py`：创建环境时传入 `seed=args.seed`，使环境随机流与训练种子显式绑定。
  - `maddpg.py` + `config.py`：补充 warmup 单位说明（transition 级别，非 episode）。
  - `tests/test_environment_metrics.py`：边界反弹测试改为注入实例 `rng`，兼容新随机数架构。
  - 新增测试：`test_running_mean_std_remaining.py`、`test_channel_model_batch_rate.py`、`test_environment_remaining.py`。
- 第二轮回归结果（均在 `.venv`）：
  - `python -m unittest discover -s tests -p "test_*.py" -v`：17/17 通过。
  - `python -m pytest -q`：17/17 通过。

- 第三轮（出图脚本问题修复）按计划完成：
  - `generate_figures.py`
    - 新增 `--allow-missing`（默认严格缺失失败）与 `--fig13-seed`（Fig13 可复现）。
    - `main()` 增加目标图所需输入文件校验：缺失时默认 `SystemExit(2)`，传 `--allow-missing` 才按图 `skip`。
    - Fig10/Fig11 的 `K=...` 系列改为数值排序，避免 `K=100` 被排在 `K=60` 前面。
    - Fig13 启发式轨迹改为确定性随机流：`_run_heuristic_trajectory(..., seed=...)`，并为每个子图使用 `base_seed + idx`。
  - `run_all.py`
    - 新增 `--full-device`（默认 `cpu`）。
    - `full` 阶段命令显式注入 `--device <full-device>`，降低并行 GPU OOM 风险。
    - 提取 `_build_full_cmd()` 便于测试与复用。
  - `train_sweep.py`
    - `_aggregate_history()` 改为 `with np.load(...)` 上下文管理，避免 `NpzFile` 句柄滞留。

- 第三轮新增测试（先失败后修复）：
  - `tests/test_generate_figures_strict_mode.py`
  - `tests/test_generate_figures_fig13_determinism.py`
  - `tests/test_generate_figures_sorting.py`
  - `tests/test_run_all_full_device.py`
  - `tests/test_train_sweep_aggregate_history.py`

- 第三轮回归与验收（均在 `.venv`）：
  - `python -m unittest discover -s tests -p "test_*.py" -v`：26/26 通过。
  - `python -m pytest -q`：26 passed。
  - `python generate_figures.py --figs all`：在当前实验数据完整时成功出图（0 退出）。
  - `python generate_figures.py --figs all --exp_root <empty> --output_dir <tmp>`：严格模式缺输入时按预期返回 2。
  - `python generate_figures.py --figs all --exp_root <empty> --output_dir <tmp> --allow-missing`：按预期 0 退出并 `skip`。
  - `python generate_figures.py --figs 13 --fig13-seed 20260225` 连续两次 MD5 一致，确认 Fig13 可复现。

- 第四轮（Colab 兼容性与收尾检查）完成：
  - 修复 `tests/test_generate_figures_strict_mode.py` 的仓库绝对路径硬编码，改为 `Path(__file__).resolve().parents[1]` 动态定位仓库根目录，提升跨机器/CI/Colab 可移植性。
  - `generate_figures.py` 新增 `MPLCONFIGDIR` 自动初始化：
    - 新增 `_ensure_writable_mplconfigdir(candidates=None)`，若用户已设置 `MPLCONFIGDIR` 则保持不变；
    - 若未设置，优先尝试 `tempfile.gettempdir()/ab_mappo_mplconfig`，失败再回退 `os.getcwd()/.mplconfig`；
    - 通过创建目录并写入探针文件验证可写性；
    - 两个候选均失败时仅回退原行为，不中断出图流程。
  - 新增回归测试 `tests/test_generate_figures_mplconfig.py`：
    - 验证未设置 `MPLCONFIGDIR` 时可自动设置到可写目录；
    - 验证已设置 `MPLCONFIGDIR` 时不会被覆盖。

- 第四轮回归与验收（均在 `.venv`）：
  - `./.venv/bin/python -m unittest tests.test_generate_figures_mplconfig -v`：先红后绿（新增函数前失败，实装后通过）。
  - `./.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v`：28/28 通过。
  - `./.venv/bin/python -m pytest -q`：28 passed。
  - `./.venv/bin/python generate_figures.py --figs all --exp_root <empty_dir> --output_dir <tmp_out>`：返回码 2（严格模式符合预期）。
  - `./.venv/bin/python generate_figures.py --figs all --allow-missing --exp_root <empty_dir> --output_dir <tmp_out>`：返回码 0，且日志含 `[skip]`。
  - `HOME=<只读目录> MPLCONFIGDIR='' ./.venv/bin/python generate_figures.py --figs 13 --exp_root <empty_dir> --output_dir <tmp_out>`：返回码 0，且无 `~/.matplotlib` 不可写告警。
