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

- 第五轮（并行稳定性修复）完成：
  - `run_all.py` 并行执行器从 `ThreadPoolExecutor + subprocess.run` 改为显式 `subprocess.Popen` 调度，修复失败场景下无法快速止损的问题。
  - 新增 fail-fast 机制：任一并行子任务非零退出时，立即终止其余运行中的子任务并抛错返回。
  - 新增单任务超时控制：
    - CLI 参数 `--job-timeout-sec`（默认 `0`，表示禁用超时）；
    - 超时后终止该任务并清理其它活动任务。
  - 修复并发日志覆盖风险：
    - 每次 `run_all.py --stage full` 运行创建独立日志目录 `run_logs/full_<timestamp>_<pid>_<uuid>/`；
    - 避免多次并行运行互相覆盖 `full_fig*.log`。
  - 新增辅助函数：
    - `_build_full_log_dir`、`_run_parallel_commands`、`_spawn_logged_process`、`_terminate_job` 等，增强可测试性与资源清理完整性。

- 第五轮新增测试：
  - `tests/test_run_all_parallel_stability.py`
    - `test_build_full_log_dir_is_unique`
    - `test_fail_fast_terminates_other_jobs`
    - `test_job_timeout_raises_error`

- 第五轮回归与验收（均在 `.venv`）：
  - `./.venv/bin/python -m unittest tests.test_run_all_full_device tests.test_run_all_parallel_stability -v`：5/5 通过。
  - `./.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v`：31/31 通过。
  - `./.venv/bin/python -m pytest -q`：31 passed。
  - `./.venv/bin/python run_all.py --help`：确认新增参数 `--job-timeout-sec` 已生效。

- 第六轮（并行吞吐 P0 修复）完成：
  - `train_sweep.py`：
    - 新增 `build_run_specs(fig, options)`，统一生成 `base/6/7/8/9/10/11/12` 全量 run 规格（`dict` 结构）。
    - run 规格字段包含 `fig/algorithm/setting_name/seed/run_dir/summary_path/cli_overrides/num_mus/num_uavs`，供外部调度器直接消费。
  - `run_all.py`：
    - `run_full` 从 fig 级并行改为全局 run 级并行，按 `num_mus * num_uavs` 降序提交任务，减少尾部拖慢。
    - 新增自动线程预算注入：`OMP_NUM_THREADS/MKL_NUM_THREADS/OPENBLAS_NUM_THREADS/NUMEXPR_NUM_THREADS`。
    - 并行度规则改为：`full-device` 包含 `cuda` 时强制 `effective_parallel=1`，否则使用 `max_parallel`。
    - 训练结束后追加一次 `train_sweep.py --fig all --resume --skip_existing` 聚合，再执行 `generate_figures.py --figs all`。
  - 新增测试：
    - `tests/test_train_sweep_run_specs.py`：校验 full run 总数 `555`，并抽查 `fig10=108`、`fig11=72`。
    - `tests/test_run_all_run_level.py`：校验 run 级 job 数量、CUDA 单并行限制、线程预算环境注入。

- 第七轮（训练侧计算路径优化）完成：
  - `mappo.py`：
    - `get_values`（Attention 分支）改为纯 `torch` 路径，使用 `F.pad + torch.cat` 构建 critic 输入，移除 `_build_attention_input` 的 `numpy` 临时拼接与额外拷贝。
    - `update` 在 epoch 循环外预构建 `all_obs_t/all_ret_t`，Attention critic 更新阶段不再重复申请 `torch.zeros` 与重复拼接返回目标。
  - `networks.py`：
    - `AttentionCritic.forward` 优先使用 `torch.nn.functional.scaled_dot_product_attention`；
    - 对旧版 torch 自动回退到原手写 `matmul + softmax`，保持兼容性；
    - 清理不再使用的 `numpy` 依赖导入。

- 第七轮回归与验收（均在 `.venv`）：
  - `./.venv/bin/python -m py_compile mappo.py networks.py`：通过。
  - `./.venv/bin/python -m unittest tests.test_mappo_critic_state_consistency tests.test_networks_stability -v`：2/2 通过。
  - `./.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v`：35/35 通过。
  - `./.venv/bin/python -m pytest -q`：35 passed。

- 第八轮（定向修复 #3/#2/#8 + 语义注释）完成：
  - `environment.py`
    - `_apply_wo_dt_noise()` 补齐 MU `task_data` 字段噪声注入（此前仅对 `task_cpu` 加噪声）。
    - 在 UAV 奖励中补充注释，说明扣减关联 MU 能量/时延属于协同 shaping 设计，若需严格论文拆解可后续参数化。
  - `buffer.py`
    - `RolloutBuffer.add()` 增加越界保护：当 `pos >= buffer_size` 时主动抛出 `RuntimeError`，错误信息包含 `overflow` 与 `reset` 提示。
  - `bench_parallel.py` / `benchmark_device.py`
    - 顶层执行逻辑下沉到 `main()`，并增加 `if __name__ == "__main__": main()` 守护；
    - 导入模块不再触发 benchmark 训练子进程。
  - `mappo.py`
    - 补充设计意图注释：联合 advantage 归一化是 baseline 选择；
    - MLP critic 标量 value 广播到各 agent 是有意的 centralized scalar 设计。

- 第八轮新增测试：
  - `tests/test_buffer_overflow_guard.py`
  - `tests/test_environment_wo_dt_task_data_noise.py`
  - `tests/test_benchmark_main_guard.py`

- 第八轮回归与验收（均在 `.venv`）：
  - `./.venv/bin/python -m unittest tests.test_buffer_overflow_guard tests.test_environment_wo_dt_task_data_noise tests.test_benchmark_main_guard -v`：4/4 通过。
  - `./.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v`：39/39 通过。
  - `./.venv/bin/python -m pytest -q`：39 passed。

- 第九轮（8 项性能优化全量落地）完成：
  - `mappo.py`
    - 新增 `_obs_to_tensors` / `_actions_from_tensors` / `_values_from_tensors` / `get_actions_and_values`，将每步动作与价值估计复用同一份观测 tensor。
    - `get_actions` 改为 `torch.as_tensor(..., device=...)` 路径，去除 `FloatTensor(...).to(...)` 的重复拷贝。
    - `get_values` 在 MLP critic 分支直接走 state value 计算，不再构建无用的 obs tensor。
    - `collect_episode` 改用融合接口，移除 `get_actions + get_values` 双调用。
    - `collect_episode` 统计改为循环内累计，替代多次 list comprehension 二次遍历。
  - `buffer.py`
    - `RolloutBuffer.get_batches` 全量 batch 路径改为切片视图 + `torch.from_numpy` 零拷贝。
    - 子采样路径保留随机索引行为，返回结构不变。
  - `environment.py`
    - `np.add.at` 改为 `np.bincount` 聚合 UAV 计算能耗。
    - 新增 `_penalty_distance_from_assoc`，在 `step` 中预计算 `assoc_mus_list` 复用，消除重复 `np.where` 扫描。
    - `_apply_wo_dt_noise` 的 UAV per-MU 噪声改为 reshape 向量化。
    - `_get_observations` 中 `other_uav` 构造改为掩码向量化。
  - 新增测试：
    - `tests/test_buffer_get_batches_zero_copy.py`
    - `tests/test_mappo_fused_actions_values.py`
    - `tests/test_environment_vectorized_paths.py`

- 第九轮回归与验收（均在 `.venv`）：
  - `./.venv/bin/python -m unittest tests.test_buffer_get_batches_zero_copy tests.test_mappo_fused_actions_values tests.test_environment_vectorized_paths -v`：5/5 通过。
  - `./.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v`：44/44 通过。
  - `./.venv/bin/python -m pytest -q`：44 passed。
  - 轻量性能验收（CPU，2400 steps，3 次）：
    - 命令：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ./.venv/bin/python train.py --algorithm AB-MAPPO --num_mus 60 --num_uavs 10 --total_steps 2400 --episode_length 300 --device cpu --seed 42 --disable_tensorboard --run_dir /tmp/ab_mappo_perf_cpu_<i>`
    - 末轮 `eps/s`：`0.62 / 0.61 / 0.62`，中位数 `0.62`。
  - 轻量性能验收（CUDA）：
    - 同参数 `--device cuda` 运行失败，报错 `AssertionError: Torch not compiled with CUDA enabled`（当前 `.venv` 为 CPU 版 PyTorch）。

- 第十轮（针对审核剩余点的性能收尾）完成：
  - `environment.py`
    - 将 `assoc_mus_list` 提前到资源分配阶段构建，资源分配与奖励阶段共用，避免对 `association` 的重复扫描。
    - 去除 `l_all/c_all` 的冗余 `astype(np.float32)` 副本，直接复用 `self.task_data/self.task_cpu`。
    - `mu_obs` 中 UAV 位置填充由 `np.tile` 改为广播赋值，减少额外分配。
  - `mappo.py`
    - `_values_from_state` 新增可选 `state` 参数，MLP critic 路径可复用 `collect_episode` 已计算状态。
    - `get_actions_and_values(..., state=...)` 支持传入预计算 state；`collect_episode` 传入每步 state，避免每步重复 `env.get_state()` 拼接。
    - 清理 `update()` 中未使用的 `mu_ret` 变量。
  - `buffer.py`
    - `compute_gae` 改为预计算 `non_terminal/next_values/delta`，移除循环内 `if t == size-1` 分支，保持递推语义不变。
  - 新增测试：
    - `tests/test_mappo_fused_actions_values.py` 新增用例：验证 MLP 路径每步只调用一次 `get_state`（外加 rollout 末尾 bootstrap 一次）。

- 第十轮回归与验收（均在 `.venv`）：
  - `./.venv/bin/python -m unittest tests.test_mappo_fused_actions_values tests.test_environment_vectorized_paths tests.test_buffer_gae_done_mask tests.test_environment_remaining -v`：8/8 通过。
  - `./.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v`：45/45 通过。
  - `./.venv/bin/python -m pytest -q`：45 passed。

- 第十一轮（性能残余问题修复）完成：
  - `mappo.py`
    - `collect_episode` 中 `state = self.env.get_state().copy()` 改为 `state = self.env.get_state()`，去除冗余拷贝。
  - `environment.py`
    - `_get_observations` 中位置归一化改为单次广播分配：`mu_pos_norm/uav_pos_norm` 由 `self.*_positions / norm_xy` 直接计算。
    - `_get_observations` 中移除 `per_mu` 中间数组，改为直接写入 `uav_obs` 切片 reshape 视图。
    - `other_uav` 构建去除 `np.tile`，改用 `mask` 的列索引重排（`j = np.where(mask)[1]`）形成 off-diagonal 视图。
    - UAV 奖励 cooperative 惩罚改为 `np.bincount` 向量化（`mu_energy_sum_per_uav + latency_sum_per_uav` 后除 `n_assoc_per_uav`），循环内仅保留边界/距离/碰撞项扣减。
    - `get_state()` 中缓存 `w/h`，避免重复 `max(self.area_width/height, 1.0)`。
  - `tests`
    - `tests/test_mappo_fused_actions_values.py`：新增/调整 `test_collect_episode_state_without_copy_semantics`。
    - `tests/test_environment_vectorized_paths.py`：新增 `test_cooperative_penalty_vector_matches_loop_reference`，验证向量化与循环参考一致。

- 第十一轮回归与验收（均在 `.venv`）：
  - `./.venv/bin/python -m unittest tests.test_mappo_fused_actions_values tests.test_environment_vectorized_paths -v`：5/5 通过。
  - `./.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v`：46/46 通过。
  - `./.venv/bin/python -m pytest -q`：46 passed。
  - 轻量性能验收（CPU，2400 steps，3 次）：
    - 命令：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ./.venv/bin/python train.py --algorithm AB-MAPPO --num_mus 60 --num_uavs 10 --total_steps 2400 --episode_length 300 --device cpu --seed 42 --disable_tensorboard --run_dir /tmp/ab_mappo_perf11_cpu_<i>`
    - 末轮 `eps/s`：`0.62 / 0.61 / 0.62`，中位数 `0.62`（不低于当前基线）。

- 第十二轮（性能残余补强）完成：
  - `mappo.py`
    - `update()` 与 critic 更新中 `optimizer.zero_grad(set_to_none=True)` 全量启用（MU actor / UAV actor / critic）。
  - `environment.py`
    - 在 `__init__` 缓存常量：
      - `_norm_xy`、`_task_data_scale`、`_task_cpu_scale`、`_edge_load_scale`、
      - `_dist_penalty_scale`、`_uav_velocity_scale`、
      - `_other_uav_col_idx`（M>1 时）。
    - `_penalty_distance_from_assoc` 使用 `_dist_penalty_scale`，避免每次重算比例常量。
    - `_get_observations` 使用缓存 scale：
      - `task_data_norm/task_cpu_norm` 改为 float32 乘法（去 `astype`）；
      - `prev_edge_load` 归一化改为乘 `_edge_load_scale`；
      - `other_uav` 直接用 `_other_uav_col_idx` 重排（不再每步构造 mask/index）。
    - UAV 奖励段中将 `boundary_penalties` 与 `collision_penalties` 提到循环外向量减法；
      循环仅保留 distance penalty 的逐 UAV 扣减。
    - `get_state()` 改为复用缓存 scale，并保持原状态向量顺序不变。
  - `tests`
    - `tests/test_mappo_fused_actions_values.py` 新增
      `test_update_uses_set_to_none_for_zero_grad`。
    - `tests/test_environment_vectorized_paths.py` 新增
      `test_get_state_matches_reference_formula`，确保状态公式与旧实现一致。

- 第十二轮回归与验收（均在 `.venv`）：
  - `./.venv/bin/python -m unittest tests.test_mappo_fused_actions_values tests.test_environment_vectorized_paths -v`：7/7 通过。
  - `./.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v`：48/48 通过。
  - `./.venv/bin/python -m pytest -q`：48 passed。

- 第十三轮（全面审查后问题收敛修复）完成：
  - `bench_parallel.py`
    - `run_parallel()` 将子进程等待从 `wait()` 改为 `communicate()`，避免 `stdout/stderr=PIPE` 场景的潜在阻塞。
    - 增加子进程返回码检查，任一并行任务失败即抛 `RuntimeError` 并带错误摘要。
    - 修正文档注释中的步数描述（实际 `STEPS=600`）。
  - `benchmark_device.py`
    - `run_parallel()` 同步改为 `communicate()` + 返回码校验，避免 silent failure。
    - `main()` 选择“最快方案”时只在 `rc==0` 的成功结果里比较耗时；若全失败则显式报错。
  - `mappo.py`
    - `RunningMeanStd.update()` 保持“每时间步一个样本”语义，`batch_count=1` 时将 `batch_var` 改为 `0.0`，避免跨 agent 方差注入。
    - MLP critic 中 `torch.allclose(mu_states, uav_states)` 改为默认关闭，仅在环境变量 `ABMAPPO_STRICT_STATE_CHECK=1/true/yes` 时启用，减少 CUDA 同步开销。

- 第十三轮测试补充与回归：
  - 更新 `tests/test_benchmark_main_guard.py`：
    - 新增并行 benchmark 子进程 `communicate`/返回码处理测试；
    - 新增 `benchmark_device` 仅在成功结果中选 best 的测试。
  - 更新 `tests/test_mappo_fused_actions_values.py`：
    - 新增 `test_mlp_update_does_not_require_allclose_by_default`，确保默认不触发全量 `allclose` 同步校验。
  - 更新 `tests/test_running_mean_std_remaining.py`：
    - 新增 `test_vector_reward_step_update_ignores_cross_agent_variance`，覆盖每步统计语义。

- 第十三轮验收（均在 `.venv`）：
  - `./.venv/bin/python -m unittest tests.test_benchmark_main_guard tests.test_mappo_fused_actions_values tests.test_running_mean_std_remaining -v`：11/11 通过（先红后绿完成 TDD）。
  - `./.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v`：53/53 通过。
  - `./.venv/bin/python -m pytest -q`：53 passed。

- 第十四轮（发烧级性能优化落地：JIT/缓存/AMP/Compile）完成：
  - `channel_model.py`
    - 新增可选 Numba JIT 批量速率核 `_compute_mu_uav_rate_batch_jit`（`njit(cache=True, fastmath=True)`）。
    - `compute_mu_uav_rate_batch()` 保持原接口，优先走 JIT，缺失 numba 时自动回退 numpy 路径（不影响可运行性）。
  - `buffer.py`
    - 新增可选高性能 GAE 实现入口：`_compute_gae_numba`（可用时）与 `_compute_gae_numpy`（回退）。
    - `RolloutBuffer.__init__` 增加 `self._compute_gae_impl` 动态绑定。
    - `compute_gae()` 改为统一调用 `_compute_gae_impl`，保持 done-mask 语义不变。
  - `environment.py`
    - 在 `__init__` 新增 `step()` 热路径预分配缓存：
      `_bw_alloc_cache/_cpu_alloc_cache/_edge_load_cache/_t_loc_cache/_e_loc_cache/_mu_energy_cache/_t_edge_cache/_uav_comp_cache/_uav_total_cache/_latency_penalty_cache/_mu_rewards_cache/_uav_rewards_cache`。
    - `step()` 中资源分配、时延/能耗、奖励计算切换为缓存复用（`.fill(0)` + 原地写入）。
    - 修复缓存复用下的状态一致性：
      `self.prev_edge_load[:] = edge_load`（避免与 cache alias），`self.prev_offload_ratio[:] = offload_ratio`。
    - 为避免外部引用被下一步覆盖，`info` 与 `rewards` 中数组字段在返回时 `copy()`。
  - `mappo.py`
    - 新增 AMP 基础设施：
      - `self.use_amp`（仅 CUDA 启用）
      - `self.scaler`（`torch.amp.GradScaler`，旧接口回退兼容）
      - `_autocast_ctx()` 统一上下文（CPU 为 `nullcontext`）
    - `update()` 中 MU/UAV actor 与 critic 更新接入 autocast + GradScaler：
      `scale(loss).backward()` → `unscale_` → 梯度裁剪 → `scaler.step(...)`。
    - 每个 PPO epoch 末调用 `self.scaler.update()`。
    - CUDA + 非 Windows 下对 critic 启用 `torch.compile`（失败自动回退，不阻塞训练）。
  - `requirements.txt`
    - 新增 `numba>=0.57.0,<1.0`（用于启用上述 JIT 路径）。

- 第十四轮新增测试（TDD 先红后绿）：
  - `tests/test_performance_runtime_features.py`
    - `test_environment_has_preallocated_step_caches`
    - `test_environment_prev_edge_load_not_aliased_to_step_cache`
    - `test_mappo_has_amp_scaler`
    - `test_rollout_buffer_has_fast_gae_path`

- 第十四轮验收（均在 `.venv`）：
  - 定向：`./.venv/bin/python -m unittest tests.test_performance_runtime_features tests.test_buffer_gae_done_mask tests.test_channel_model_batch_rate tests.test_mappo_fused_actions_values tests.test_environment_vectorized_paths -v`：14/14 通过。
  - 全量：`./.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v`：57/57 通过。
  - 全量：`./.venv/bin/python -m pytest -q`：57 passed。
  - 运行确认（本机 CPU 短基准，2400 steps）：
    - `AB-MAPPO` 末轮 `eps/s`：`0.60 / 0.61`。
    - 当前 `.venv` 下 `numba_jit_enabled=False`（未安装 numba，自动回退 numpy 路径，功能正常）。
