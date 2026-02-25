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
