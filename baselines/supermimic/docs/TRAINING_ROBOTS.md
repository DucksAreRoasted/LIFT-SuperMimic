# 多机器人训练

统一入口：`uv run --extra train sm-train --robot g1|r1|k1`。
运行前执行 `uv sync --extra train --locked`。R1/K1 可直接读取已有 50 Hz 重定向 NPZ。

| 机器人 | 关节数 | 状态维度 | 10 帧观测 | 10 帧目标 | 根 / 跟踪锚点 |
|---|---:|---:|---:|---:|---|
| G1 | 29 | 71 | 930 | 680 | pelvis / torso_link |
| R1 | 26 | 65 | 840 | 620 | pelvis_link / waist_yaw_link |
| K1 | 22 | 57 | 720 | 540 | trunk / trunk |

机器人定义统一在 `src/supermimic/robots/profiles.py`。状态、动作、目标、历史观测、仿真上下文、replay、世界模型和策略/critic 的尺寸随机器人变化。R1/K1 的头部关节保留为可控关节。

## 模型与控制

- G1 保持上游 mjlab 的资产、默认站姿、驱动及碰撞设置。
- R1/K1 从 `assets/retarget/` 的模型读取官方运动学和惯量，保留固定重定向标记，并在加载时修正当前工程的 mesh 路径。训练不修改原始 XML 或动作文件。
- R1 四肢/腰部 PD 使用本工程 `assets/raw/r1_constants.py` 的基线：腿/腰 100/2、踝 40/2、肩 pitch/roll 40/2、其余手臂 20/1；头部暂用 20/1 的仿真基线。扭矩限制从官方 URDF 转换模型读取，包括头部的独立限制。
- K1 扭矩和 armature 读取官方串联 22-DoF XML；PD 采用与 G1 一致的构造：`kp = armature * (2π*10)^2`，`kd = 4 * armature * (2π*10)`。这是可调整的仿真初始配置。
- 两种新机器人使用 `action_scale = 0.25 * effort_limit / kp`，控制周期 0.02 s、物理步长 0.005 s，力矩受限。网格仅供显示；足部盒体和四肢胶囊/躯干球体参与地面碰撞，不启用自碰撞。
- FK 与数据仍使用官方关节/刚体顺序；加载数据逐文件核对关节名称与顺序及 50 Hz 帧率。`--data.npz-dir` 必须属于所选机器人。

## 启动

R1 与 K1 的常用启动、续训、曲线及策略回放命令见 [命令记录](../命令记录.md)。
未指定数据目录时，R1/K1 优先找 `data/retarget/lafan1_modelscope_v1/<robot>`，其次找已有的 `lafan1_official_v1/<robot>`；G1 默认 `data/npz/lafan1`。

`--resume` 要与 `--robot` 一致，程序拒绝跨机器人加载。旧 checkpoint 缺少机器人字段时按 G1 处理。`scripts/play.py`、`scripts/eval_benchmark.py` 和轨迹录制从 checkpoint 自动确定机器人；不指定数据目录时同样使用上述默认规则。

## 验证

```bash
# CPU：各机器人 FK 对照 MuJoCo、状态集成、数据加载和策略梯度
env -u PYTHONPATH uv run --extra train pytest tests/test_multi_robot.py -k 'not cuda' -q

# GPU：短训练、策略权重更新、保存/加载和机器人身份保护
env -u PYTHONPATH uv run --extra train pytest tests/test_multi_robot.py -k cuda -q
```

GPU 测试每种机器人使用 4 个环境、25 步、缩小的网络，覆盖普通执行及带随机化的 `torch.compile` 执行。它验证训练链路，不代表已经得到收敛策略。完整动作中先前标记的穿地/速度峰值仍需依据质量报告筛选或修正；PD 和网络超参数可在后续实验中调整。

2026-10-29 本机验证：CPU 回归 11 项通过；RTX 4060 Laptop 上 GPU 测试 4 项通过，包含策略权重更新、保存及恢复。两种机器人生成的 checkpoint 均完成了 `walk1_subject1` 短片段的评估调用，得到有限数值结果。此处未进行完整数据集的收敛训练。
