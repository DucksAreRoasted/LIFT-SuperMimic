# TADG 代码执行说明

更新：2026-11-05。训练和评估只在用户服务器执行，本地不启动。

## 当前代码

基线已从用户指定 GitHub 仓库导入 `baselines/supermimic/`，其第三方代码展开到根 `third_party/`。TADG 位于 `src/tadg/`，通过同一 uv workspace 的 `supermimic` 包复用环境、世界模型和原训练循环。

| 文件 | 实际改动 |
|---|---|
| `src/tadg/policy.py` | 原均值策略＋高斯 log-std 头，保留无 tanh 动作接口 |
| `src/tadg/sac.py` | 双 Q／target Q、真实转移 Bellman 更新、Q actor 目标 |
| `src/tadg/experience.py` | 原 replay 到 SAC transition 的转换与历史／目标重建 |
| `src/tadg/imagination.py` | 随机 Actor 的 soft-return BPTT 与 soft-Q 末端值 |
| `src/tadg/gradients.py` | 同快照计算两种梯度，融合后一次 Actor step |
| `src/tadg/trainer.py` | 替换原 policy trainer，复用 collect→WM→policy 循环 |
| `scripts/run_suite.py` | 四组对照、多种子与统一记录，默认本仓库基线和 uv 解释器 |

## 服务器环境

在本项目根目录使用唯一的锁文件：

```bash
uv sync --extra train --locked
```

不再需要旧目录 `/home/ducks/SuperMimic_Reproduce`、手工 `PYTHONPATH` 或单独维护基线虚拟环境。`train` 和 `retarget` 的原依赖版本存在冲突，由根 uv 声明为互斥选项；训练对比统一使用 `train`。

## 运行已有基线与 TADG

```bash
# 两个命令使用相同 uv 环境
uv run --extra train --locked sm-train --help
uv run --extra train --locked tadg-train --help

# 暖启动示例，替换为服务器上对应机器人的实际检查点和数据
uv run --extra train --locked tadg-train \
  --robot r1 \
  --run-name tadg_fixed_mix \
  --warm-start /server/path/supermimic_checkpoint.pt \
  --data.npz-dir /server/path/r1_motions \
  --data.clips walk1_subject1 \
  --env.num-envs 256 \
  --policy-trainer.batch-size 256 \
  --policy-trainer.horizon 8 \
  --tadg.dynamics-weight 0.5 \
  --schedule.total-iterations 2000
```

暖启动读取原 actor 均值与世界模型，Q 与方差头新建；网络尺寸和观测配置仍须与 checkpoint 匹配。不会把原 V critic 当作 Q。多方法对照请使用[服务器实验说明](docs/服务器实验与记录.md)的启动方式，实验器会从检查点配置还原网络结构，并统一记录环境和数据。

## 直接可调参数

- `--tadg.dynamics-weight 0`：Q actor 更新，跳过想象反传；原循环仍训练世界模型。
- `--tadg.dynamics-weight 1`：BPTT actor 更新；Q 仍学习并提供末端值。
- `--tadg.dynamics-weight 0.5`：固定双梯度。
- `--tadg.alpha`：两条路径共用的固定熵温度。
- `--tadg.q-updates`：每次 Actor 更新前的 Q 更新次数。
- `--tadg.log-std-init`：高斯策略初始 log 标准差。
- `--tadg.deterministic-collection`：采集使用均值动作；默认随机采集。

随机组关闭原环境额外动作噪声，世界模型使用实际执行动作。当前支持单进程／单 GPU。

## 检查点与评估

`--resume` 恢复 TADG 的 actor、Q／target Q、世界模型和优化器；replay 重新预热，不代表逐位恢复整条轨迹。`--warm-start` 与 `--resume` 二选一。

保存的 `policy` 仍为原均值网络格式，可使用纳入本仓库的基线评估脚本：

```bash
uv run --extra train --locked python baselines/supermimic/scripts/eval_benchmark.py \
  --ckpt /server/path/checkpoint.pt --npz-dir /server/path/motions
```

完整对照记录、封顶与整段评估区别、CPU／GPU 成本与统计说明见[服务器实验说明](docs/服务器实验与记录.md)。本次整合只做路径、语法和依赖锁定检查，没有执行训练或 GPU 验证。

## 接下来改哪里

先由用户在服务器运行固定混合，再把 `trainer.py` 传给 `dual_actor_step` 的固定权重替换为双侧可信度输出。自适应时域随后接入 `imagination.py`。不再将检查铺成独立任务队列。
