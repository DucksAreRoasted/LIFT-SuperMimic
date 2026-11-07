# 利用预训练 V 培养 Q，并保留 V 引导的随机 BPTT 对照

更新：2026-11-06。用于验证已有 SuperMimic 检查点上的更新方向；不是从随机 Actor 重新训练。

## 实际实现

1. `TADGTrainer` 显式加载原检查点的 `critic` 权重，冻结它；同时保存原均值 Actor 的冻结副本。缺少预训练 V 的检查点会报错，不会悄悄使用随机 V。
2. Q-only 预热采用真实仿真转移和固定 V 标签：

   `y = reward + gamma * bootstrap * V0(next_state, next_goal, context)`。

   标签不查询随机 target Q，也不对 V/Actor 反传。目标使用原始 goal，由 V 自己缩放，避免重复缩放。Q 的权重仍是新建的，但长期监督信息来自成熟 V。
3. 到达 `q_warmup_updates` 后，先执行真实仿真的局部动作校准，未通过时继续 V 监督 Q，保持 Actor 不变。不会因次数足够就自动放行。
4. 校准在固定参考状态附近比较多个候选动作。第一步施加候选扰动，后续按冻结的原 Actor 均值执行真实 rollout；标签为真实 n-step 回报加冻结 V 的尾值，而非学生 Q 的 bootstrap。
5. 校准同时要求值误差与动作排序通过，且有足够可区分的动作对。没有可比较动作对时不算通过。首次通过后，小学习率开放 Q-only Actor，Q 随后转为当前策略的普通 Bellman 更新。切换前仅一次将 target Q 同步为已预热的在线 Q，避免重新使用仍明显滞后的随机起点 target；同步标志同样保存在检查点。
6. 独立 V 末端 BPTT 对照使用原 V、固定标准差、alpha=0。Actor 仍在模型中重参数化展开；末端不查询 Q，直接 Q 梯度也不参与。保留原 `-J/H` 和逐元素裁剪处理。原 V 在本轮冻结，不声称实现了原完整 GAE 在线更新路径；它是验证“随机 BPTT＋已有 V”是否稳定的受控起点。
7. 冻结 V/Actor 副本及 Q 校准通过标志写入检查点。恢复时必须有老师权重，不能从缺少老师的旧 TADG 检查点直接恢复成该模式。

## 新增配置

| 文件 | 用途 |
|---|---|
| `configs/experiments/q_only_v_teacher_smoke.json` | 80 步单动作，V 标签、校准闸门与保存恢复检查 |
| `configs/experiments/q_only_v_teacher_5000.json` | 40 动作、256 环境、5000 向量步，至少 400 次 Q 更新后校准 |
| `configs/experiments/bptt_v_teacher_smoke.json` | 单动作短测固定 V 末端的随机 BPTT |

Q-only 新协议设置 `q_teacher_v=true`、`require_q_calibration=true`、`fixed_std=true`、`actor_lr_scale=0.1`，alpha=0。旧修复协议仍保留，避免覆盖已有实验定义。

V 末端协议在 `bptt_soft` 配置块中显式设置 `terminal_value_mode="v"`、`q_teacher_v=true`、固定方差和小 Actor 步长，alpha=0。该版本的 soft 最大熵项暂时关闭；不能与首轮 Q 末端最大熵版本混称为同一个实验条件。Q 在旁路学习 V 标签，Actor 不接受它的梯度。

默认 5000 步校准检查 4 个状态、每状态 5 个候选动作、真实 rollout 50 步。初始门槛为：显著回报差超过 0.01 的动作对至少 6 个；排序准确率至少 0.65；RMSE 除以平均绝对目标值不超过 0.25。它们是明确可调整的工程初始门槛，不是统计显著性标准，也不保证整个状态空间的 Q 导数可靠。

## 指标和完成状态

- `q_target_source_v=1`：该次 Q 标签来自冻结 V；0 表示普通 soft Bellman 标签。
- `calibration_pending=1`：校准尚未通过，禁止 Q-only Actor 更新。
- `calibration/step_*/calibration.json`：每次候选动作预测、真实 n-step＋V 标签、排序与误差。
- `calibration_env_transitions`、`calibration_s`：校准交互与耗时独立计数，不计入训练转移。
- `terminal_value_source_v=1`：BPTT 使用已有 V 末端。
- `reward_objective`、`entropy_objective`、`terminal_objective`：目标分项之和对应 `imag_objective`。
- `q_calibration_passed`：实际校准结果；不要求校准的方法也不会被写成已校准。

校准报告只在周期评估点检查，达到预热预算到下一评估之间 Actor 继续冻结。若 5000 步结束仍未通过，训练 worker 以 `calibration_not_passed` 状态和非零退出码结束；不能把全程冻结的结果当成 Q-only Actor 改进成功。

## 验证入口

在安装训练环境的项目根目录：

```bash
uv run --extra train --locked python scripts/run_value_guidance_smoke.py \
  --checkpoint "$CHECKPOINT" --motions "$MOTIONS" \
  --output outputs/value_guidance_smoke_v1
```

该入口在真实 GPU 仿真中检查：老师权重与原检查点逐项一致，老师不变，Q 更新，校准闸门冻结 Actor，V 末端不查询 Q，以及老师和闸门的检查点恢复。输出为集成验收状态，不把 Q 校准未过误称为 Q 已成熟。

正式 Q-only 继续训练使用全新输出目录：

```bash
uv run --extra train --locked python scripts/run_suite.py \
  --checkpoint "$CHECKPOINT" --motions "$MOTIONS" \
  --protocol configs/experiments/q_only_v_teacher_5000.json \
  --methods q_only --seeds 0 --output outputs/q_only_v_teacher_5000_v1
```

训练从原 SuperMimic 预训练模型开始，不从首轮退化模型接着训练。5000 步包含数据填充、Q 预热和可能的等待校准阶段。

## 本地验证记录

- CPU 全套测试 37 项通过，覆盖 V 标签数值与梯度边界、原始 goal 缩放、校准排序和样本不足拒绝、闸门及恢复，以及不查询 Q 的 V 末端 BPTT 路径导数。
- RTX 4060 上两条 GPU 短路径通过：Q-only 完成 24 次 V 标签 Q 更新，老师和 Actor 均未改变；Q 尚未校准，因此未放行 Actor。
- 本地第 80 步、24 次 Q 更新后，nRMSE≈0.914，局部排序准确率 0，说明 24 次更新不足，不能当作成熟 Q。闸门正确阻止了过早接管。
- V 末端 BPTT 完成 24 次 Actor 更新，冻结 V 和老师 Actor 不变，目标分项一致，未查询冷启动 Q。短测不证明长期收敛。
- 本地记录保存在 `outputs/value_guidance_validation/local_01/`；后续服务器结果以相应输出目录及实际 Git 提交为准。

## 边界

V 自身也有误差；小幅随机采集与原 V 所评价的确定性策略并不完全相同。校准使用 n-step＋V，尚不是完整 Monte Carlo 真值验证。当前实现没有自动调低通过阈值，没有复制 V 权重到不同结构的 Q，也没有把校准通过解释成融合方向已经被证明。
