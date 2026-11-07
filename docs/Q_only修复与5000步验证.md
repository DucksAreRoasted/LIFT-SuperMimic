# Q-only 修复与下一轮 5000 步验证

更新：2026-11-06。本地修复已通过真实 GPU 冒烟和全套 29 项测试，详见 [冒烟实测结果](Q_only修复冒烟结果_20261106.md)。尚未部署服务器或启动 5000 步长训练。

## 已修改

1. **终止标签统一为状态语义。** Recorder 的非 terminal 状态行统一写 done=0；真正到达的 terminal 行写实际 terminated。原 Q/WM 消费者读取下一状态标签，普通 time limit 保留 bootstrap，参考任务结束仍由 frame 条件关闭 bootstrap。该共享数据修复也影响其他方法，因此修复后的实验应记录新源码版本，不能把变化全部归因于 Q-only 超参。
2. **Q 独立预热。** `q_warmup_updates` 按 Q optimizer step 计数，达到预算前跳过 Actor optimizer step。继续采集固定 Actor 的数据，WM 和 Q 照常训练。预热进度写入 `policy_update_state`，恢复检查点时不重复预热。
3. **固定标准差。** `fixed_std=true` 固定 sigma=exp(-2.3)≈0.1003，不计算标准差 head。均值仍通过 rsample 获得梯度；固定标准差也不会通过共享 encoder 发生变化。
4. **降低 Actor 学习率。** `actor_lr_scale=0.1` 将原 2e-4 起始学习率缩到 2e-5。Q 学习率保留原 3e-4 及其调度，不跟随 Actor 的额外缩放。
5. **正确计数和异常保护。** 日志新增 critic_warmup 事件、critic_only_updates、均值输出最大幅度；Actor 更新计数不含预热。Q loss 或梯度范数非有限时在 optimizer step 前停止。

原标准差可学习等行为仍可通过配置选择；原协议未指定新字段时默认不启用独立预热、固定方差或额外学习率缩放。共享终止标签修复与 Q 数值检查始终生效。

## 新协议

- `configs/experiments/q_only_repair_smoke.json`：单动作、16 环境、80 向量步；前 4 次 Q 更新不更新 Actor，评估最多 100 步，关闭编译。用于覆盖“进入预热—退出预热—保存—评估”的真实 GPU 路径。
- `configs/experiments/q_only_repair_5000.json`：40 动作、256 环境、5000 向量步、种子通过命令指定；前 400 次 Q 更新冻结 Actor，每 500 步完整动作评估，固定方差、Actor 学习率缩放 0.1。

5000 是包含 replay 填充与 Q 预热阶段的总向量环境步数，不是预热后另加 5000 步。按当前 warmup=40、UTD=0.4、q_updates=1，预计总 Q 更新 1984 次，其中前 400 次不更新 Actor，后续 Actor 更新约 1584 次。最终以日志实际计数为准。

**400 次是可复现的初始预热预算，不是自动校准通过的判据。** 本次没有实现基于 Monte Carlo 回报的 Q 可靠性门控，也没有声称预热后 Q 已经准确。后续要结合完整动作表现、Q/TD 指标和真实 rollout 校准判断是否足够。

alpha 仍为 0.001；本轮先固定方差，不同时调整熵温度、reward scale 或增加策略锚定项，以便追踪这次组合修复的效果。

## 服务器执行

在安装了项目训练环境、已有初始预训练模型和 R1 动作数据的服务器根目录设置 CHECKPOINT、MOTIONS。必须使用原预训练模型，避免把已经崩溃的 Q-only 模型当成本次起点。

先执行短冒烟，选择从未使用的输出目录：

```bash
uv run --extra train --locked python scripts/run_suite.py \
  --checkpoint "$CHECKPOINT" --motions "$MOTIONS" \
  --protocol configs/experiments/q_only_repair_smoke.json \
  --methods q_only --seeds 0 \
  --output outputs/q_only_repair_smoke_v1
```

冒烟完成并确认日志中同时出现 critic_warmup 和 actor_update 后，再执行：

```bash
uv run --extra train --locked python scripts/run_suite.py \
  --checkpoint "$CHECKPOINT" --motions "$MOTIONS" \
  --protocol configs/experiments/q_only_repair_5000.json \
  --methods q_only --seeds 0 \
  --output outputs/q_only_repair_5000_v1
```

需要后台运行时，由现有 tmux 会话承载。这里只提供运行命令，本次没有远程执行。短冒烟不是任务表现验证，5000 步仍是单种子初步实验。

## 判断效果

- step 0 与预热期间的均值 Actor 完整评估，应基本保留预训练能力；GPU 数值差异可能导致轨迹不逐位一致。
- 检查 policy_std_mean 始终约 0.1003，action_mean_abs_max 是否增长异常。
- 预热结束后，优先看完整成功片段数、参考完成比例、全窗口归一化回报，不只看存活步归一化回报。
- Q loss 小不代表 Q 校准准确；最终若仍退化，应继续拆分 Actor 步长、Q 校准和采样分布，而不能直接认定 Q 引导不可行。
- 旧 recent_replay 使用旧标签语义；不应直接导入新训练。新运行从空 replay 采集，旧数据不覆盖。

## 本地验证边界

新增 CPU 回归使用真实 recorder、ReplayBuffer、sampler 和 DualPolicyTrainer 方法，只替代模拟器构造及固定 batch 输入；覆盖三类终止边界、固定方差的路径梯度、预热冻结与释放、真实 save/load 进度恢复、日志计数及 Q 非有限目标拒绝。

已从本地只读缓存组装 Python 3.12／Torch 2.14.1／MuJoCo 3.16／锁定 mjlab 提交的隔离环境，在 RTX 4060 上完成正常训练和预热中途恢复两条 GPU 冒烟。全套 29 项测试在该环境通过，修复前 Python 3.10 的 hashlib.file_digest 环境限制已排除。冒烟使用同一预训练检查点和本地官方版行走数据，不等同于服务器完整 40 动作长期评估。
