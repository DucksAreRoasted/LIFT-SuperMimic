# Q-only 如何接入 SuperMimic：配置、数据、网络与更新过程

更新：2026-11-06。本文解释 2026-11-05 服务器首轮实验使用的实现路径。后续已新增终止标签修复、Q 独立预热和固定方差选项，最新行为及验证配置见 [Q-only 修复与5000步验证](Q_only修复与5000步验证.md)。下文中的首轮行为和问题保留为历史说明。

Q-only 的含义是：**Actor 只通过真实数据训练出的 Q 函数获取策略梯度，不通过世界模型做 BPTT。** 环境、数据采集、世界模型训练和保存调度继续复用原 SuperMimic。因此它是共享循环下的 Q 梯度消融，仍承担世界模型训练成本。

另一条路径见 [soft BPTT 接入与实现详解](soft_BPTT接入与实现详解.md)。

## 1. 原来的 SuperMimic 在什么位置被扩展

原循环定义在 [training/loop.py](../baselines/supermimic/src/supermimic/training/loop.py) 的 `Trainer`，每次向量环境交互后执行：

```text
Trainer.run()
  ├─ self.act(obs)                         原确定性 Actor
  ├─ env.step(action)                      仿真、奖励、终止和 replay 记录
  ├─ self.wm_trainer.update(buffer)         拟合世界模型
  └─ self.policy_trainer.update(buffer)     原 BPTT Actor 更新及 V 拟合
```

接入 Q-only 时，没有重写 `Trainer.run()`，而是新增 [trainer.py](../src/tadg/trainer.py) 中的子类 `TADGTrainer(Trainer)`：

| 扩展位置 | 原行为 | Q-only 路径 |
|---|---|---|
| `policy` | 原确定性 `Policy` | `StochasticPolicy` 包装同一个原 Policy |
| `act()` | 返回均值动作 | 默认采样高斯动作 |
| `policy_trainer` | 原 `PolicyTrainer` | 新 `DualPolicyTrainer` |
| `wm_trainer` | 原世界模型训练器 | 保留 |
| `run()` | 原交互和更新循环 | 继承并直接使用 |
| `save()/load()` | 原 Actor、WM、V 及优化器 | 增加随机策略、Q、target Q 及相应优化器 |

子类初始化的关键顺序为：

```python
super().__init__(cfg)
self.policy = StochasticPolicy(self.policy, cfg.tadg.log_std_init)
self.policy_trainer = DualPolicyTrainer(
    self.wm, self.policy, self.lib, cfg.policy_trainer, cfg.tadg
)
```

父类创建的环境、replay、动作库和 WM 仍被使用。父类也会构造原 V critic，但替换后的 Q-only 更新路径不使用它。新的世界模型参数加载到同一个 WM 对象中，已有 `wm_trainer` 仍引用这个对象。

## 2. 从 `--method q_only` 到真正进入 Q-only 分支

正式对照入口是 [scripts/run_suite.py](../scripts/run_suite.py)，而非另一个独立 Q-only 程序。

```text
run_suite.py --methods q_only
  → 生成冻结 protocol.json、commands.json 和源码快照
  → 启动 train_experiment.py --method q_only
  → make_config(checkpoint, protocol, "q_only", ...)
  → RecordedTADGTrainer(cfg)
  → TADGTrainer.__init__()
  → 原 Trainer.run()
  → DualPolicyTrainer.update(buffer)
  → dynamics_weight == 0 分支
```

[experiment.py](../src/tadg/experiment.py) 的 `make_config()` 先读取检查点中的网络与机器人配置，再覆盖当前协议的环境数、批量、时域、步数和评估间隔。方法映射为：

```python
cfg.tadg.dynamics_weight = {
    "q_only": 0.0,
    "bptt_soft": 1.0,
    "dual_fixed": float(protocol["dynamics_weight"]),
}[method]
```

因此只要 `method=q_only`，权重就被设为 0；协议中的公共 `dynamics_weight=0.5` 不会将这一组变成混合更新。

`RecordedTADGTrainer` 的继承顺序为 `RecordedMixin, TADGTrainer`。Mixin 增加统一评估、事件记录和 replay 导出，不改变 Q-only 的损失定义。直接使用 `tadg-train` CLI 可以配置同一个训练器，但不自动包含这套实验记录封装。

## 3. 预训练加载了什么，新建了什么

首轮对照使用 `warm_start`：

```python
self.policy.mean_policy.load_state_dict(ck["policy"])
self.wm.load_state_dict(ck["wm"])
```

| 对象 | 初始化来源 |
|---|---|
| 均值 Actor 的 goal encoder、decoder | 原 SuperMimic 检查点 |
| 世界模型，包括终止预测部分 | 原检查点 |
| 标准差 head | 新建；权重 0，bias 为 -2.3 |
| Q1、Q2 | 新建的随机网络 |
| target Q1、Q2 | 深拷贝新建 Q |
| Actor、Q、WM 优化器 | 新建 Adam，未恢复原动量 |
| 迭代计数、采样器和 replay | 新运行，从头建立 |

`warm_start` 不等于 `resume`。真正恢复 TADG 检查点时，`load()` 会恢复 `tadg_policy`、Q、target Q、优化器、采样器和迭代计数；replay 本身仍需重新填充。两种入口不能同时指定。

首轮 `warmup_iterations=40` 只填充 replay，不做 Q-only 预热训练。根据实际日志，第一次 Actor 更新出现在第 42 步，此前 Q 仅在同一次 `DualPolicyTrainer.update()` 中更新了一次。这是当前实现的重要初始化风险。

## 4. 如何把原确定性 Actor 变成随机 Actor

[policy.py](../src/tadg/policy.py) 的 `StochasticPolicy` 保留原网络作为 `mean_policy`，并新增一个线性标准差 head。

```text
goal ──原 motion encoder──→ z（32 维）
                             │
scaled history ───────────────┤
                             ↓
                     x = concat(z, history)
                       ├─ 原 decoder MLP → μ
                       └─ 新 linear head → log σ
```

实际计算为：

\[
z=\mathrm{enc}_{\theta}(g),\quad
\mu=\mathrm{decoder}_{\theta}([z,o\odot scale_o])
\]

\[
\log\sigma=\mathrm{clamp}(Wx+b,-5,1),\quad
a=\mu+\sigma\odot\epsilon,\quad\epsilon\sim\mathcal N(0,I)
\]

代码使用 `Normal.rsample()` 保留动作对 Actor 参数的导数。`log_prob` 沿动作维度求和，每个样本得到一个标量。

初始化时 `W=0,b=-2.3`，所以所有动作维度的标准差约为 0.1003。允许的标准差范围约为 `[0.00674,2.718]`。均值不经过 tanh，仍沿用原始的归一化关节位置目标接口。

这里没有复制第二套均值网络；`mean_policy` 就是原 Actor。Encoder、decoder 和标准差 head 都属于 `policy.parameters()`，都会被 Actor 优化器更新。标准差 head 的输入包含共享 latent，因此其梯度也可能更新 goal encoder。

`forward()` 返回确定性均值；`sample()` 返回随机动作。这一区别用于训练采集与统一评估。

## 5. 动作如何进入环境，以及如何记录

`TADGTrainer.act()` 默认执行：

```python
return self.policy.sample(obs["policy"], obs["goal"]).action
```

只有显式设置 `deterministic_collection=True` 才改为均值采集。

同时 `TADGTrainer` 将 `cfg.env.action_noise_std=0.0`，因此采集噪声来自策略分布，不再叠加原 SuperMimic 的额外执行噪声。

这个变化还影响记录语义：原基线的 clean action 不包含额外执行噪声；当前随机组的高斯样本本身就是 raw action，也进入 last-action 观测和动作变化惩罚。因此，初始标准差接近原执行噪声并不表示两种采集方式完全等价。

环境负责缩放动作并加默认关节位置偏置。本文中的 `action_abs_max` 等统计是归一化动作量，不是关节角、力矩或能耗。

## 6. 真实 replay 怎么适配成 Q 的 batch

新增适配器位于 [experience.py](../src/tadg/experience.py)。它不建立第二个 replay，而是读取原 `ReplayBuffer`。

`sample_real_batch()` 的处理顺序是：

1. 使用 `buffer.window_mask(1)` 选择同一 episode 内相邻的状态行，排除从 terminal 行出发及跨写指针的窗口。
2. 增加 age 条件，避免构造历史时读取已被环形 buffer 覆盖的行。
3. 采样 `(env,time)`，令 `next_time=(time+1)%T`。
4. `_start_at()` 重建当前及下一状态的历史观测。episode 开始处不足的历史，用最早的同 episode 状态回填；不存在的前一动作填零。
5. 修正下一状态参考帧为 `min(start.frame+1,last_frame)`，避免 terminal 行的 frame bookkeeping 影响目标构造。
6. `ModelInputs.at()` 构造 Actor 观测、goal 与 critic features。
7. 返回 `SACBatch` 以及 `StartBatch`。Q-only 不使用后者展开想象；soft BPTT 使用它。

整个采样函数在 `@torch.no_grad()` 下运行，真实 replay 的状态、历史和目标作为固定输入，不对采样过程反向传播。

### 6.1 Actor 和 Q 的输入不同

以下维度适用于首轮 R1、26 关节、history=10、future_frames=10、anchor_obs=False。换机器人或开启 anchor_obs 后维度会变化。

| 数据 | 维度 | 内容 |
|---|---:|---|
| 单帧 Actor 观测 | 84 | 关节位置 26、速度 26、角速度 3、重力 3、前一动作 26 |
| 历史观测 | 840 | 10 帧，按 observation term 展平 |
| 单个参考帧 goal | 62 | 26+26+3+3+3+1 |
| goal window | 620 | 10 个未来参考帧 |
| canonical state | 65 | 经过 heading 处理及固定缩放的状态特征 |
| context xi | 30 | 摩擦、编码器偏差与质心偏移等上下文，经归一化 |
| root error | 4 | 根部跟踪误差特征 |
| critic features x | 1559 | 65+620+30+4+840 |
| Q 输入 | 1585 | x 再拼 26 维动作 |

Q 使用比 Actor 更丰富的状态信息，属于非对称 critic。它没有直接拿 Actor 的 latent 当唯一状态表示。

### 6.2 SACBatch 的字段

| 字段 | R1 batch=256 时的形状 | 来源 |
|---|---|---|
| `features` | `[256,1559]` | 当前状态、goal、context、root error、历史 |
| `obs` | `[256,840]` | 当前 Actor 历史观测 |
| `goal` | `[256,620]` | 当前参考窗口 |
| `action` | `[256,26]` | `buffer.a_exec[env,time]` |
| `reward` | `[256]` | 实际仿真 `buffer.r[env,time]` |
| `bootstrap` | `[256]` | 终止和参考结束条件 |
| `next_features/obs/goal` | 对应下一状态维度 | 同 episode 下一行，重建历史 |

Q 学习的奖励是真实仿真奖励，不是 WM 预测奖励；Q-only 的策略更新也不使用 WM 预测的下一状态。

## 7. 双 Q、Bellman 目标和 target 更新

[sac.py](../src/tadg/sac.py) 中的 `TwinQ` 包含两套独立 MLP：

```text
concat(features, action) → Linear(1585,512) → SiLU
                        → Linear(512,256)  → SiLU
                        → Linear(256,1)
```

Q1 与 Q2 不共享 MLP 参数；`minimum()` 逐样本取二者最小值。

一次 `SACBranch.update()` 先在 `no_grad` 下采样下一动作：

\[
a'\sim\pi_\theta(\cdot|o',g')
\]

\[
y=r+\gamma b\left[\min_{i=1,2}\bar Q_i(x',a')
-\alpha\log\pi_\theta(a'|o',g')\right]
\]

其中 `bar Q` 是 target Q，`b` 是 bootstrap 开关。意图是：真摔倒或参考任务结束时不 bootstrap，普通 time limit 仍保留 bootstrap；当前终止读取的错位见第 12 节。

然后最小化：

\[
L_Q=\mathbb E[(Q_1(x,a)-y)^2+(Q_2(x,a)-y)^2]
\]

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
self.opt.step()
```

最后软更新 target：

\[
\bar\phi\leftarrow(1-\tau)\bar\phi+\tau\phi,\qquad\tau=0.005
\]

Q 的优化不会通过 `y` 更新 Actor 或 target Q。当前使用 MSE，没有自动温度调节，`alpha=0.001` 固定。

## 8. Q-only 的 Actor 更新到底如何执行

`DualPolicyTrainer.update()` 先更新 Q，再进入 `dynamics_weight==0` 分支。以下是当前关键代码：

```python
with frozen(self.wm, self.sac.q):
    if self.dual.dynamics_weight == 0:
        sample = self.policy.sample(batch.obs, batch.goal)
        q_objective = self.sac.actor_objective(batch.features, sample)
        metrics.update(dual_actor_step(
            self.policy, self.opt, q_objective, q_objective * 0,
            0.0, self.cfg.grad_clip_norm
        ))
```

`actor_objective()` 返回上升目标：

\[
J_Q=\mathbb E[\min_iQ_i(x,a_\theta)-\alpha\log\pi_\theta(a_\theta|o,g)]
\]

`frozen()` 将 Q 和 WM 参数的 `requires_grad` 暂时设为 false，但没有进入 `torch.no_grad()`。因此 Q 的动作输入导数仍然保留：

```text
Actor θ → aθ → Q(x,aθ) → JQ
       └──────── log πθ ──┘
```

真实状态 features 是固定值；这条 Actor 梯度路径没有调用 WM.step。WM 在这里被冻结只是为了公共更新接口的一致性。

### 8.1 为什么还调用 dual_actor_step

[gradients.py](../src/tadg/gradients.py) 提供统一的两梯度更新器：

\[
g_Q=\nabla_\theta J_Q,\quad g_B=\nabla_\theta J_B,
\quad g=(1-w)g_Q+wg_B
\]

Q-only 传入 `w=0`，并用 `q_objective*0` 作为另一目标，因此最终 `g=g_Q`。

函数使用 `torch.autograd.grad()` 显式取梯度，检查逐元素有限性，然后设置：

```python
p.grad = -((1 - weight) * q + weight * d).detach()
```

负号用于把“最大化 J”转成 Adam 的下降更新。之后做一次全局范数裁剪、一次 `optimizer.step()`。这里没有把两个分支各优化一次。

当前 Actor 的 Adam 初始 lr=2e-4、betas=(0.9,0.95)、全局范数裁剪 1.0。不同于原 BPTT 的 `-J/H` 及后续逐元素裁剪，新路径未除以 H，也未做逐元素裁剪；不能据此直接推断 Adam 参数步长放大了 H 倍。

## 9. 一次完整训练更新的伪代码

下列伪代码按当前调用顺序组织，省略了日志、学习率调度和保存，不是替代训练入口：

```text
for each vector environment step:
    a = stochastic_actor.sample(obs, goal)
    env.step(a)                              # recorder 写入真实 replay

    if warmup 已结束且 UTD 累加器允许更新:
        world_model.update(replay)           # 本轮每次做 2 个 minibatch

        repeat q_updates times:              # 本轮为 1
            batch, start = sample_real_batch(replay)
            update_online_Q(batch)
            soft_update_target_Q()

        freeze WM and online Q parameters:
            sample new Actor action at batch.obs
            JQ = mean(min Q - alpha * log_prob)
            update Actor using gradient of JQ
```

`q_updates>1` 时会重复采样和更新 Q；最后一次采样的 batch 用于本次 Actor 更新。当前每次 Actor 更新只发生一次，增大 `q_updates` 不会等比例增加 Actor 更新次数。

## 10. 本轮参数与运行方式

| 参数 | 首轮服务器实际值 | 作用 |
|---|---:|---|
| 机器人／动作数 | R1／40 段 | 训练与评估使用同批动作 |
| 环境数 | 256 | 每次向量步产生 256 条转移 |
| 训练步数 | 5000 | 总共 128 万条训练转移 |
| replay capacity | 每环境 300 行 | 含额外 terminal 行 |
| warmup | 40 | 填充数据，不是 critic 预训练 |
| updates_per_env_step | 0.4 | 每个向量环境步累加 0.4 次更新 |
| Actor／Q batch | 256 | 两者用同一次真实 batch 的起点 |
| WM batch／批次数 | 256／2 | 每个更新事件对应两个 WM optimizer step |
| q_updates | 1 | 每个 Actor 更新前的 Q 更新次数 |
| gamma／alpha／tau | 0.99／0.001／0.005 | 折扣、固定熵温度、target 软更新 |
| horizon | 8 | 公共配置保留，Q-only 不执行想象 |

首轮实际完成 1984 次 Actor/Q 更新、3968 次 WM optimizer step；UTD 是按向量步定义，不能直接读成“每条单环境转移更新 0.4 次”。

当前仓库 `configs/experiments/r1_lafan1_40clips.json` 默认是 2000 步且评估封顶 1000 步，与首轮 5000 步、完整评估的冻结协议不同。比较实验必须核对输出中的实际 `protocol.json`，不能仅看文件名。

以下命令仅生成计划，`CHECKPOINT` 和 `MOTIONS` 由执行者设置为实际服务器路径：

```bash
uv run --extra train --locked python scripts/run_suite.py \
  --checkpoint "$CHECKPOINT" \
  --motions "$MOTIONS" \
  --protocol configs/experiments/r1_lafan1_40clips.json \
  --methods q_only --seeds 0 \
  --output outputs/q_only_plan_v1 \
  --dry-run
```

真正训练时去掉 `--dry-run` 并选择新的输出目录。不要复用 dry-run 已创建的目录。本文没有执行该命令或启动训练。

## 11. 怎么检查保存结果和指标

`TADGTrainer.save()` 保存：均值 `policy`、完整 `tadg_policy`、WM、Q、target Q、三类优化器、采样器状态和配置。单独保留 `policy` 键，是为了兼容原有确定性评估入口；它不是完整随机策略的恢复数据。

统一实验额外保存 `events.jsonl`、`status.json`、`resolved_config.json`、评估 JSON、抽样轨迹，以及可选的 `recent_replay.pt`。后者是末尾环形快照，不是完整训练流。

建议依次看：

| 指标 | 能回答的问题 | 限制 |
|---|---|---|
| `q_loss/q_td_abs/q_td_max` | Q 对当前 Bellman 标签拟合得怎样 | TD 小不等于真实长期价值准确 |
| `q_disagreement_abs` | 两个 Q 是否分歧 | 两者可以同时错误 |
| `bootstrap_fraction` | 目标中有多少样本继续 bootstrap | 受当前终止错位影响 |
| `policy_std_mean` | 探索幅度是否扩大 | 跨关节平均可能掩盖个别极端维度 |
| `action_abs_max` | 采样 Actor 在真实 batch 状态上的输出是否异常 | 是本次重新采样动作，不必然等于已执行动作 |
| `q_actor_grad_norm` | Q 给 Actor 的原始梯度尺度 | 不能直接读成最终参数步长 |
| 完整动作成功数、完成比例、窗口回报 | 实际跟踪是否保持或提高 | 单种子仍不能证明稳定优势 |

Q-only 中 `dynamics_grad_norm=0`、`gradient_cosine=0` 是统一代码的占位结果；没有有效的动力学梯度可供比较，不应解释成“方向正交”。

## 12. 当前实现中已经确认或需要验证的问题

**已确认：终止标签语义不一致。** Recorder 将 transition 的 done_next 写入起点行，而 Q 读取下一状态行的 done；真正终止前的一条非终止转移也可能被关闭 bootstrap。WM 的状态终止目标同样受影响。这是需要同步修复写入端与消费端的错误，不能只改某一个下标而不检查全部消费者。复现和结果见 [首轮退化分析](首轮退化分析_20261105.md)。

**风险：新 Q 立即引导成熟 Actor。** 没有独立 critic 预热、延迟 Actor 更新或保留行为约束。Q-only 在第 67 步已出现超过 100 的归一化采样动作，远早于后期 WM 梯度范数异常；这说明不能只把 WM 后期失稳当成最初诱因。

**风险：策略分布与动作接口缺少偏离控制。** 均值无界，标准差可快速增大；直接套用 tanh 会改变原预训练动作语义，应设计与既有缩放接口兼容的约束。

**实验解释边界。** 本组仍训练 WM；冷启动 Q 与原成熟 V 的差别、随机策略变化、Actor 裁剪协议变化都应单独核验。现有结果不构成标准 SAC 在该任务上的完整基准结论。

## 13. 建议按什么顺序读源码与验证

1. [experiment.py](../src/tadg/experiment.py)：方法如何映射到配置。
2. [trainer.py](../src/tadg/trainer.py)：父类循环如何复用、模块如何替换、warm_start 如何加载。
3. [policy.py](../src/tadg/policy.py)：均值保留、标准差 head 与 rsample。
4. [experience.py](../src/tadg/experience.py)：真实 replay 如何构成当前／下一状态 batch。
5. [sac.py](../src/tadg/sac.py)：Q 目标、损失、target 更新与 Actor 目标。
6. [gradients.py](../src/tadg/gradients.py)：上升梯度如何变成一次 Adam 更新。

现有 [test_dual_gradient.py](../tests/test_dual_gradient.py) 覆盖均值策略保留、Bellman 数值、SAC 更新时的梯度边界，以及一次混合更新。数值测试通过不能代替 recorder/sampler 边界测试和稳定的继续训练对照。
