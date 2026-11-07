# soft BPTT 如何接入 SuperMimic：可微想象、soft return 与参数更新

更新：2026-11-06。本文解释 2026-11-05 首轮服务器实验的实际实现。后续已修复共享 replay 的终止标签；新增的 Q 预热和固定方差修复协议针对 Q-only，详见 [修复记录](Q_only修复与5000步验证.md)。下文的终止标签问题是首轮历史问题，soft BPTT 的目标和展开路径本次未改。

soft BPTT 的含义是：**从 replay 的真实状态出发，在冻结的可微世界模型中展开随机策略，计算含熵与末端 Q 的短期回报，再沿时间展开图反向传播更新 Actor。**

“soft”在这里指最大熵目标；BPTT 指 Backpropagation Through Time。终止概率也采用可微的存活权重，但它不是该方法名称中 soft 的唯一含义。

本组仍训练 Q，而且末端价值依赖 Q。它不是去掉 Q 后的纯奖励 BPTT。Q 的完整实现另见 [Q-only 接入与实现详解](Q_only接入与实现详解.md)。

## 1. 在原 SuperMimic 基础上改变了哪些部分

原 SuperMimic 的 Actor 更新已经具备“真实起点 → WM 展开 → 奖励与末端 V → 反向传播”链条。新增 soft BPTT 保留环境和 WM，实现新的随机 rollout 与 soft-return 目标。

| 部分 | 原 SuperMimic | 当前 soft BPTT |
|---|---|---|
| 环境与动作库 | 原 tracking 环境 | 复用 |
| replay 与 WM 拟合 | 原数据窗口与 WMTrainer | 复用 |
| Actor | 确定性均值策略 | 原均值策略加高斯标准差 head |
| 想象动作 | 每步使用均值 | 每步重参数化随机采样 |
| 想象奖励 | 预测状态上的解析跟踪奖励 | 沿用同类解析奖励 |
| 末端价值 | 已有 V critic | 新建双 Q 的 soft value |
| critic 学习 | 想象轨迹的 GAE 目标拟合 V | 真实仿真转移的 soft Bellman 目标拟合 Q |
| Actor 目标 | 奖励与末端 V | 奖励、策略熵与末端 soft Q |
| Actor 梯度处理 | `-J/H`、全局范数与逐元素裁剪 | 显式求梯度、`-J`、仅全局范数裁剪 |

因此这里同时改变了策略分布、critic 类型和初始化、目标与优化协议。它不是只给原 Actor 加执行噪声的对照。

## 2. 方法入口和类的替换位置

```text
run_suite.py --methods bptt_soft
  → train_experiment.py --method bptt_soft
  → experiment.make_config()
      cfg = TADGConfig()
      cfg.tadg.dynamics_weight = 1.0
  → RecordedTADGTrainer
  → TADGTrainer
      包装随机 Actor，替换 policy_trainer
  → 原 Trainer.run()
  → DualPolicyTrainer.update()
      真实 batch 更新 Q
      imagine_soft(...)
      dual_actor_step(..., dynamics_weight=1)
```

关键接入点在 [trainer.py](../src/tadg/trainer.py)：`TADGTrainer` 继承原 `Trainer`，不重写 `run()`，通过替换 `self.policy`、`self.policy_trainer` 以及覆盖 `act/save/load` 接入新方法。

世界模型训练器继续使用父类创建的对象。新的 `DualPolicyTrainer` 使用已有 WM 和动作库，并建立自己的 TwinQ、target Q、Q optimizer 和 Actor optimizer。

`RecordedMixin` 位于继承链前端，用于记录和评估；它不会把 soft BPTT 损失改写成其他目标。

## 3. 初始化和采集：与 Q-only 共用随机 Actor

`warm_start` 加载已有均值 Actor 和 WM；标准差 head、Q 和 target Q 新建，不加载原 V 作为末端值。新运行不恢复原优化器动量。

高斯策略定义于 [policy.py](../src/tadg/policy.py)：

\[
z=\mathrm{enc}_{\theta}(g),\quad
x=[z,o\odot scale_o]
\]

\[
\mu=\mathrm{decoder}_{\theta}(x),\quad
\sigma=\exp(\mathrm{clamp}(Wx+b,-5,1))
\]

\[
a=\mu+\sigma\odot\epsilon,\qquad\epsilon\sim\mathcal N(0,I)
\]

均值沿用原 Policy；新增标准差 head 初始权重为零，bias=-2.3，初始标准差约 0.1003。动作不经过 tanh。`Normal.rsample()` 保留动作对策略参数的路径导数。

实际仿真采集使用随机动作，并关闭原环境额外执行噪声。统一评估调用 `policy.forward()` 使用均值动作，关闭噪声与域随机化。

均值 encoder、decoder 与标准差 head 都在同一个 Actor 优化器中。因此 soft BPTT 会改变探索分布，不只是改变均值。停止评估采样不能恢复已退化的均值策略。

## 4. 一次外部更新先做哪些事

原 `Trainer.run()` 完成一次向量环境采集，并达到更新条件后，先调用 `wm_trainer.update(buffer)`，拟合 WM，再调用新的 `policy_trainer.update(buffer)`。

`DualPolicyTrainer.update()` 内部顺序为：

```python
for _ in range(self.dual.q_updates):
    batch, start = sample_real_batch(
        buffer, self.inputs, self.cfg.batch_size, self.cfg.history
    )
    metrics = self.sac.update(self.policy, batch)

with frozen(self.wm, self.sac.q):
    rollout = imagine_soft(
        self.wm, self.policy, self.sac, self.inputs,
        start, self.cfg.horizon, self.step_fn
    )
    metrics.update(dual_actor_step(
        self.policy, self.opt,
        rollout.q_objective, rollout.objective,
        self.dual.dynamics_weight, self.cfg.grad_clip_norm
    ))
```

本组 `dynamics_weight=1`，所以不进入 Q-only 的提前返回分支。

`q_updates>1` 时，Q 更新多次，随后仅对最后一次采样返回的 `start` 做本次想象与一次 Actor 更新。

### 4.1 Q 仍然用真实数据训练

Q 的训练目标与 Q-only 相同：

\[
y=r+\gamma b\left[\min_i\bar Q_i(x',a')
-\alpha\log\pi_\theta(a'|o',g')\right]
\]

\[
L_Q=\mathbb E[(Q_1-y)^2+(Q_2-y)^2]
\]

奖励和下一状态来自仿真 replay，不来自想象轨迹。Q optimizer 更新在线 Q，target Q 按 tau=0.005 软更新。当前没有将想象样本额外写入 Q 训练数据。

首次 Actor 更新前只有一次 Q 更新，40 步 warmup 不包含 critic-only 预热。soft BPTT 也会从第一次更新开始依赖这个新 Q。

## 5. 想象起点如何从 replay 构造

[experience.py](../src/tadg/experience.py) 的 `sample_real_batch()` 同时返回：

- `SACBatch`：用于上述真实数据 Q 更新。
- `StartBatch`：用于想象展开的固定起点。

`StartBatch` 包含：

| 字段 | 首轮 R1 形状 | 用途 |
|---|---|---|
| `state` | `[N,65]` | 当前物理状态，作为 s0 |
| `xi` | `[N,30]` | 本 episode 的仿真上下文 |
| `frame` | `[N]` | 当前参考目标帧 |
| `prev_action` | `[N,26]` | 第一步动作变化惩罚所需的上一动作 |
| `hist_states` | `[10,N,65]` | 重建历史观测 |
| `hist_actions` | `[10,N,26]` | 与历史状态对应的上一动作 |

N 是 Actor batch，本轮为 256。起点选择与历史构造在 `no_grad()` 下完成，不对 replay 或仿真求导。

episode 起点处的历史采用最早同 episode 行回填；不存在的前一动作置零，避免混入上一个 episode。环形覆盖与跨 episode 窗口由 mask 过滤。

## 6. ModelInputs 如何连接状态、策略和 Q

`ModelInputs.history(start)` 用解析观测函数将历史状态与前一动作转换为观测，并在训练时重新采样观测噪声。

`ModelInputs.at(state,xi,frame,hist)` 每一步返回：

```text
obs       历史观测，按 observation term 展平
goal      对应 frame 的未来参考窗口
features  Q 所需的状态、目标、上下文及观测历史
ref       该步解析奖励和终止判断所需参考量
```

首轮 R1 的维度为：obs `[N,840]`、goal `[N,620]`、features `[N,1559]`、action `[N,26]`。Q 的 features 包含 canonical state 65、scaled goal 620、normalized xi 30、root error 4、scaled history 840。

初始 `ModelInputs.at()` 的输入没有 Actor 梯度；后续的 state 和 hist 来自可微模型展开，会携带 Actor 梯度。该方法内部没有统一 `no_grad()`，因此不会阻断这些后续梯度。

goal 的参考内容与 frame 按动作库推进，不是由 Actor 生成。开启 `anchor_obs` 时，goal 还会拼接依赖预测机器人位姿的相对量；本轮 anchor_obs=False。

## 7. imagine_soft 内部逐步发生了什么

核心函数位于 [imagination.py](../src/tadg/imagination.py) 的 `imagine_soft()`。

### 7.1 建立初始变量

```python
state, frame, previous_action = start.state, start.frame, start.prev_action
hist = inputs.history(start)
alive = torch.ones_like(frame, dtype=state.dtype)
survival = [alive]
last = inputs.lib.last_frame_of(frame)
```

`alive` 是每个 batch 样本的存活权重，初始为 1，不是环境返回的布尔 reset 标记。

### 7.2 每一步采样动作，同时在第一步记录直接 Q 目标

```python
obs, goal, features, ref = inputs.at(state, start.xi, frame, hist)
sample = policy.sample(obs, goal)
if q_objective is None:
    q_objective = sac.actor_objective(features, sample)
```

第一步的 `q_objective` 与 BPTT 路径共享这一次 `sample`，使双梯度模式能在同一个起点、同一初始样本上比较梯度。这里是在 `imagine_soft()` 内共享；前面 Q 更新用的另一份重建观测噪声不保证与此逐位相同。

在 soft BPTT 组，直接 Q 目标用于梯度诊断，最终权重为零；第 10 节说明它与末端 Q 的区别。

### 7.3 调用可微模型一步

```python
next_state, reward, _, logits, obs_clean, _ = step_fn(
    state, sample.action, previous_action, start.xi, ref
)
```

`step_fn` 是原 SuperMimic 的 [ImagStep](../baselines/supermimic/src/supermimic/imagination/compiled.py)。它组合：

1. `wm.step(state, action, xi)` 预测下一状态。
2. 通过 FK 算出预测身体位置、朝向和速度。
3. 用预测状态、当前动作、前一动作和该转移的参考帧计算解析奖励。
4. 预测终止 logits。
5. 用预测下一状态和本步动作重建下一帧干净观测。

奖励包含根部位置／朝向、身体位置／朝向、线速度／角速度跟踪项，以及动作变化和关节限位惩罚。不存在另一个训练出来的 reward 网络。

训练时可以用 `torch.compile` 编译这个复合一步函数。是否编译影响执行方式，不改变本文的目标定义。

### 7.4 累乘存活概率

```python
termination = torch.where(frame >= last, torch.ones_like(logits), logits.sigmoid())
alive = alive * (1 - termination)
survival.append(alive)
```

记第 h 步到达下一状态的终止概率为 d(h+1)，则：

\[
c_0=1,\qquad c_{h+1}=c_h(1-d_{h+1})
\]

参考已到末帧时强制终止；否则使用终止头的 sigmoid 概率。存活概率没有 detach，因此 Actor 梯度还会经过终止预测对状态、动作的导数。

当前转移的奖励乘 `c_h`，后续奖励和末端值乘之后的存活权重。因此终止转移本身的奖励仍计入，后继回报受到抑制。

### 7.5 推进观测历史、状态、帧和前一动作

```python
hist = hist[1:] + [inputs.add_noise(obs_clean)]
state, frame, previous_action = (
    next_state, torch.minimum(frame + 1, last), sample.action
)
```

`next_state`、`obs_clean` 和 `sample.action` 都没有 detach。后续 Actor 既依赖预测状态形成的观测，也依赖历史动作，因此梯度会沿历史窗口继续传播。

循环固定执行 H 次，不会逐个删除低存活概率的样本。目标使用 survival 加权；某些动作或奖励诊断均值则未使用 survival 加权，不能直接等同于有效存活轨迹的均值。

## 8. 第 H 步的末端 soft value

完成 H 步模型展开后，代码再构造一次 Actor 与 Q 输入，并采样末端动作：

```python
obs, goal, features, _ = inputs.at(state, start.xi, frame, hist)
terminal_sample = policy.sample(obs, goal)
terminal_value = (
    sac.q.minimum(features, terminal_sample.action)
    - sac.alpha * terminal_sample.log_prob
)
```

这里的 terminal 指想象窗口末端，不意味着机器人已经终止。

\[
V_H^{soft}=\min_iQ_i(x_H,a_H)
-\alpha\log\pi_\theta(a_H|o_H,g_H)
\]

三个实现细节直接影响梯度：

1. 使用在线 Q，即 `sac.q`，不是 `sac.target_q`；target Q 只用于真实转移的 critic 标签。
2. 第 H 步的动作是一次新的策略采样，不是复用第 H-1 步动作。
3. features、action 和 log_prob 均未 detach，所以末端 Q 对预测状态、历史观测和末端动作的导数都会进入 Actor 梯度。

首轮 H=8、控制周期 0.02 秒，显式模型展开覆盖 0.16 秒；`gamma**H≈0.9227`。末端值还要乘 cH，不能只看 0.9227 就说它必然主导梯度，但它也没有被折扣到可忽略。

## 9. soft return 的公式如何落成代码

[gradients.py](../src/tadg/gradients.py) 的 `soft_return()` 接收：

| 张量 | 形状 | 内容 |
|---|---|---|
| rewards | `[H,N]` | h 步动作后得到的 r(h+1) |
| log_probs | `[H,N]` | h 步随机动作 log probability，已对 26 关节求和 |
| survival | `[H+1,N]` | c0 到 cH |
| terminal_value | `[N]` | 末端 soft Q |

目标为：

\[
J_B=\frac1N\sum_{n=1}^N\left[
\sum_{h=0}^{H-1}\gamma^h c_{h,n}
\left(r_{h+1,n}-\alpha\log\pi_\theta(a_{h,n}|o_{h,n},g_{h,n})\right)
+\gamma^Hc_{H,n}V_{H,n}^{soft}\right]
\]

当前核心代码为：

```python
discount = gamma ** torch.arange(horizon, device=rewards.device, dtype=rewards.dtype)
stage = rewards - alpha * log_probs
return (
    (discount[:, None] * survival[:-1] * stage).sum(0)
    + gamma**horizon * survival[-1] * terminal_value
).mean()
```

`sum(0)` 对时间求和，`mean()` 对 batch 求平均，没有除以 horizon。熵项和 Q-only 使用相同固定 alpha=0.001；高斯的连续分布 log density 可以为正，单个样本的 `-log_prob` 不必为正，这不构成公式符号错误。

## 10. Actor 更新的计算图与梯度边界

进入想象前，调用 `frozen(self.wm,self.sac.q)`。它只冻结参数，不关闭自动求导。

```text
固定真实 s0、history0
       │
       ↓
Actor θ → a0 → frozen WM → s1 → 解析奖励 r1
       │                    ├→ 终止概率 d1 → c1
       │                    └→ 新观测与历史 → Actor θ → a1 → ...
       │
       └──────────────────────────────→ sH、historyH
                                            ├→ online Q 的状态输入
                                            └→ Actor θ → aH → online Q 的动作输入

所有带 θ 的路径和中间状态路径汇入 JB，再对 θ 求梯度
```

| 对象 | Actor 更新时是否改参数 | 是否保留输入导数 |
|---|---|---|
| Actor 均值 encoder、decoder | 是 | 是 |
| Actor std head | 是 | 是，通过 rsample 和 log_prob |
| 世界模型和终止头 | 否 | 是，保留对状态／动作输入的导数 |
| 在线 Q | 否 | 是，保留对 features／action 的导数 |
| target Q | 否 | 不参与本次 Actor 目标 |
| 真实 replay 起点 | 否 | 起点是固定数据 |

冻结上下文结束后，原 requires_grad 状态恢复，以便下一次单独训练 WM 和 Q。

### 10.1 为什么还计算直接 Q 梯度

统一更新函数计算：

\[
g_Q=\nabla_\theta J_Q,\quad g_B=\nabla_\theta J_B,
\quad g=(1-w)g_Q+wg_B
\]

soft BPTT 组 w=1，最终 g=gB。gQ 仍被计算，用于记录范数和余弦，但没有作为独立起点 Q 梯度加进实际更新。

这不等于“Q 对 Actor 没影响”：JB 内部仍含末端 Q。当前组去掉的是直接 gQ 融合权重，未去掉 Q 末端引导。

函数显式取梯度、检查有限性、写入负梯度，经全局范数裁剪后执行一次 Adam step；不是先 Q 更新 Actor 再 BPTT 更新 Actor。

### 10.2 和原 BPTT 的裁剪差异

原 Actor 使用 `loss=-J/H`，全局范数裁剪后还做逐元素裁剪；本组的统一梯度器没有 `/H`，也没有逐元素裁剪。它们是需要对齐的优化协议差异，但 Adam 和裁剪会改变尺度关系，不能直接称本组参数步长大 H 倍。

## 11. 一次完整更新的伪代码

以下用于理解调用顺序，省略学习率调度、记录与保存：

```text
collect one stochastic environment step into real replay

if warmup and UTD conditions allow update:
    fit world model on real replay windows

    sample real transition batch and start batch
    fit Q1/Q2 on real soft Bellman targets
    soft-update target Q

    freeze WM and online Q parameters:
        s, hist, frame, prev_action = fixed replay start
        survival = 1
        for h in 0..H-1:
            build policy and critic inputs
            sample action with rsample
            if h == 0: compute direct JQ for diagnostics
            predict next state, reward, termination, observation
            save reward, log_prob and survival
            advance state, history, reference frame, previous action

        sample action at imagined endpoint
        endpoint_value = min online Q - alpha * log_prob
        JB = discounted survival-weighted soft return + endpoint value
        compute gQ and gB
        update Actor once with gB only
```

## 12. 参数、计数和运行方式

| 参数 | 首轮实际值 | 对 soft BPTT 的意义 |
|---|---:|---|
| 环境数／训练向量步 | 256／5000 | 128 万条训练交互 |
| Actor batch N | 256 | 每次展开 256 条想象轨迹 |
| H／dt | 8／0.02 s | 显式时域 0.16 s |
| policy history／future frames | 10／10 | 观测历史和参考窗口 |
| WM batch／batches_per_update | 256／2 | 每次 Actor 更新前做两次 WM optimizer step |
| q_updates | 1 | 每次 Actor 更新前更新一次双 Q |
| dynamics_weight | 1.0 | Actor 使用 gB |
| gamma／alpha／tau | 0.99／0.001／0.005 | 折扣、固定熵温度、target 软更新 |
| Actor lr／Q lr 初始值 | 2e-4／3e-4 | 随公共余弦时间表下降 |
| warmup | 40 | 只填充数据 |
| UTD | 0.4 | 按向量环境步累加 |

每个 Actor 更新的记录想象量为 N×H=2048；首轮 1984 次更新总计 4,063,232 个想象步。这里不把额外的末端 Q/Actor 查询算成一次 WM 想象步。

首轮协议是 5000 步、每 1000 步完整动作评估；仓库 `configs/experiments/r1_lafan1_40clips.json` 当前默认 2000 步、评估封顶 1000 步，不应混称为同一份协议。

以下只生成计划，不启动训练；实际执行者需先提供 CHECKPOINT 和 MOTIONS：

```bash
uv run --extra train --locked python scripts/run_suite.py \
  --checkpoint "$CHECKPOINT" \
  --motions "$MOTIONS" \
  --protocol configs/experiments/r1_lafan1_40clips.json \
  --methods bptt_soft --seeds 0 \
  --output outputs/bptt_soft_plan_v1 \
  --dry-run
```

真正执行需移除 `--dry-run`，并换新输出目录。本文没有执行该计划或启动训练。

## 13. 记录、保存与指标解释

保存格式与 Q-only 一致：均值 policy、完整 tadg_policy、WM、Q、target Q、优化器、采样器和配置。恢复随机训练应使用完整 TADG 检查点；只载入 policy 键仅恢复均值网络。

统一评估使用均值策略，因此本轮 0/40 完整完成不是仅由评估时的高斯采样造成。末尾 `recent_replay.pt` 只保留最近环形数据，不能替代完整训练流。

| 指标 | 实际含义 | 阅读时注意 |
|---|---|---|
| `imag_objective` | 含折扣奖励、熵和末端 Q 的 JB | 上升不保证真实任务变好 |
| `imag_return_undisc` | 存活加权、未折扣的 H 步奖励和 | 不含熵与末端 Q |
| `imag_reward_mean` | 所有步和 batch 的原始奖励均值 | 没有 survival 加权 |
| `imag_survival_H` | 预测 cH 的 batch 均值 | 是 WM 预测，不是真实完整动作成功率 |
| `dynamics_grad_norm` | 整个 JB 的梯度范数 | 包括熵、终止和末端 Q 路径，不是纯即时奖励梯度 |
| `q_actor_grad_norm` | 起点直接 JQ 的梯度范数 | 本组记录它，但直接融合权重为零 |
| `gradient_cosine` | gQ 与 gB 的夹角余弦 | 不表示两条梯度均用于本组更新 |
| `action_abs_max` | 想象 H 步中的最大采样动作幅度 | 不保证这些动作全部在真实仿真执行过 |
| `policy_std_mean` | 想象轨迹上的策略标准差均值 | 与 Q-only 的真实 batch 统计分布不同 |

如果要量化奖励、熵、末端值各自的贡献，当前总目标和未折扣奖励日志不够，需要额外分解记录；目标项数值大也不自动意味着梯度贡献大。

## 14. 已确认错误、稳定性风险与验证边界

**已确认错误：终止标签提前一步。** Replay 写入端把转移 done_next 放在起点行，Q 与 WM 读取端按状态标签消费。它会污染真实 Q 的 bootstrap 和 WM 的终止头；后者再通过 survival 影响本组 Actor 梯度。具体复现见 [首轮退化分析](首轮退化分析_20261105.md)。

**初始化风险：成熟策略配新 Q。** 本组没有保留原 V，也没有独立 Q 预热。新 Q 从第一次更新就承担想象末端值；w=1 不会解除这种依赖。

**分布风险：Q 用真实数据拟合，却在预测状态上被查询。** Actor 可以沿模型状态和动作导数更新，模型误差、Q 外推误差和终止头误差可能被优化过程利用。冻结参数不会消除这种风险。

**随机展开改变了原 BPTT。** 每一步都采样动作且训练标准差，观测历史和后续状态持续受采样影响。与“确定性均值 BPTT＋旧 V＋固定探索”的路径需要分别对照。

**当前表现不能判定动力学梯度本身无效。** 本组同时涉及新 Q、随机策略、终止标签和优化协议变化，应先保证标签正确，再分离无末端值、旧 V、新 Q，以及固定／可学习方差的影响。

## 15. 阅读源码与验证顺序

1. [experiment.py](../src/tadg/experiment.py)：确认 bptt_soft 将权重固定为 1。
2. [trainer.py](../src/tadg/trainer.py)：确认共享循环、Q 更新先后顺序、冻结边界。
3. [experience.py](../src/tadg/experience.py)：确认真实起点、历史和 Q features 的构造。
4. [policy.py](../src/tadg/policy.py)：确认高斯 head 和 rsample 的路径导数。
5. [imagination.py](../src/tadg/imagination.py)：逐步跟踪 state、frame、hist、alive 和末端 Q。
6. [compiled.py](../baselines/supermimic/src/supermimic/imagination/compiled.py)：确认 WM、解析奖励、终止与观测的一步计算。
7. [gradients.py](../src/tadg/gradients.py)：确认 soft-return 维度、时间折扣和最终参数更新。

[test_dual_gradient.py](../tests/test_dual_gradient.py) 已有 soft-return 独立数值检查和模型／末端 Q 路径导数检查。它们能检验公式与局部计算图，不能证明完整仿真长训练稳定，也没有覆盖当前 recorder/sampler 的终止语义组合。
