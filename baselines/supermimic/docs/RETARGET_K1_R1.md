# LAFAN G1 → Booster K1 / Unitree R1

本地已接入 [robot_retargeter](https://github.com/ccrpRepo/robot_retargeter) 的机器人间重定向流程，并处理 `data/lafan1/g1` 中的 40 条完整动作。原始数据与此前其他脚本生成的数据均保留，新结果位于：

```
data/retarget/lafan1_official_v1/
├── k1/                       # 40 条，22 个关节
├── r1/                       # 40 条，26 个关节（含头部两个关节）
├── manifest.json             # 运行结果与耗时
├── quality_report.json       # 数值核验与质量筛查
└── quality_report.csv
```

## 模型与方法

- [Booster 官方模型](https://github.com/BoosterRobotics/booster_assets)：`third_party/booster_assets/robots/K1/K1_22dof.xml`。采用串联 22 DoF 版本。
- [Unitree 官方模型](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/r1_description)：`third_party/unitree_ros/robots/r1_description/R1.urdf`。转为 MuJoCo XML，增加自由根关节，保留 26 个转动关节；不是此前的 24 DoF mjlab 版本。
- 生成模型位于 `assets/retarget/`，新增髋中心、颈部、手端和脚底固定标记；关节顺序、限位及版本记录位于 `assets/retarget/provenance.json`。URDF 经 MuJoCo XML 序列化后有约 1e-5 rad 以内的限位舍入差异。
- 直接调用参考项目的 G1 前向运动学、逐段连杆长度缩放、按腿长缩放根位移、接触状态检测、高度修正、接触位置锁定和 Mink/DAQP IK。K1 补充了关键点映射与手臂零姿态坐标变换；它的手臂自由度少于 G1，不能逐关节一一复制。
- 不启用参考项目默认的额外膝/肘限位偏移，也不额外增加膝关节弯曲角；第一帧先做 IK 收敛，后续逐帧以上一帧结果初始化。

参考项目源码统一放在 `third_party/robot_retargeter/`，作为当前项目的依赖；所有上游版本记录在 `third_party/UPSTREAM.json`。其配置解析器要求连杆等列表采用行内 YAML，本集成已兼容。未修改参考项目代码。

## 直接查看结果

以下命令均在项目根目录 `/home/ducks/SuperMimic_Reproduce` 执行。统一使用 uv 管理：先执行 `uv sync --extra retarget --no-dev --locked`。完整操作见根目录 [命令记录本](../命令记录.md)。旧 `.venv-retarget` 仅保留作历史环境。

```bash
uv run --extra retarget --no-dev --locked python scripts/retarget/replay_motion.py \
  --motion data/retarget/lafan1_official_v1/k1/walk1_subject1.npz \
  --seconds 261.3 --loop
```

将 `k1` 换成 `r1` 可查看 R1。关闭窗口退出。脚本默认只播放前 10 秒，`--seconds` 可指定观看长度。

三机器人并排预览已生成：

- `outputs/retarget/dance_comparison.mp4`：同一段舞蹈，G1 / K1 / R1。
- `outputs/retarget/walk_smoke.mp4`：步行开头的短片段。

重新录制对比视频，不需要桌面窗口：

```bash
uv run --extra retarget --no-dev --locked python scripts/retarget/replay_motion.py \
  --source-csv data/lafan1/g1/dance1_subject2.csv \
  --motion data/retarget/lafan1_official_v1/k1/dance1_subject2.npz \
           data/retarget/lafan1_official_v1/r1/dance1_subject2.npz \
  --start-time 20 --seconds 4 --fps 15 \
  --video outputs/retarget/dance_comparison_new.mp4
```

## 重跑

```bash
# 重新生成模型和配置；不修改官方资源。
uv run --extra retarget --no-dev --locked python scripts/retarget/prepare_models.py

# 默认 40 条 × 两种机器人；同配置的已有输出会跳过。
uv run --extra retarget --no-dev --locked python scripts/retarget/retarget_lafan.py \
  --output-dir data/retarget/lafan1_official_v1 --workers 8

# 调参时换一个输出目录，保留旧结果；先做短片段测试。
uv run --extra retarget --no-dev --locked python scripts/retarget/retarget_lafan.py \
  --pattern walk1_subject1.csv --max-frames 120 \
  --output-dir data/retarget/my_smoke --workers 2

uv run --extra retarget --no-dev --locked python scripts/retarget/test_retarget.py
uv run --extra retarget --no-dev --locked python scripts/retarget/audit_results.py \
  data/retarget/lafan1_official_v1
```

配置文件：`config/retarget/g1.yaml`、`k1.yaml`、`r1.yaml`。生成的 XML/YAML 和 NPZ 保存了本机绝对模型路径；移动工作区后先重新执行 `prepare_models.py`，并重新生成相应动作。改变输入/配置后，批处理要求另选输出目录，避免覆盖已有结果。

## 数据格式

每条动作包含 `.csv`、`.npz`、`.json`、`.keypoints.pkl`、`.log`。

| 文件/字段 | 约定 |
|---|---|
| 输入 G1 CSV | 30 Hz；根位置米；四元数 xyzw；29 个关节弧度 |
| 输出 CSV | 30 Hz；根位置米；四元数 xyzw；目标模型的 XML 关节顺序 |
| NPZ `fps` | `[50.0]` |
| NPZ `qpos` | `[T,7+N]`，根位置 + **wxyz** 四元数 + 关节角 |
| NPZ `qvel` | `[T,6+N]`，根世界线速度 + **根局部角速度** + 关节速度 |
| `joint_pos`, `joint_vel` | `[T,N]`，弧度 / 弧度每秒 |
| `body_pos_w`, `body_quat_w` | 排除 world 的全部 body，世界坐标系，wxyz |
| `body_lin_vel_w`, `body_ang_vel_w` | 世界坐标系；线速度定义在 link 原点 |
| `joint_names`, `body_names` | 显式记录顺序，包含新增固定标记体 |
| `model_xml`, `metadata_json` | 模型路径、输入哈希、配置哈希及仓库版本 |

50 Hz 重采样复用原仓库的四元数插值和速度计算，FK 使用实际目标模型。读取时不要截取前 N 列来推断 SDK 顺序，应按 `joint_names` 映射。

## 验证与使用边界

完整结果：每种机器人 40 条、441140 帧（50 Hz）、2.45056 小时；全部通过数值回读与抽样运动学核验。经验质量阈值标记 K1 的 30 条、R1 的 20 条动作，需要在训练前结合报告和回放检查。全库最深脚底标记位置分别约 -9.05 cm / -12.75 cm，最高关节速度约 54.79 / 45.26 rad/s；这些峰值是未施加动力学约束的 IK 参考数据中实际存在的问题。

`test_retarget.py` 验证 G1 源关节顺序、官方目标限位、四元数符号翻转下的重采样，以及导出速度与 MuJoCo 独立速度接口的一致性。`audit_results.py` 对生成文件检查有限值、维度、关节顺序、四元数范数与关节限位，并抽样回算 FK 和 link 原点速度；同时报告关键点误差、脚底标记穿地和关节速度峰值。

报告里的 `numeric_ok` 表示数据结构和数值核验通过。`quality_flags` 使用脚底标记低于 -5 cm、超过 1% 标记样本低于 -2 cm、关节速度超过 30 rad/s 等经验阈值筛查，不能作为动力学可执行性证明。剧烈动作可能仍有滑脚、穿地或受限位影响的姿态偏差，未施加自碰撞约束。预览视频是运动学回放。

训练入口现已支持 `--robot r1` 和 `--robot k1`，可直接读取对应重定向 NPZ；无需再转换为 G1 格式。机器人驱动与跟踪配置见 [多机器人训练](TRAINING_ROBOTS.md)，常用指令见根目录命令记录。

验证环境：MuJoCo 3.17.0、Mink 1.1.1、DAQP 0.8.7、NumPy 2.2.6、SciPy 1.15.3、PyYAML 6.0.3。动作数据与机器人资源遵循各自上游许可。

uv 迁移后另用 Python 3.12.14、MuJoCo 3.17.0、NumPy 2.5.4、SciPy 1.18.1、CPU PyTorch 2.6.0 完成了短片段重定向、数值核验、窗口退出及离屏视频测试。实际安装版本以根目录 `uv.lock` 为准。
