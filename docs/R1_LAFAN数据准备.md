# R1 的 LAFAN 数据准备

统一入口是 `scripts/prepare_r1_lafan.py`。在本项目根目录执行，本地和服务器使用相同命令。脚本不启动训练，不依赖 CUDA。

## 复用原 SuperMimic 数据

如果已经跑过 R1 的原生 SuperMimic，直接复用同一批重定向 NPZ，不必重新下载或重定向：

```bash
uv run --locked python scripts/prepare_r1_lafan.py \
  --source /path/to/SuperMimic_Reproduce/data/retarget/lafan1_modelscope_v1/r1 \
  --checkpoint /path/to/SuperMimic_Reproduce/logs/r1_lafan1/ckpt_latest.pt
```

`--source` 指向包含 `.npz` 文件的 R1 目录。`--checkpoint` 可选，它只是复制已有检查点；机器人类型和网络配置会在训练加载时进一步校验。

默认输出：

```text
data/retarget/lafan1_modelscope_v1/r1/*.npz
data/retarget/lafan1_modelscope_v1/r1/dataset_manifest.json
data/checkpoints/r1_supermimic_pretrained.pt
```

使用 `--output /path/to/r1` 改变动作输出目录，使用 `--checkpoint-output /path/to/file.pt` 改变检查点位置。默认复制文件，不创建依赖旧仓库的软链接；相同内容跳过，不同内容报错，不覆盖旧数据。可用 `--pattern 'walk*'` 只准备一类动作。

## 没有现成数据时下载并重定向

```bash
uv run --locked python scripts/prepare_r1_lafan.py --download --workers 4
```

这会调用已有的 ModelScope LAFAN 准备流程：下载固定清单中的 G1 参考 CSV，再用现有 IK 流程重定向到 R1、输出 50 Hz NPZ。默认是 40 个片段的已有子集，并非对 LAFAN1 所有原始序列的全量转换。

脚本自动通过 uv 创建／使用独立的 `.venv-retarget`，只重定向 R1，不额外生成 K1；训练环境 `.venv` 不会被重定向依赖替换。该流程需要网络和 CPU，可能耗时较长。可以先处理单段：

```bash
uv run --locked python scripts/prepare_r1_lafan.py \
  --download --pattern walk1_subject1 --workers 1
```

使用 `--offline` 复用已有 ModelScope 下载缓存；使用 `--dataset-id`、`--revision` 指定数据来源。下载流程沿用已有代码，本轮只实跑了现成数据导入和训练，没有重新下载／运行 IK。

## 准备后训练

脚本核对 R1 的 26 个关节及顺序、模型身体名称、50 Hz、数组形状、有限数值和四元数，并为选中文件保存 SHA-256、帧数与时长。直接运行以下命令可重新生成已有目录的数据清单：

```bash
uv run --locked python scripts/prepare_r1_lafan.py
```

数据清单描述本次选择的文件，不划分训练／测试集，也不自动删除动作。仅行走数据不能直接套用要求五类动作的 `pilot.json`。现有 pilot 的训练与评估动作重合，用于训练机制对照，不能报告为未见动作泛化结果。

准备好已有预训练检查点后，在训练服务器运行：

```bash
uv sync --extra train --locked
uv run --extra train --locked python scripts/run_suite.py \
  --checkpoint data/checkpoints/r1_supermimic_pretrained.pt \
  --motions data/retarget/lafan1_modelscope_v1/r1 \
  --protocol configs/experiments/r1_lafan1_smoke.json \
  --methods supermimic dual_fixed --seeds 0 \
  --output outputs/r1_smoke_server_01
```

本地已有数据和检查点已复制到上述默认位置；`data/` 被 Git 忽略，不会随源码推送到服务器。服务器需要自行执行准备命令或额外传输整个数据目录。没有已有 R1 检查点时，当前这套继续训练对照不能直接启动，先使用原 SuperMimic 训练或传入你已有的模型。
