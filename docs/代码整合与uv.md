# 代码整合与 uv 使用

更新：2026-11-05。整合后已获用户授权完成 R1 本地短训练，见[R1 冒烟结果](R1_LAFAN冒烟结果_20261105.md)。正式训练仍在服务器执行。

## 目录

```text
LIFT-SuperMimic/
  pyproject.toml                 统一依赖、train／retarget 选项、workspace
  uv.lock                        唯一的依赖锁文件
  .python-version                Python 3.12
  src/tadg/                      新算法
  baselines/supermimic/
    src/supermimic/               已复现的基线算法
    scripts/                     基线评估、数据处理和重定向工具
    assets/                      基线自身的模型／配置资产
    config/
    tests/
    LICENSE
    UPSTREAM.json                导入地址与提交
    pyproject.upstream.toml       原始依赖配方存档，不参与 uv 解析
  third_party/
    robot_retargeter/
    booster_assets/
    unitree_ros/
    UPSTREAM.json
  data/                          所有方法共用，忽略于 Git
  logs/                          训练结果，忽略于 Git
```

基线来自用户指定的仓库 `DucksAreRoasted/SuperMimic_Reproduce`，提交
`555d1b18232ef110de4526e221d2fb441068dd07`。原本地仓库恰好位于该提交，导入内容由 Git 提交导出，与其未提交的部署改动隔离。原目录未修改。

不使用子模块或嵌套 Git；服务器只需取本仓库，不必再克隆 SuperMimic。第三方目录已经展开到根目录，基线的引用路径已调整。未复制数据、检查点和演示视频，它们使用服务器现有资源。

## 服务器安装

始终在本项目根目录执行：

```bash
uv sync --extra train --locked
```

这会在根 `.venv/` 中以 editable workspace 包安装 `tadg` 和 `supermimic`，共享 Torch、MuJoCo、Warp 和 mjlab。无需设置 `PYTHONPATH`，也无需提供旧仓库的路径或解释器。保留原训练所用 mjlab 提交和 CUDA 13.0 Torch 索引；普通 PyPI 包沿用原本地复现环境的华为镜像。

统一锁文件由现有可用环境的锁文件作为种子重新解析，共 170 个包。已运行 `uv lock --check` 检查元数据一致性；GPU 环境随后已用于 R1 短训练，结果单独记录。

## 基线与 TADG

```bash
# 原 SuperMimic 更新方式
uv run --extra train --locked sm-train --robot r1 --run-name baseline_r1

# TADG 固定双梯度，按实际检查点和动作目录传参
uv run --extra train --locked tadg-train \
  --robot r1 \
  --warm-start /server/path/checkpoint.pt \
  --data.npz-dir /server/path/r1_npz \
  --run-name tadg_r1

# 使用原基线的独立评估脚本
uv run --extra train --locked python baselines/supermimic/scripts/eval_benchmark.py \
  --ckpt /server/path/checkpoint.pt --npz-dir /server/path/r1_npz
```

服务器上使用 G1 时替换 `--robot`、检查点和对应动作数据。原网络尺寸与 checkpoint 仍须匹配。

多方法对照入口现在默认使用仓库内的基线和当前 uv 解释器：

```bash
uv run --extra train --locked python scripts/run_suite.py \
  --checkpoint /server/path/checkpoint.pt \
  --motions /server/path/retargeted_npz \
  --protocol configs/experiments/server_smoke.json \
  --methods supermimic dual_fixed --seeds 0 \
  --output /server/results/tadg_first_run
```

不再使用 `--supermimic-root`。通常也不需要 `--python`；只有明确覆盖解释器时才使用该参数，并自行保证仍指向本 workspace 的环境。完整数据记录与对照设置见[服务器实验说明](服务器实验与记录.md)。

## 重定向依赖

原工程的训练与重定向配方分别要求不同版本的 Torch 和 MuJoCo，不能强行同时安装。根 uv 将二者声明为互斥选项，但使用同一份锁文件。基线和 TADG 的训练环境始终相同。

如需重定向，可另外指定一个由同一 uv 项目管理的环境，避免来回替换训练环境：

```bash
UV_PROJECT_ENVIRONMENT=.venv-retarget uv sync --extra retarget --no-dev --locked
UV_PROJECT_ENVIRONMENT=.venv-retarget uv run --extra retarget --no-dev --locked \
  python baselines/supermimic/scripts/data/prepare_lafan_modelscope.py
```

此命令会进行数据准备，按需在服务器执行；本次整合没有执行它。默认数据输出已统一到根 `data/`，生成的基线专属模型与 IK 配置仍位于 `baselines/supermimic/assets/` 和 `config/`。

## 对比可追溯性

实验清单会记录 workspace Git 状态、基线导入提交、根 `uv.lock` 哈希、当前解释器及包版本。源码快照包含两个算法、基线资产与共享第三方代码，避免以后无法判断是否使用了不同环境或不同基线。

基线 README 的原始说明保留作来源资料；当前执行目录和 uv 命令以本页为准。
