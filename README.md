# LIFT-SuperMimic / TADG

基于已有 SuperMimic 复现，直接实现 mimic 的共享随机 Actor、SAC／Q 梯度与世界模型 BPTT 梯度融合。

当前已有固定权重双梯度训练入口，复用原环境与世界模型。R1＋LAFAN 的原基线／双梯度短训练已通过；可信度权重、自适应时域尚未实现。

**正式训练在服务器执行。** 本次经用户允许完成本地 R1 冒烟测试，原基线和双梯度的更新、评估、保存与汇总均已跑通；尚未完成服务器长训练及多种子对照。

- [服务器启动、对照设置与数据记录](docs/服务器实验与记录.md)
- [一次启动自动化服务器实验](docs/自动化服务器实验.md)
- [R1 的 LAFAN 数据准备](docs/R1_LAFAN数据准备.md)
- [R1 冒烟测试结果](docs/R1_LAFAN冒烟结果_20261105.md)
- [服务器实验启动器](scripts/run_suite.py)
- [代码执行说明](任务执行指南_TADG.md)
- [按算法改动组织的实施计划](实施计划_TADG.md)
- [研究方案与相关工作](研究方案_LIFT与SuperMimic融合.md)
- [训练入口](scripts/train_tadg.py)

基线代码已经纳入 `baselines/supermimic/`，第三方依赖统一位于根目录 `third_party/`。根 uv workspace 和 `uv.lock` 统一管理基线与 TADG，不再依赖旧仓库目录。

在服务器本项目根目录执行：

```bash
uv sync --extra train --locked
uv run --extra train --locked sm-train --help
uv run --extra train --locked tadg-train --help
```

目录、重定向选项与迁移说明见[代码整合与 uv](docs/代码整合与uv.md)。数据和检查点不随 Git 上传，请在服务器准备或单独传输。
