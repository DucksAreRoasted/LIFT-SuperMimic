# Shared third-party code and robot assets

These directories were imported from the committed `third_party/` tree of
`DucksAreRoasted/SuperMimic_Reproduce` at
`555d1b18232ef110de4526e221d2fb441068dd07`.

- `robot_retargeter/`: retargeting reference code and the G1 model assets.
- `booster_assets/`: the K1 model and meshes included by the source repository.
- `unitree_ros/`: the R1 URDF and meshes included by the source repository.
- `UPSTREAM.json`: original dependency URLs/commits and the imported reproduction revision.

The baseline uses these shared directories directly. There is no nested
`baselines/supermimic/third_party/`, no nested Git checkout, and no runtime dependency
on `/home/ducks/SuperMimic_Reproduce`.

Each dependency retains its existing license/notice files and attribution.
This directory does not change the upstream licensing of code or robot assets.
