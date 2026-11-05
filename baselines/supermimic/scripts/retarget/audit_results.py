"""Audit generated reference motions. Numeric validity and motion quality are distinct.

Quality thresholds below are screening heuristics, not dynamic feasibility tests.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO.parents[1]


def audit(path):
    with np.load(path) as arr:
        qpos = arr["qpos"].astype(float)
        qvel = arr["qvel"].astype(float)
        model = mujoco.MjModel.from_xml_path(str(arr["model_xml"]))
        joint_names, body_names = arr["joint_names"].tolist(), arr["body_names"].tolist()
        fps = float(arr["fps"][0])
        recorded_pos = arr["body_pos_w"]
        recorded_quat = arr["body_quat_w"]
        recorded_lin = arr["body_lin_vel_w"]
        recorded_ang = arr["body_ang_vel_w"]
        joint_pos = arr["joint_pos"]
        joint_vel = arr["joint_vel"]
        fields_finite = all(np.isfinite(arr[k]).all() for k in
                            ("qpos", "qvel", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"))
        if joint_names != [model.joint(j).name for j in range(1, model.njnt)]:
            raise ValueError(f"Joint names/order mismatch: {path}")
        if body_names != [model.body(b).name for b in range(1, model.nbody)]:
            raise ValueError(f"Body names/order mismatch: {path}")
        assert qpos.shape == (len(qpos), model.nq) and qvel.shape == (len(qpos), model.nv)
        assert np.array_equal(qpos[:, 7:], joint_pos) and np.array_equal(qvel[:, 6:], joint_vel)
        quat_error = float(np.abs(np.linalg.norm(qpos[:, 3:7], axis=1) - 1).max())
        ids = np.flatnonzero(model.jnt_limited)[1:] if model.jnt_limited[0] else np.flatnonzero(model.jnt_limited)
        limits = model.jnt_range[ids]
        joint_vals = qpos[:, model.jnt_qposadr[ids]]
        limit_violation = float(max(0, (limits[:, 0] - joint_vals).max(), (joint_vals - limits[:, 1]).max()))
        # Independent FK spot checks, including velocities expressed at the body origin.
        data = mujoco.MjData(model)
        max_fk_error = max_velocity_error = 0.0
        for frame in np.linspace(0, len(qpos) - 1, 12, dtype=int):
            data.qpos[:] = qpos[frame]
            data.qvel[:] = qvel[frame]
            mujoco.mj_forward(model, data)
            max_fk_error = max(max_fk_error, float(np.abs(data.xpos[1:] - recorded_pos[frame]).max()))
            dots = np.abs((data.xquat[1:] * recorded_quat[frame]).sum(axis=1))
            assert np.max(np.abs(dots - 1)) < 1e-5
            for b in (1, model.body("left_toe_link").id, model.body("right_hand").id):
                vel = np.zeros(6)
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)
                # mj_objectVelocity BODY uses the inertial COM; export stores link origins.
                lin_at_origin = vel[3:] + np.cross(vel[:3], data.xpos[b] - data.xipos[b])
                max_velocity_error = max(max_velocity_error, float(np.abs(lin_at_origin - recorded_lin[frame, b - 1]).max()),
                                         float(np.abs(vel[:3] - recorded_ang[frame, b - 1]).max()))
        foot_ids = [model.body(n).id - 1 for n in ("left_foot_end_link", "left_toe_link", "right_foot_end_link", "right_toe_link")]
        foot_z = recorded_pos[:, foot_ids, 2]
        max_speed = float(np.abs(qvel[:, 6:]).max())
        penetration = float((foot_z < -0.02).mean())
        numerical_ok = fields_finite and quat_error < 1e-5 and limit_violation < 1e-5 and max_fk_error < 1e-4 and max_velocity_error < 1e-3
        # These broad screens expose problematic references without declaring them trainable.
        warnings = []
        if foot_z.min() < -0.05:
            warnings.append("foot marker penetrates >5cm")
        if penetration > 0.01:
            warnings.append(">1% foot marker samples below -2cm")
        if max_speed > 30:
            warnings.append("joint speed >30rad/s")
        cfg = yaml.safe_load((REPO / f"config/retarget/{path.parent.name}.yaml").read_text())
        with path.with_suffix(".keypoints.pkl").open("rb") as f:
            import pickle
            targets = pickle.load(f)
        raw = np.loadtxt(path.with_suffix(".csv"), delimiter=",")
        raw[:, 3:7] = raw[:, [6, 3, 4, 5]]
        point_errors = []
        for frame in np.linspace(0, len(raw) - 1, min(300, len(raw)), dtype=int):
            data.qpos[:] = raw[frame]
            mujoco.mj_forward(model, data)
            for name, (body, cost, _) in cfg["ik_match_table"].items():
                if cost:
                    slot = targets["keypoint_names"].index(name)
                    point_errors.append(np.linalg.norm(data.xpos[model.body(body).id] - targets["positions"][frame, slot]))
        return dict(robot=path.parent.name, clip=path.stem, numeric_ok=bool(numerical_ok), frames=len(qpos), fps=fps,
                    duration_s=(len(qpos) - 1) / fps, max_quaternion_norm_error=quat_error,
                    max_joint_limit_violation_rad=limit_violation, max_fk_error_m=max_fk_error,
                    max_body_velocity_error=max_velocity_error, max_joint_speed_rad_s=max_speed,
                    min_foot_marker_z_m=float(foot_z.min()), below_minus_2cm_fraction=penetration,
                    sampled_keypoint_error_mean_m=float(np.mean(point_errors)),
                    sampled_keypoint_error_p95_m=float(np.percentile(point_errors, 95)),
                    quality_flags="; ".join(warnings))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    rows = [audit(path) for path in sorted(args.directory.glob("*/*.npz")) if not path.name.endswith(".tmp.npz")]
    if not rows:
        parser.error("No motions found")
    summary = {}
    for robot in sorted({r["robot"] for r in rows}):
        group = [r for r in rows if r["robot"] == robot]
        summary[robot] = dict(clips=len(group), numeric_pass=sum(r["numeric_ok"] for r in group),
                              quality_flagged_clips=sum(bool(r["quality_flags"]) for r in group),
                              frames=sum(r["frames"] for r in group), duration_hours=sum(r["duration_s"] for r in group) / 3600,
                              worst_foot_marker_z_m=min(r["min_foot_marker_z_m"] for r in group),
                              max_joint_speed_rad_s=max(r["max_joint_speed_rad_s"] for r in group))
    report = dict(summary=summary, notes="Quality flags are heuristic screens. IK is kinematic and does not establish dynamic feasibility or self-collision freedom.", per_clip=rows)
    (args.directory / "quality_report.json").write_text(json.dumps(report, indent=2))
    with (args.directory / "quality_report.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    if any(not r["numeric_ok"] for r in rows):
        raise SystemExit("Numerical validation failed")


if __name__ == "__main__":
    main()
