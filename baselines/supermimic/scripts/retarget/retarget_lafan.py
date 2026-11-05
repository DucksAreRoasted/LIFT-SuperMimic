"""G1 LAFAN -> official K1/R1 using robot_retargeter keypoints + Mink IK.

Output: 30 Hz CSV (xyzw) and 50 Hz NPZ (wxyz qpos/qvel plus mjlab body states).
The base reproduction still has G1-specific training code; these files do not
by themselves add K1/R1 training support.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO.parents[1]
WORKSPACE = PROJECT_ROOT
sys.path.insert(0, str(REPO / "src"))

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from supermimic.data.motion_convert import forward_kinematics, quat_rotate_inverse, resample_motion


def build_keypoints(source, target_config, reference, max_frames, output):
    sys.path.insert(0, str(reference / "scripts"))
    import robot_replay as replay
    import smpl_replay as smpl

    # Upstream progress bars are noisy in a batch log; the runner logs each clip.
    smpl.tqdm = lambda iterable, **kwargs: iterable
    source_config = REPO / "config/retarget/g1.yaml"
    src_cfg = yaml.safe_load(source_config.read_text())
    cfg = yaml.safe_load(target_config.read_text())
    qp = replay.load_robot_motion_csv(source)
    if qp.shape[1] != 36 or not np.isfinite(qp).all():
        raise ValueError(f"Invalid G1 LAFAN CSV: {source}")
    if max_frames:
        qp = qp[:max_frames]
    if len(qp) < 3:
        raise ValueError("At least three input frames are required")
    qp[:, 3:7] /= np.linalg.norm(qp[:, 3:7], axis=1, keepdims=True)
    source_links = {k: tuple(v) for k, v in src_cfg["robot_links"].items()}
    positions, quats, contact_pos = replay.build_robot_replay_buffers(
        qp, Path(src_cfg["robot_xml_path"]), source_links, tuple(src_cfg["contact_links"]))
    contact_names = tuple(replay.canonicalize_contact_name(n) for n in src_cfg["contact_links"])
    speeds, contacts = replay.compute_robot_contact_sequence(
        contact_pos, 30.0, src_cfg["contact_vel_calculate_window"],
        src_cfg["contact_vel_threshold"], src_cfg["contact_height_threshold"])
    links = {k: tuple(v) for k, v in cfg["robot_links"].items()}
    model_path = Path(cfg["robot_xml_path"])
    lengths = replay.compute_robot_link_lengths(target_config, model_path)
    points, rotations, raw_rotations, vectors, _, _ = replay.build_robot_retarget_keypoints(
        positions, quats, lengths, source_config, target_config)
    points, rotations, extra_names = replay.append_target_contact_keypoints(
        points, rotations, raw_rotations, links, target_config, model_path)
    scale, _, _ = replay.compute_leg_displacement_scale(lengths, vectors, 0.0)
    points = replay.scale_keypoint_frame_displacements(points, scale, root_keypoint_idx=0)
    names = ["hips_mean", *links, *extra_names]
    points, offsets = replay.offset_keypoints_by_contact_height(
        points, names, contact_names, contact_pos, contacts, src_cfg["contact_height_lpf_alpha"])
    payload = dict(keypoint_names=names, positions=points, quaternions=rotations,
                   contact_names=contact_names, contact_states=contacts, fps=30.0,
                   leg_displacement_scale=float(scale), contact_height_offsets=offsets)
    with output.open("wb") as f:
        pickle.dump(payload, f)
    return cfg, payload


def export_motion(qpos, model, source, output, config_path, metadata):
    # The generic resampler does not assume G1 joint count.
    raw = qpos.copy()
    raw[:, 3:7] = qpos[:, [4, 5, 6, 3]]
    motion = resample_motion(raw, 30.0, 50.0)
    bodies = forward_kinematics(model, motion)
    qp = np.concatenate((motion.root_pos, motion.root_quat, motion.joint_pos), axis=1)
    qv = np.concatenate((motion.root_lin_vel_w, quat_rotate_inverse(motion.root_quat, motion.root_ang_vel_w), motion.joint_vel), axis=1)
    joint_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    limited = model.jnt_limited[joint_ids].astype(bool)
    limits = model.jnt_range[joint_ids]
    violation = np.maximum(limits[:, 0] - motion.joint_pos, motion.joint_pos - limits[:, 1])[:, limited]
    max_violation = float(max(0.0, violation.max(initial=0)))
    if not np.isfinite(qp).all() or not np.isfinite(qv).all() or max_violation > 1e-5:
        raise ValueError(f"Invalid output: nonfinite or joint limit violation {max_violation}")
    foot_ids = [model.body(n).id - 1 for n in ("left_foot_end_link", "left_toe_link", "right_foot_end_link", "right_toe_link")]
    foot_z = bodies.pos_w[:, foot_ids, 2]
    speed = np.linalg.norm(bodies.lin_vel_w[:, foot_ids], axis=-1)
    stance = foot_z < 0.05
    metrics = dict(frames_30hz=len(qpos), frames_50hz=len(qp), duration_s=(len(qp) - 1) / 50,
                   max_joint_limit_violation_rad=max_violation,
                   max_joint_speed_rad_s=float(np.abs(motion.joint_vel).max()),
                   min_foot_marker_z_m=float(foot_z.min()),
                   foot_marker_below_minus_2cm_fraction=float((foot_z < -0.02).mean()),
                   near_floor_marker_speed_mean_m_s=float(speed[stance].mean()) if stance.any() else None,
                   root_height_range_m=[float(qp[:, 2].min()), float(qp[:, 2].max())])
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, fps=np.array([50.0], np.float32), qpos=qp.astype(np.float32), qvel=qv.astype(np.float32),
                        joint_pos=motion.joint_pos.astype(np.float32), joint_vel=motion.joint_vel.astype(np.float32),
                        body_pos_w=bodies.pos_w.astype(np.float32), body_quat_w=bodies.quat_w.astype(np.float32),
                        body_lin_vel_w=bodies.lin_vel_w.astype(np.float32), body_ang_vel_w=bodies.ang_vel_w.astype(np.float32),
                        joint_names=np.array([model.joint(j).name for j in joint_ids]), body_names=np.array(bodies.body_names),
                        source=np.array(str(source)), input_fps=np.array([30.0], np.float32),
                        model_xml=np.array(str(yaml.safe_load(config_path.read_text())["robot_xml_path"])),
                        name=np.array(source.stem), metadata_json=np.array(json.dumps(metadata)))
    tmp.replace(output)
    np.savetxt(output.with_suffix(".csv"), raw, delimiter=",", fmt="%.9f")
    output.with_suffix(".json").write_text(json.dumps({**metadata, **metrics}, indent=2))
    return metrics


def run_clip(job):
    robot, source, output_root, reference, max_frames = job
    source, output_root, reference = Path(source), Path(output_root), Path(reference)
    output = output_root / robot / f"{source.stem}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    config_path = REPO / f"config/retarget/{robot}.yaml"
    try:
        with output.with_suffix(".log").open("w", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            cfg, payload = build_keypoints(source, config_path, reference, max_frames, output.with_suffix(".keypoints.pkl"))
            import mink
            import robot_retarget as upstream
            upstream.tqdm = lambda iterable, **kwargs: iterable
            class Retarget(upstream.RobotRetarget):
                def _load_model_with_ground(self, path):
                    # Batch IK needs no extra visualization ground. K1 already
                    # has one, and the upstream loader adds a duplicate name.
                    return mujoco.MjModel.from_xml_path(path)

            rt = Retarget(
                cfg["robot_xml_path"], str(output.with_suffix(".keypoints.pkl")), cfg["ik_match_table"],
                solver="daqp", verbose=False, render_debug=False, joints_limit_offset_degrees={},
                contact_body_names=cfg["contact_links"], contact_position_cost=cfg["contact_pos_fixed_factor"])
            # Seed the free root near its first target instead of solving from ground zero.
            init = rt.model.qpos0.copy()
            init[3:7] = payload["quaternions"][0, 0]
            root_marker = rt.model.body("hips_sphere").pos
            init[:3] = payload["positions"][0, 0] - Rotation.from_quat(init[[4, 5, 6, 3]]).apply(root_marker)
            for j in range(1, rt.model.njnt):
                if rt.model.jnt_limited[j]:
                    adr = rt.model.jnt_qposadr[j]
                    init[adr] = np.clip(init[adr], *rt.model.jnt_range[j])
            rt.configuration.update(init)
            rt.update_targets(0)
            for _ in range(100):
                dt = rt.model.opt.timestep
                vel = mink.solve_ik(rt.configuration, rt.tasks, dt, rt.solver, rt.damping, limits=rt.ik_limits)
                rt.configuration.integrate_inplace(vel, dt)
                if np.linalg.norm(vel) * dt < 1e-6:
                    break
            rt.retarget()
            qpos = np.asarray(rt.result_pos)
            provenance = json.loads((REPO / "assets/retarget/provenance.json").read_text())
            metadata = dict(robot=robot, input=str(source.resolve()), input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                            model_xml=cfg["robot_xml_path"], config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                            max_frames=max_frames, provenance=provenance, method="robot_retargeter keypoint scaling + contact locking + Mink DAQP IK")
            metrics = export_motion(qpos, rt.model, source, output, config_path, metadata)
        return dict(robot=robot, clip=source.stem, ok=True, elapsed_s=time.time() - started, **metrics)
    except Exception as exc:
        import traceback
        with output.with_suffix(".log").open("a") as log:
            traceback.print_exc(file=log)
        return dict(robot=robot, clip=source.stem, ok=False, error=repr(exc), elapsed_s=time.time() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=WORKSPACE / "data/lafan1/g1")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/retarget/lafan1_v1")
    parser.add_argument("--reference", type=Path, default=WORKSPACE / "third_party/robot_retargeter")
    parser.add_argument("--robots", nargs="+", choices=["r1", "k1"], default=["r1", "k1"])
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--max-frames", type=int, default=0, help="Input frame cap for smoke checks only")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.max_frames < 0 or args.workers < 1:
        parser.error("Invalid frame cap or worker count")
    files = sorted(args.input_dir.resolve().glob(args.pattern))
    if not files:
        parser.error("No matching CSV files")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    results = []
    for robot in args.robots:
        for source in files:
            target = args.output_dir / robot / source.with_suffix(".npz").name
            if target.exists():
                meta_path = target.with_suffix(".json")
                if not meta_path.exists():
                    raise ValueError(f"Existing output without metadata: {target}. Choose a fresh --output-dir")
                meta = json.loads(meta_path.read_text())
                cfg_hash = hashlib.sha256((REPO / f"config/retarget/{robot}.yaml").read_bytes()).hexdigest()
                if meta["input_sha256"] != hashlib.sha256(source.read_bytes()).hexdigest() or meta["max_frames"] != args.max_frames or meta["config_sha256"] != cfg_hash:
                    raise ValueError(f"Existing output uses different input/config: {target}. Choose a fresh --output-dir")
                metric_names = ("frames_30hz", "frames_50hz", "duration_s", "max_joint_limit_violation_rad",
                                "max_joint_speed_rad_s", "min_foot_marker_z_m",
                                "foot_marker_below_minus_2cm_fraction", "near_floor_marker_speed_mean_m_s",
                                "root_height_range_m")
                results.append(dict(robot=robot, clip=source.stem, ok=True, skipped=True,
                                    **{name: meta[name] for name in metric_names}))
                continue
            jobs.append((robot, str(source), str(args.output_dir), str(args.reference.resolve()), args.max_frames))
    print(f"{len(files)} source clips, {len(jobs)} jobs, {args.workers} workers", flush=True)
    # Spawn avoids inheriting MuJoCo/BLAS state across processes.
    import multiprocessing as mp
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(run_clip, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
            (args.output_dir / "manifest.json").write_text(json.dumps(results, indent=2))
    (args.output_dir / "manifest.json").write_text(json.dumps(results, indent=2))
    if any(not r["ok"] for r in results):
        raise SystemExit("Some clips failed; see manifest.json and per-clip .log")


if __name__ == "__main__":
    main()
