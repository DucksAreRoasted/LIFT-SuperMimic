"""Replay NPZ references in MuJoCo, or render a side-by-side comparison video.

This is kinematic reference playback, not a learned policy or dynamic rollout.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO.parents[1]
WORKSPACE = PROJECT_ROOT


@contextmanager
def passive_viewer(model, data):
    from mujoco import viewer as mj_viewer

    existing_threads = set(threading.enumerate())
    handle = mj_viewer.launch_passive(model, data)
    viewer_threads = set(threading.enumerate()) - existing_threads
    try:
        yield handle
    finally:
        # Handle.close() only requests exit. MuJoCo 3.17 launches a daemon
        # render thread; let it destroy its GL context before GLFW's atexit
        # cleanup runs, otherwise short replays can segfault on shutdown.
        handle.close()
        for thread in viewer_threads:
            thread.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, nargs="+", required=True)
    parser.add_argument("--source-csv", type=Path, help="Add the source G1 to a comparison video")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--start-time", type=float, default=0)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--headless", action="store_true", help="Check kinematics without rendering")
    args = parser.parse_args()
    if args.start_time < 0 or args.seconds <= 0 or args.fps <= 0:
        parser.error("Invalid start time, duration or fps")
    if not args.video and not args.headless and (len(args.motion) != 1 or args.source_csv):
        parser.error("Interactive playback accepts one NPZ; use --video for comparisons")
    if args.video:
        os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    import numpy as np

    clips = []
    if args.source_csv:
        qp = np.loadtxt(args.source_csv, delimiter=",")
        qp[:, 3:7] = qp[:, [6, 3, 4, 5]]
        clips.append(("G1 source", WORKSPACE / "third_party/robot_retargeter/asset/robot/g1_description/mjcf/g1.xml", qp, 30.0))
    for path in args.motion:
        with np.load(path) as arr:
            qp, fps = arr["qpos"].copy(), float(arr["fps"][0])
            xml = Path(str(arr["model_xml"]))
            names = arr["joint_names"].tolist()
        model = mujoco.MjModel.from_xml_path(str(xml))
        if names != [model.joint(j).name for j in range(1, model.njnt)]:
            raise ValueError(f"NPZ/model joint order mismatch: {path}")
        if qp.shape[1] != model.nq or not np.isfinite(qp).all():
            raise ValueError(f"Invalid qpos: {path}")
        clips.append((f"{path.parent.name.upper()} retarget", xml, qp, fps))

    scenes = []
    for name, xml, qp, fps in clips:
        spec = mujoco.MjSpec.from_file(str(xml))
        if not any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in spec.geoms):
            spec.worldbody.add_geom(name="replay_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                    size=[0, 0, 0.05], rgba=[0.25, 0.28, 0.32, 1], contype=0, conaffinity=0)
        spec.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
        model = spec.compile()
        model.vis.global_.offwidth = 512
        model.vis.global_.offheight = 512
        data = mujoco.MjData(model)
        scenes.append((name, model, data, qp, fps))
    end_time = min(args.start_time + args.seconds, min((len(s[3]) - 1) / s[4] for s in scenes))
    if end_time <= args.start_time:
        parser.error("Start time is beyond the available clip")
    times = np.arange(args.start_time, end_time, 1 / args.fps)
    if args.headless:
        for _, model, data, qp, fps in scenes:
            for t in times:
                data.qpos[:] = qp[min(round(t * fps), len(qp) - 1)]
                mujoco.mj_forward(model, data)
                if not np.isfinite(data.xpos).all():
                    raise ValueError("Non-finite body position")
        print(f"Checked {len(times)} sample frames for each of {len(scenes)} clips")
        return
    if args.video:
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw

        args.video.parent.mkdir(parents=True, exist_ok=True)
        renderers = [mujoco.Renderer(model, height=512, width=512) for _, model, *_ in scenes]
        try:
            with imageio.get_writer(args.video, fps=args.fps, codec="libx264", quality=7) as writer:
                for t in times:
                    panels = []
                    for renderer, (name, model, data, qp, fps) in zip(renderers, scenes):
                        data.qpos[:] = qp[min(round(t * fps), len(qp) - 1)]
                        mujoco.mj_forward(model, data)
                        camera = mujoco.MjvCamera()
                        camera.lookat[:] = [*data.qpos[:2], max(0.35, data.qpos[2] * 0.8)]
                        camera.distance = 2.4
                        camera.azimuth = 135
                        camera.elevation = -12
                        renderer.update_scene(data, camera=camera)
                        panel = Image.fromarray(renderer.render())
                        draw = ImageDraw.Draw(panel)
                        draw.rectangle((0, 0, 512, 30), fill=(20, 24, 30))
                        draw.text((12, 8), f"{name} | t={t:.2f}s | reference playback", fill="white")
                        panels.append(np.asarray(panel))
                    writer.append_data(np.concatenate(panels, axis=1))
        finally:
            for renderer in renderers:
                renderer.close()
        print(args.video.resolve())
    else:
        _, model, data, qp, fps = scenes[0]
        with passive_viewer(model, data) as viewer:
            viewer.cam.distance = 2.4
            viewer.cam.elevation = -12
            viewer.cam.azimuth = 135
            while viewer.is_running():
                for t in times:
                    if not viewer.is_running():
                        return
                    started = time.perf_counter()
                    data.qpos[:] = qp[min(round(t * fps), len(qp) - 1)]
                    mujoco.mj_forward(model, data)
                    viewer.cam.lookat[:] = data.qpos[:3]
                    viewer.sync()
                    time.sleep(max(0, 1 / args.fps - (time.perf_counter() - started)))
                if not args.loop:
                    break


if __name__ == "__main__":
    main()
