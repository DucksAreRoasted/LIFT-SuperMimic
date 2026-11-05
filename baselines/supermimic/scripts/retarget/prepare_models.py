"""Build IK models from the official assets, adding only fixed semantic markers.

Run with uv run --extra retarget --no-dev --locked python. No original assets are edited.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

REPO = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO.parents[1]
WORKSPACE = PROJECT_ROOT


class ConfigDumper(yaml.SafeDumper):
    pass


ConfigDumper.add_representer(list, lambda dumper, value: dumper.represent_sequence("tag:yaml.org,2002:seq", value, flow_style=True))


def marker(root, parent, name, pos):
    body = root.find(f".//body[@name='{parent}']")
    if body is None:
        raise ValueError(f"Missing marker parent {parent}")
    ET.SubElement(body, "body", name=name, pos=" ".join(map(str, pos)))


def absolute_meshes(root, meshdir):
    compiler = root.find("compiler")
    compiler.set("meshdir", str(meshdir.resolve()))
    compiler.set("fusestatic", "false")


def save(root, path):
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(path))
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"{path}: expected a free root joint")
    return model


def make_r1(official, out):
    urdf = ET.parse(official / "R1.urdf").getroot()
    compiler = urdf.find("mujoco/compiler")
    compiler.set("meshdir", str((official / "meshes").resolve()))
    compiler.set("fusestatic", "false")
    compiler.set("balanceinertia", "true")
    for mesh in urdf.iter("mesh"):
        mesh.set("filename", Path(mesh.get("filename")).name)
    model = mujoco.MjModel.from_xml_string(ET.tostring(urdf, encoding="unicode"))
    mujoco.mj_saveLastXML(str(out), model)
    root = ET.parse(out).getroot()
    absolute_meshes(root, official / "meshes")
    pelvis = root.find("worldbody/body")
    pelvis.set("pos", "0 0 0.78")
    pelvis.insert(0, ET.Element("freejoint", name="floating_base_joint"))
    marker(root, "pelvis_link", "hips_sphere", [0.0325, 0, -0.0902351])
    marker(root, "waist_yaw_link", "neck_sphere", [0, 0, 0.19749])
    for side, sign in (("left", 1), ("right", -1)):
        marker(root, f"{side}_shoulder_pitch_link", f"{side}_shoulder_sphere", [0, sign * 0.047132, -0.025693])
        marker(root, f"{side}_wrist_roll_link", f"{side}_hand", [0.07, 0, 0])
        marker(root, f"{side}_ankle_roll_link", f"{side}_foot_end_link", [-0.04, 0, -0.04])
        marker(root, f"{side}_ankle_roll_link", f"{side}_toe_link", [0.12, 0, -0.04])
    return save(root, out)


def make_k1(official, out):
    root = ET.parse(official / "K1_22dof.xml").getroot()
    absolute_meshes(root, official / "meshes")
    marker(root, "trunk", "hips_sphere", [0, 0, -0.103])
    marker(root, "trunk", "neck_sphere", [0.0025, 0, 0.171])
    for side, sign in (("left", 1), ("right", -1)):
        # The physical elbow is the elbow-yaw origin, beyond the short pitch link.
        # Wrist/hand point is the end of the last arm segment (no wrist actuator).
        marker(root, f"{side}_elbow_yaw_link", f"{side}_hand", [0, sign * 0.12, 0])
        marker(root, f"{side}_ankle_roll_link", f"{side}_foot_end_link", [-0.064, 0, -0.038])
        marker(root, f"{side}_ankle_roll_link", f"{side}_toe_link", [0.116, 0, -0.038])
    return save(root, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=WORKSPACE / "third_party/robot_retargeter")
    parser.add_argument("--booster", type=Path, default=WORKSPACE / "third_party/booster_assets")
    parser.add_argument("--unitree", type=Path, default=WORKSPACE / "third_party/unitree_ros")
    args = parser.parse_args()
    reference = args.reference.resolve()
    outdir = REPO / "assets/retarget"
    outdir.mkdir(parents=True, exist_ok=True)
    configdir = REPO / "config/retarget"
    configdir.mkdir(parents=True, exist_ok=True)
    models = {
        "r1": make_r1(args.unitree / "robots/r1_description", outdir / "r1.xml"),
        "k1": make_k1(args.booster / "robots/K1", outdir / "k1.xml"),
    }
    configs = {}
    for name in ("g1", "r1"):
        configs[name] = yaml.safe_load((reference / f"config/robot/{name}.yaml").read_text())
    configs["g1"]["robot_xml_path"] = str(reference / "asset/robot/g1_description/mjcf/g1.xml")
    configs["r1"]["robot_xml_path"] = str(outdir / "r1.xml")
    k1 = copy.deepcopy(configs["r1"])
    k1["robot_xml_path"] = str(outdir / "k1.xml")
    k1["robot_links"] = {
        "left_hip": ["hips_sphere", "left_hip_roll_link"],
        "left_thigh": ["left_hip_roll_link", "left_knee_pitch_link"],
        "left_calf": ["left_knee_pitch_link", "left_ankle_roll_link"],
        "right_hip": ["hips_sphere", "right_hip_roll_link"],
        "right_thigh": ["right_hip_roll_link", "right_knee_pitch_link"],
        "right_calf": ["right_knee_pitch_link", "right_ankle_roll_link"],
        "neck": ["hips_sphere", "neck_sphere"],
        "head": ["neck_sphere", "aahead_pitch_link"],
        "left_shoulder": ["neck_sphere", "left_shoulder_roll_link"],
        "left_arm": ["left_shoulder_roll_link", "left_elbow_yaw_link"],
        "left_fore_arm": ["left_elbow_yaw_link", "left_hand"],
        "right_shoulder": ["neck_sphere", "right_shoulder_roll_link"],
        "right_arm": ["right_shoulder_roll_link", "right_elbow_yaw_link"],
        "right_fore_arm": ["right_elbow_yaw_link", "right_hand"],
    }
    k1["ik_match_table"] = {"hips_mean": ["hips_sphere", 100, 3]}
    for side in ("left", "right"):
        for link, cost, rot_cost in (("hip", 30, 1), ("thigh", 5, 2), ("calf", 50, 3),
                                     ("shoulder", 15, 1), ("arm", 10, 1), ("fore_arm", 10, 0.3)):
            semantic = f"{side}_{link}"
            k1["ik_match_table"][semantic] = [k1["robot_links"][semantic][1], cost, rot_cost]
    k1["ik_match_table"]["head"] = ["aahead_pitch_link", 0, 0.5]
    # K1 has an outstretched-arm zero pose; G1/R1 have downward upper arms.
    for side, angle in (("left", -90), ("right", 90)):
        for semantic in (f"{side}_arm", f"{side}_fore_arm", f"{side}_hand"):
            source = configs["g1"]["key_frame_config"][semantic]["axis_map_cols"]
            mat = np.column_stack([source[axis] for axis in "xyz"]) @ Rotation.from_euler("x", angle, degrees=True).as_matrix()
            k1["key_frame_config"][semantic]["axis_map_cols"] = {axis: mat[:, i].round(8).tolist() for i, axis in enumerate("xyz")}
    configs["k1"] = k1
    for name, cfg in configs.items():
        cfg["verbose"] = False
        cfg["render_debug"] = False
        # Preserve the official mechanical limits. Optional knee bend is disabled.
        cfg["joints_limit_offset_degrees"] = {}
        cfg["enable_knee_angle_offset_degrees"] = False
        if name == "r1":
            cfg["ik_match_table"]["hips_mean"][2] = 3
        (configdir / f"{name}.yaml").write_text(yaml.dump(cfg, Dumper=ConfigDumper, sort_keys=False))
    provenance = {}
    upstream = json.loads((PROJECT_ROOT / "third_party/UPSTREAM.json").read_text())
    for name, path in (("supermimic", REPO), ("robot_retargeter", reference), ("booster_assets", args.booster), ("unitree_ros", args.unitree)):
        provenance[name] = {"path": str(path.resolve()), **upstream[name]}
    for name, model in models.items():
        provenance[name] = {"nq": model.nq, "nv": model.nv, "joint_names": [model.joint(j).name for j in range(1, model.njnt)]}
        print(name, model.nq, model.nv, provenance[name]["joint_names"])
    (outdir / "provenance.json").write_text(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
