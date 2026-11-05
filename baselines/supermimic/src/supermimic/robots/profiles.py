"""Single source of joint dimensions, tracking bodies, collision and PD setup.

R1 uses the vendor baseline in assets/raw/r1_constants.py for its limbs.
Head gains are a simulation baseline; effort limits come from the official URDF.
K1 effort limits and armatures come from the official 22-DoF MJCF. Its PD gains
use the same 10 Hz, damping-ratio 2 construction as mjlab's G1.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = ROOT.parents[1]
THIRD_PARTY = PROJECT_ROOT / "third_party"
DATA_ROOT = PROJECT_ROOT / "data"
G1_TRACKED = (
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)


@dataclass(frozen=True)
class RobotDimensions:
    n_joints: int = 29

    @property
    def nq(self): return 7 + self.n_joints
    @property
    def nv(self): return 6 + self.n_joints
    @property
    def state(self): return self.nq + self.nv
    @property
    def delta(self): return 12 + 2 * self.n_joints
    @property
    def xi(self): return 4 + self.n_joints
    @property
    def obs(self): return 6 + 3 * self.n_joints
    @property
    def goal_frame(self): return 10 + 2 * self.n_joints
    @property
    def obs_terms(self):
        n = self.n_joints
        return (("joint_pos", n), ("joint_vel", n), ("base_ang_vel", 3), ("projected_gravity", 3), ("actions", n))


@dataclass(frozen=True)
class RobotProfile:
    name: str
    n_joints: int
    root: str
    anchor: str
    tracked: tuple[str, ...]
    foot_regex: str

    @property
    def dims(self): return RobotDimensions(self.n_joints)

    def default_motion_dir(self) -> Path:
        if self.name == "g1":
            return DATA_ROOT / "npz/lafan1"
        for version in ("lafan1_modelscope_v1", "lafan1_official_v1"):
            path = DATA_ROOT / "retarget" / version / self.name
            if path.is_dir():
                return path
        return DATA_ROOT / "retarget/lafan1_modelscope_v1" / self.name

    def spec(self) -> mujoco.MjSpec:
        if self.name == "g1":
            from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML, get_spec
            spec = get_spec()
            spec.meshdir = str((G1_XML.parent / spec.meshdir).resolve())
            return spec
        path = ROOT / "assets/retarget" / f"{self.name}.xml"
        if not path.exists():
            raise FileNotFoundError(f"{path}: run bash scripts/prepare_lafan.sh first")
        tree = ET.parse(path).getroot()
        mesh_dir = (THIRD_PARTY / "unitree_ros/robots/r1_description/meshes" if self.name == "r1"
                    else THIRD_PARTY / "booster_assets/robots/K1/meshes")
        tree.find("compiler").set("meshdir", str(mesh_dir))
        # Training supplies a terrain and configured actuators. Keep the official
        # inertials and kinematic tree, including the fixed retarget markers.
        for tag in ("actuator", "keyframe", "sensor", "contact", "option"):
            for node in list(tree.findall(tag)):
                tree.remove(node)
        world = tree.find("worldbody")
        for node in list(world):
            if node.tag in ("geom", "light", "camera"):
                world.remove(node)
        for geom in tree.iter("geom"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            geom.set("group", "2")
        for joint in tree.iter("joint"):
            if joint.get("type") != "free":
                joint.set("damping", "0")
                joint.set("frictionloss", "0")
        spec = mujoco.MjSpec.from_string(ET.tostring(tree, encoding="unicode"))
        model = spec.compile()
        if model.nq != self.dims.nq or model.nv != self.dims.nv:
            raise ValueError(f"{self.name}: expected {self.n_joints} joints, got nq={model.nq}, nv={model.nv}")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        # Primitive, floor-only collision. Visual meshes do not enter the solver.
        # Segment endpoints use the official zero-pose tree, not guessed lengths.
        for side in ("left", "right"):
            ankle = spec.body(f"{side}_ankle_roll_link")
            pos = [0.04, 0, -0.025] if self.name == "r1" else [0.026, 0, -0.02]
            size = [0.08, 0.035, 0.015] if self.name == "r1" else [0.09, 0.035, 0.018]
            ankle.add_geom(name=f"{side}_foot_collision", type=mujoco.mjtGeom.mjGEOM_BOX,
                           pos=pos, size=size, contype=0, conaffinity=1, condim=3,
                           density=0, friction=[1, 0.005, 0.0001], group=3)
            knee = f"{side}_knee_link" if self.name == "r1" else f"{side}_knee_pitch_link"
            elbow = f"{side}_elbow_link" if self.name == "r1" else f"{side}_elbow_yaw_link"
            hand = f"{side}_wrist_roll_link" if self.name == "r1" else f"{side}_hand"
            for start, end, radius in ((f"{side}_hip_roll_link", knee, 0.04),
                                       (knee, f"{side}_ankle_roll_link", 0.035),
                                       (f"{side}_shoulder_roll_link", elbow, 0.03),
                                       (elbow, hand, 0.025)):
                a, b = model.body(start).id, model.body(end).id
                offset = data.xmat[a].reshape(3, 3).T @ (data.xpos[b] - data.xpos[a])
                if np.linalg.norm(offset) > 1e-4:
                    spec.body(start).add_geom(name=f"sm_{start}_collision", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                                               size=[radius, 0, 0], fromto=[0, 0, 0, *offset], density=0,
                                               contype=0, conaffinity=1, condim=1, group=3)
        for body in {self.root, self.anchor}:
            spec.body(body).add_geom(name=f"sm_{body}_collision", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                     size=[0.07, 0, 0], density=0, contype=0, conaffinity=1, condim=1, group=3)
        return spec

    def model(self):
        return self.spec().compile()

    def entity_cfg(self):
        if self.name == "g1":
            from mjlab.asset_zoo.robots import get_g1_robot_cfg
            return get_g1_robot_cfg()
        from mjlab.actuator import BuiltinPositionActuatorCfg
        from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
        model = self.model()
        official = (mujoco.MjModel.from_xml_path(str(THIRD_PARTY / "booster_assets/robots/K1/K1_22dof.xml"))
                    if self.name == "k1" else model)
        actuators = []
        defaults = {}
        for j in range(1, model.njnt):
            name = model.joint(j).name
            if self.name == "r1":
                kp, kd = ((100., 2.) if any(k in name for k in ("hip_", "knee", "waist")) else
                          (40., 2.) if any(k in name for k in ("ankle", "shoulder_pitch", "shoulder_roll")) else (20., 1.))
                effort = float(max(abs(model.jnt_actfrcrange[j])))
                armature = 0.01
            else:
                oj = official.joint(name).id
                armature = float(official.dof_armature[official.jnt_dofadr[oj]])
                kp = armature * (2 * math.pi * 10)**2
                kd = 4 * armature * (2 * math.pi * 10)
                effort = float(max(abs(official.actuator(name).forcerange)))
            if kp <= 0 or effort <= 0:
                raise ValueError(f"Missing actuator parameters: {self.name}/{name}")
            actuators.append(BuiltinPositionActuatorCfg(target_names_expr=(name,), stiffness=kp, damping=kd,
                                                        effort_limit=effort, armature=armature))
            value = (-0.1 if "hip_pitch" in name else 0.3 if "knee" in name else -0.2 if "ankle_pitch" in name else 0.)
            if self.name == "r1":
                if "shoulder_pitch" in name: value = 0.35
                if "elbow" in name: value = 0.87
                if "shoulder_roll" in name: value = 0.18 if name.startswith("left") else -0.18
            elif "shoulder_roll" in name:
                value = -1.2 if name.startswith("left") else 1.2
            defaults[name] = float(np.clip(value, *model.jnt_range[j]))
        data = mujoco.MjData(model)
        data.qpos[:] = model.qpos0
        for j in range(1, model.njnt):
            data.qpos[model.jnt_qposadr[j]] = defaults[model.joint(j).name]
        mujoco.mj_forward(model, data)
        bottom = min(data.geom_xpos[model.geom(f"{s}_foot_collision").id, 2] - model.geom(f"{s}_foot_collision").size[2]
                     for s in ("left", "right"))
        height = float(data.qpos[2] - bottom)
        return EntityCfg(spec_fn=self.spec, sort_actuators=True,
                         init_state=EntityCfg.InitialStateCfg(pos=(0., 0., height), joint_pos=defaults, joint_vel={".*": 0.}),
                         articulation=EntityArticulationInfoCfg(actuators=tuple(actuators), soft_joint_pos_limit_factor=0.9))

    def action_scale(self):
        if self.name == "g1":
            from mjlab.asset_zoo.robots import G1_ACTION_SCALE
            return G1_ACTION_SCALE
        return {a.target_names_expr[0]: 0.25 * a.effort_limit / a.stiffness for a in self.entity_cfg().articulation.actuators}


def get_robot(name: str = "g1") -> RobotProfile:
    if name == "g1":
        return RobotProfile(name, 29, "pelvis", "torso_link", G1_TRACKED, r"^(left|right)_foot[1-7]_collision$")
    if name == "r1":
        tracked = ("pelvis_link", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
                   "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "waist_yaw_link",
                   "left_shoulder_roll_link", "left_elbow_link", "left_wrist_roll_link",
                   "right_shoulder_roll_link", "right_elbow_link", "right_wrist_roll_link", "head_yaw_link")
        return RobotProfile(name, 26, "pelvis_link", "waist_yaw_link", tracked, r"^(left|right)_foot_collision$")
    if name == "k1":
        tracked = ("trunk", "left_hip_roll_link", "left_knee_pitch_link", "left_ankle_roll_link",
                   "right_hip_roll_link", "right_knee_pitch_link", "right_ankle_roll_link",
                   "left_shoulder_roll_link", "left_elbow_yaw_link", "left_hand",
                   "right_shoulder_roll_link", "right_elbow_yaw_link", "right_hand", "aahead_pitch_link")
        return RobotProfile(name, 22, "trunk", "trunk", tracked, r"^(left|right)_foot_collision$")
    raise ValueError(f"Unknown robot {name!r}; choose g1, r1 or k1")
