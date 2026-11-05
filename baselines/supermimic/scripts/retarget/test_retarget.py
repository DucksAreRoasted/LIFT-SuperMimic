"""Numerical regression checks for the standalone retargeting integration."""
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO.parents[1]
sys.path.insert(0, str(REPO / "src"))
from supermimic.data.motion_convert import G1_JOINT_NAMES, forward_kinematics, quat_rotate_inverse, resample_motion


class RetargetNumerics(unittest.TestCase):
    def test_source_order_and_official_target_limits(self):
        source = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / "third_party/robot_retargeter/asset/robot/g1_description/mjcf/g1.xml"))
        self.assertEqual(tuple(source.joint(j).name for j in range(1, source.njnt)), G1_JOINT_NAMES)
        r1 = mujoco.MjModel.from_xml_path(str(REPO / "assets/retarget/r1.xml"))
        urdf = ET.parse(PROJECT_ROOT / "third_party/unitree_ros/robots/r1_description/R1.urdf")
        joints = [j for j in urdf.findall("joint") if j.get("type") == "revolute"]
        self.assertEqual(len(joints), 26)
        self.assertEqual(r1.nq, 33)
        for joint in joints:
            limit = joint.find("limit")
            # MuJoCo's XML writer serializes URDF values to six significant figures.
            np.testing.assert_allclose(r1.joint(joint.get("name")).range,
                                       [float(limit.get("lower")), float(limit.get("upper"))], atol=1e-5)
        official = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / "third_party/booster_assets/robots/K1/K1_22dof.xml"))
        k1 = mujoco.MjModel.from_xml_path(str(REPO / "assets/retarget/k1.xml"))
        self.assertEqual(k1.nq, 29)
        for j in range(1, official.njnt):
            a, b = official.joint(j), k1.joint(official.joint(j).name)
            np.testing.assert_allclose(a.range, b.range)
            np.testing.assert_allclose(a.axis, b.axis)

    def test_slerp_and_rotating_root_velocities(self):
        times = np.arange(7) / 30
        raw = np.zeros((len(times), 8))
        raw[:, :3] = np.array([0.2, -0.3, 0.4]) * times[:, None]
        raw[:, 3:7] = Rotation.from_euler("z", 0.8 * times[:, None]).as_quat()
        raw[::2, 3:7] *= -1  # Quaternion sign must not induce a full rotation.
        motion = resample_motion(raw, 30, 50)
        self.assertEqual(motion.num_frames, 11)
        np.testing.assert_allclose(motion.root_lin_vel_w, np.broadcast_to([0.2, -0.3, 0.4], (11, 3)), atol=1e-12)
        # The reproduction converter uses normalized lerp for tiny quaternion angles.
        np.testing.assert_allclose(motion.root_ang_vel_w, np.broadcast_to([0, 0, 0.8], (11, 3)), atol=3e-6)
        np.testing.assert_allclose(np.linalg.norm(motion.root_quat, axis=1), 1, atol=1e-12)

    def test_exported_body_velocities_at_link_origins(self):
        model = mujoco.MjModel.from_xml_path(str(REPO / "assets/retarget/k1.xml"))
        raw = np.repeat(model.qpos0[None], 10, axis=0)
        times = np.arange(10) / 30
        raw[:, 0] += times * 0.3
        raw[:, 3:7] = (Rotation.from_euler("x", 0.4) * Rotation.from_euler("z", times[:, None] * 0.8)).as_quat()
        raw[:, 7:] = 0.05 * np.sin(times[:, None] * np.arange(1, model.nq - 6)[None])
        motion = resample_motion(raw, 30, 50)
        bodies = forward_kinematics(model, motion)
        data = mujoco.MjData(model)
        for f in (0, 5, 10):
            data.qpos[:] = np.r_[motion.root_pos[f], motion.root_quat[f], motion.joint_pos[f]]
            data.qvel[:] = np.r_[motion.root_lin_vel_w[f], quat_rotate_inverse(motion.root_quat[f], motion.root_ang_vel_w[f]), motion.joint_vel[f]]
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(bodies.pos_w[f], data.xpos[1:], atol=1e-12)
            for body in range(1, model.nbody):
                v = np.zeros(6)
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, v, 0)
                lin = v[3:] + np.cross(v[:3], data.xpos[body] - data.xipos[body])
                np.testing.assert_allclose(bodies.lin_vel_w[f, body - 1], lin, atol=1e-10)
                np.testing.assert_allclose(bodies.ang_vel_w[f, body - 1], v[:3], atol=1e-10)


if __name__ == "__main__":
    unittest.main()
