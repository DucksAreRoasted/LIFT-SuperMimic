"""Robot-specific FK, data, learning dimensions and real CUDA training smoke."""
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from supermimic.robots import get_robot
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg

BASELINE_ROOT = Path(__file__).resolve().parents[1]
ROOT = BASELINE_ROOT.parents[1]


@pytest.mark.parametrize("name", ["g1", "r1", "k1"])
def test_robot_fk_and_dimensions(name):
    p = get_robot(name)
    model = p.model()
    dims = p.dims
    assert model.nq == dims.nq and model.nv == dims.nv
    fk = G1Kinematics(model, dtype=torch.float64)
    rng = np.random.default_rng(13)
    qp = np.repeat(model.qpos0[None], 3, axis=0)
    qp[:, 3:7] = rng.normal(size=(3, 4))
    qp[:, 3:7] /= np.linalg.norm(qp[:, 3:7], axis=1, keepdims=True)
    qp[:, 7:] = rng.uniform(-0.3, 0.3, (3, p.n_joints))
    qv = rng.normal(size=(3, dims.nv))
    out = fk(torch.tensor(qp), torch.tensor(qv))
    for frame in range(3):
        data = mujoco.MjData(model)
        data.qpos[:], data.qvel[:] = qp[frame], qv[frame]
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(out["pos"][frame], data.xpos[1:], atol=1e-10)
        for body in range(1, model.nbody):
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, vel, 0)
            v_link = vel[3:] + np.cross(vel[:3], data.xpos[body] - data.xipos[body])
            np.testing.assert_allclose(out["lin_vel"][frame, body-1], v_link, atol=1e-10)
    spec = StateSpec(StateSpecCfg(n_joints=p.n_joints, explicit_integration=False), "cpu", torch.float64)
    states = torch.tensor(np.concatenate((qp, qv), axis=1))
    np.testing.assert_allclose(spec.integrate(states, torch.zeros(3, dims.delta)).numpy(), states, atol=1e-12)
    assert spec.canonical(states).shape == (3, dims.state)
    assert spec.xi_scale.numel() == dims.xi


@pytest.mark.parametrize("name", ["r1", "k1"])
def test_motion_library_and_learning_graph(name):
    from supermimic.envs.mdp.motion_library import MotionLibrary
    from supermimic.policy.nets import PolicyCfg, Policy, Critic, CriticCfg
    from supermimic.world_model.heads import AnalyticHeads, analytic_observation, flatten_history_term_major
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg
    from supermimic.world_model.wm_trainer import reference_at
    p = get_robot(name)
    d = p.dims
    model = p.model()
    path = ROOT / "data/retarget/modelscope_server_smoke" / name / "walk1_subject1.npz"
    if not path.exists():
        path = ROOT / "data/retarget/lafan1_official_v1" / name / "walk1_subject1.npz"
    if not path.exists():
        pytest.skip("Run scripts/prepare_lafan.sh to prepare a reference clip")
    q0 = torch.zeros(p.n_joints)
    lib = MotionLibrary([path], p.tracked, q0, "cpu", robot_name=name)
    compact = MotionLibrary([path], p.tracked, q0, "cpu", compact=True, store_dtype=torch.float32, robot_name=name)
    frames = torch.tensor([0, 1, 2, 3])
    np.testing.assert_allclose(compact.bodies(frames).pos_w, lib.bodies(frames).pos_w, atol=2e-6)
    np.testing.assert_allclose(compact.goal_window(frames, 3), lib.goal_window(frames, 3), atol=1e-6)
    qp, qv = lib.ref_qpos_qvel(frames)
    state = torch.cat((qp, qv), dim=-1)
    limits = torch.tensor(model.jnt_range[1:], dtype=torch.float32)
    heads = AnalyticHeads(G1Kinematics(model), p.tracked, p.anchor, p.root, limits, q0)
    wm = WorldModel(WorldModelCfg(hidden=(32, 32), state_spec=StateSpecCfg(n_joints=p.n_joints)), heads, "cpu")
    pc = PolicyCfg(n_joints=p.n_joints, history=3, goal_dim=3*d.goal_frame, encoder_hidden=(32,), decoder_hidden=(32,))
    policy = Policy(pc)
    critic = Critic(CriticCfg(hidden=(16,)), pcfg=pc)
    xi = torch.zeros(4, d.xi); xi[:, 0] = 1.
    prev = torch.zeros(4, p.n_joints)
    obs = analytic_observation(state, prev, xi, q0)
    hist = flatten_history_term_major(obs[None].expand(3, -1, -1))
    goal = lib.goal_window(frames, 3)
    action = policy(hist, goal)
    nxt = wm.step(state, action, xi)
    ref = reference_at(lib, frames)
    reward = heads.reward(nxt, action, prev, ref)["total"]
    value = critic(wm.spec.canonical(nxt), goal, wm.spec.normalize_xi(xi), heads.root_error_features(nxt, ref))
    loss = -reward.mean() + value.square().mean()
    loss.backward()
    assert action.shape == (4, p.n_joints) and obs.shape == (4, d.obs)
    grads = [x.grad for x in policy.parameters() if x.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0
    # Joint count alone is insufficient: reject a reordered SDK/joint vector.
    with pytest.raises(ValueError, match="joint names/order"):
        MotionLibrary([path], p.tracked, q0, "cpu", robot_name=name,
                      joint_names=tuple(reversed([model.joint(j).name for j in range(1, model.njnt)])))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("name", ["r1", "k1"])
@pytest.mark.parametrize("compile_mode", [None, "default"], ids=["eager", "compiled"])
def test_cuda_training_and_resume(name, compile_mode, tmp_path):
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    p = get_robot(name)
    directory = ROOT / "data/retarget/modelscope_server_smoke" / name
    if not directory.exists():
        directory = p.default_motion_dir()
    if not directory.exists():
        pytest.skip("Prepare retargeted motions first")
    cfg = TrainConfig(robot=name, run_name=name, log_root=tmp_path)
    cfg.data.npz_dir, cfg.data.clips = directory, ["walk1_subject1"]
    cfg.env.num_envs = 4
    cfg.env.replay_capacity = 100
    cfg.env.sampling_mode = "start"
    cfg.env.obs_noise = cfg.env.reset_noise = cfg.env.domain_randomization = bool(compile_mode)
    cfg.world_model.hidden = (32, 32)
    cfg.world_model.term_hidden = (16,)
    cfg.policy.encoder_hidden = cfg.policy.decoder_hidden = (32,)
    cfg.critic.hidden = (16,)
    cfg.wm_trainer.batch_size = cfg.policy_trainer.batch_size = 8
    cfg.wm_trainer.batches_per_update = 1
    cfg.wm_trainer.window = 2
    cfg.wm_trainer.compile = cfg.policy_trainer.compile = compile_mode
    cfg.policy_trainer.horizon = 2
    cfg.policy_trainer.critic_epochs = 1
    cfg.policy_trainer.critic_minibatches = 1
    cfg.schedule.total_iterations = 25
    cfg.schedule.warmup_iterations = 12
    cfg.schedule.log_interval = 25
    cfg.schedule.checkpoint_interval = 25
    trainer = Trainer(cfg)
    try:
        obs, _ = trainer.env.reset()
        assert obs["policy"].shape[-1] == p.dims.obs * 10
        assert obs["goal"].shape[-1] == p.dims.goal_frame * 10
        state = trainer.recorder.state_io.read()
        predicted = trainer.wm.heads.robot_bodies(state)["pos"]
        actual = trainer.command.robot_body_pos_w - trainer.env.scene.env_origins[:, None, :]
        torch.testing.assert_close(predicted, actual, atol=2e-5, rtol=2e-5)
        before = [x.detach().clone() for x in trainer.policy.parameters()]
        trainer.run()
        assert trainer.num_updates > 0
        assert any(not torch.equal(a, b) for a, b in zip(before, trainer.policy.parameters()))
        checkpoint = tmp_path / name / "ckpt_latest.pt"
        policy, saved = load_policy_from_checkpoint(checkpoint, "cpu")
        assert saved["robot"] == name and policy.cfg.n_joints == p.n_joints
        assert all(torch.isfinite(v).all() for v in policy.parameters())
        trainer.load(checkpoint)
        assert trainer.iteration == 25
        # Checkpoint identity is checked before loading differently sized weights.
        trainer.cfg = replace(cfg, robot="g1")
        with pytest.raises(ValueError, match="Checkpoint robot"):
            trainer.load(checkpoint)
    finally:
        trainer.env.close()
