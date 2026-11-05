"""Adapt the existing replay and analytic observations to goal-conditioned SAC."""

import torch

from supermimic.data.replay_buffer import StartBatch
from supermimic.world_model.heads import (
    analytic_observation, anchor_relative_obs, flatten_history_term_major,
)
from supermimic.world_model.wm_trainer import reference_at

from tadg.sac import SACBatch


class ModelInputs:
    def __init__(self, wm, policy, library, cfg):
        self.wm, self.policy, self.lib, self.cfg = wm, policy, library, cfg
        self.q0 = library.default_joint_pos
        nj = self.q0.numel()
        self.noise_half = self.q0.new_tensor(
            [0.01] * nj + [0.5] * nj + [0.2] * 3 + [0.05] * 3 + [0.0] * nj)

    def add_noise(self, obs):
        if not self.cfg.obs_noise:
            return obs
        return obs + (2 * torch.rand_like(obs) - 1) * self.noise_half

    def history(self, start):
        return [self.add_noise(analytic_observation(s, a, start.xi, self.q0))
                for s, a in zip(start.hist_states, start.hist_actions)]

    def at(self, state, xi, frame, history):
        ref = reference_at(self.lib, frame)
        obs = flatten_history_term_major(torch.stack(history))
        goal = self.lib.goal_window(frame, self.cfg.future_frames)
        if self.cfg.anchor_obs:
            bodies = self.wm.heads.robot_bodies(state)
            ai = self.wm.heads.anchor_index
            rel = anchor_relative_obs(bodies["pos"][..., ai, :], bodies["quat"][..., ai, :],
                                      ref.body_pos_w[..., ai, :], ref.body_quat_w[..., ai, :])
            if self.cfg.obs_noise:
                rel = rel + (2 * torch.rand_like(rel) - 1) * rel.new_tensor([0.25] * 3 + [0.05] * 6)
            goal = torch.cat((goal, rel), dim=-1)
        mean_policy = self.policy.mean_policy
        features = torch.cat((self.wm.spec.canonical(state),
                              goal * mean_policy.encoders["motion"].scale,
                              self.wm.spec.normalize_xi(xi),
                              self.wm.heads.root_error_features(state, ref),
                              obs * mean_policy.decoder.scale), dim=-1)
        return obs, goal, features, ref


def _start_at(buffer, env, time, history):
    n = env.numel()
    offset = torch.arange(history, device=time.device) - history + 1
    times = (time[None] + offset[:, None]) % buffer.T
    envs = env[None].expand(history, n)
    episode = buffer.ep_id[env, time]
    same = buffer.valid[envs, times] & (buffer.ep_id[envs, times] == episode[None])
    first = same.int().argmax(0)
    first_time = times[first, torch.arange(n, device=time.device)]
    times = torch.where(same, times, first_time[None])
    previous = (times - 1) % buffer.T
    valid_previous = (buffer.valid[envs, previous]
                      & (buffer.ep_id[envs, previous] == episode[None])
                      & ~buffer.ep_start[envs, times])
    actions = torch.where(valid_previous[..., None], buffer.a[envs, previous],
                          torch.zeros_like(buffer.a[envs, previous]))
    return StartBatch(buffer.s[env, time], buffer.xi[env, time], buffer.frame[env, time],
                      actions[-1], buffer.s[envs, times], actions)


@torch.no_grad()
def sample_real_batch(buffer, inputs, n, history):
    mask = buffer.window_mask(1)
    t_all = torch.arange(buffer.T, device=buffer.device)[None]
    age = (buffer.ptr[:, None] - t_all - 1) % buffer.T
    mask &= age < buffer.T - history - 1
    env, time = buffer._sample_from_mask(mask, n)
    next_time = (time + 1) % buffer.T
    start = _start_at(buffer, env, time, history)
    following = _start_at(buffer, env, next_time, history)
    # The terminal row still belongs to the old episode. Its frame bookkeeping can be stale;
    # reconstruct reference advancement from the transition's own frame.
    following.frame = torch.minimum(start.frame + 1, inputs.lib.last_frame_of(start.frame))
    following.xi = start.xi
    obs, goal, features, _ = inputs.at(start.state, start.xi, start.frame, inputs.history(start))
    obs_next, goal_next, features_next, _ = inputs.at(
        following.state, following.xi, following.frame, inputs.history(following))
    # Time limits retain bootstrap; true falls and the end of the reference task do not.
    bootstrap = (1.0 - buffer.done[env, next_time]) * (
        start.frame < inputs.lib.last_frame_of(start.frame)).to(buffer.r.dtype)
    batch = SACBatch(features, obs, goal, buffer.a_exec[env, time], buffer.r[env, time], bootstrap,
                     features_next, obs_next, goal_next)
    return batch, start
