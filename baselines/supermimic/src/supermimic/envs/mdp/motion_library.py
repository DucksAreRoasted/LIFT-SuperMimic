"""Multi-clip motion library held as flat GPU tensors.

All clips are concatenated along a single global frame axis.  A per-frame
``clip_id`` and per-clip ``start`` / ``length`` tables let a batch of
environments index arbitrary clips and frames with plain gathers.

Two storage modes:

* **full** (small sets, e.g. LAFAN1): per frame we keep the reference joints, the pelvis
  state and the world-frame link states of the *tracked* bodies, plus precomputed 68-d
  goal features. ~1 KB/frame.
* **compact** (BONES-SEED, 47.5M frames): per frame only the 71-d reference state
  ``[root_pos(3), root_quat(4), joint_pos(29), root_lin_vel_w(3), root_ang_vel_w(3),
  joint_vel(29)]`` in fp16/fp32 (284 B/frame in fp32); tracked-body states and goal
  features are computed on the fly with the differentiable FK (plan 1.3).

Goal features (per frame, 68-d, plan D5):
  [joint_pos - q0 (29), joint_vel (29), projected gravity of the ref root (3),
   ref root angular velocity in the root frame (3),
   ref root linear velocity in the ref heading frame (3), ref root height (1)]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from supermimic.kinematics.rotations import projected_gravity, quat_apply_inverse, yaw_quat
from supermimic.robots import get_robot

GOAL_FEAT_DIM = 68
REF_DIM = 71


@dataclass
class ClipInfo:
    name: str
    start: int
    length: int

    @property
    def end(self) -> int:  # exclusive
        return self.start + self.length


@dataclass
class RefState:
    root_pos_w: torch.Tensor  # (N,3)
    root_quat_w: torch.Tensor  # (N,4) wxyz
    joint_pos: torch.Tensor  # (N,29)
    root_lin_vel_w: torch.Tensor  # (N,3)
    root_ang_vel_w: torch.Tensor  # (N,3)
    joint_vel: torch.Tensor  # (N,29)


@dataclass
class RefBodies:
    pos_w: torch.Tensor  # (N,B,3)
    quat_w: torch.Tensor  # (N,B,4)
    lin_vel_w: torch.Tensor  # (N,B,3)
    ang_vel_w: torch.Tensor  # (N,B,3)


def goal_features(ref: RefState, default_joint_pos: torch.Tensor) -> torch.Tensor:
    q = ref.root_quat_w
    grav = projected_gravity(q)
    ang_b = quat_apply_inverse(q, ref.root_ang_vel_w)
    lin_h = quat_apply_inverse(yaw_quat(q), ref.root_lin_vel_w)
    height = ref.root_pos_w[..., 2:3]
    return torch.cat([ref.joint_pos - default_joint_pos, ref.joint_vel, grav, ang_b, lin_h, height], dim=-1)


class MotionLibrary:
    def __init__(
        self,
        npz_files: list[str | Path],
        body_names: tuple[str, ...],
        default_joint_pos: torch.Tensor | np.ndarray,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
        bin_frames: int = 50,
        compact: bool = False,
        store_dtype: torch.dtype = torch.float16,
        robot_name: str = "g1",
        joint_names: tuple[str, ...] | None = None,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.body_names = tuple(body_names)
        self.bin_frames = int(bin_frames)
        self.compact = compact
        self.profile = get_robot(robot_name)
        self.dims = self.profile.dims
        self.goal_frame_dim = self.dims.goal_frame
        if joint_names is None:
            model = self.profile.model()
            joint_names = tuple(model.joint(j).name for j in range(1, model.njnt))
        expected_joint_names = tuple(joint_names)
        if len(npz_files) == 0:
            raise ValueError("MotionLibrary needs at least one clip")
        if isinstance(default_joint_pos, torch.Tensor):
            q0 = default_joint_pos.detach().to(device=self.device, dtype=dtype).reshape(-1)
        else:
            q0 = torch.as_tensor(np.asarray(default_joint_pos), dtype=dtype, device=self.device).reshape(-1)
        self.default_joint_pos = q0
        if q0.numel() != self.dims.n_joints:
            raise ValueError(f"{robot_name}: default joint count mismatch: {q0.numel()}")

        clips: list[ClipInfo] = []
        start = 0
        body_idx = root_idx = None
        ref_chunks: list[np.ndarray] = []
        body_chunks = {"pos": [], "quat": [], "lin": [], "ang": []}
        # A single packed cache (build_library.py) can stand in for the npz list (compact mode only).
        if len(npz_files) == 1 and str(npz_files[0]).endswith(".pt"):
            assert compact, "packed .pt libraries are compact-only"
            pack = torch.load(npz_files[0], map_location="cpu", weights_only=False)
            if pack.get("robot", "g1") != robot_name:
                raise ValueError(f"Packed library robot does not match {robot_name}")
            if "joint_names" in pack and tuple(pack["joint_names"]) != expected_joint_names:
                raise ValueError("Packed library joint order mismatch")
            self._ref = pack["ref"].to(device=self.device, dtype=store_dtype)
            clips = [ClipInfo(n, int(st), int(ln)) for n, st, ln in zip(pack["names"], pack["starts"], pack["lengths"])]
            start = int(self._ref.shape[0])
            ref_chunks = None
        for f in (npz_files if ref_chunks is not None else []):
            with np.load(f) as d:
                names = [str(n) for n in d["body_names"]]
                body_idx = np.array([names.index(n) for n in self.body_names])
                root_idx = names.index(self.profile.root)
                if "joint_names" not in d or tuple(d["joint_names"].tolist()) != expected_joint_names:
                    raise ValueError(f"{f}: joint names/order do not match {robot_name}")
                if not np.isclose(float(np.asarray(d["fps"]).reshape(-1)[0]), 50.):
                    raise ValueError(f"{f}: training motions must be 50 Hz")
                if d["joint_pos"].shape[1] != self.dims.n_joints:
                    raise ValueError(f"{f}: joint width does not match {robot_name}")
                n = int(d["joint_pos"].shape[0])
                bp, bq, blv, bav = d["body_pos_w"], d["body_quat_w"], d["body_lin_vel_w"], d["body_ang_vel_w"]
                ref = np.concatenate(
                    [bp[:, root_idx], bq[:, root_idx], d["joint_pos"], blv[:, root_idx], bav[:, root_idx], d["joint_vel"]], axis=1
                ).astype(np.float32)
                ref_chunks.append(ref)
                if not compact:
                    body_chunks["pos"].append(bp[:, body_idx])
                    body_chunks["quat"].append(bq[:, body_idx])
                    body_chunks["lin"].append(blv[:, body_idx])
                    body_chunks["ang"].append(bav[:, body_idx])
            clips.append(ClipInfo(Path(f).stem, start, n))
            start += n
        if ref_chunks is None and self._ref.shape[1] != self.dims.state:
            raise ValueError(f"Packed library state width does not match {robot_name}")
        self.clips = clips
        self.num_clips = len(clips)
        self.num_frames = start

        if ref_chunks is not None:
            ref_np = np.concatenate(ref_chunks, 0)
            assert ref_np.shape[1] == self.dims.state
            self._ref = torch.as_tensor(ref_np, device=self.device, dtype=store_dtype if compact else dtype)
            del ref_np, ref_chunks
        if not compact:
            t = lambda arrs: torch.as_tensor(np.concatenate(arrs, 0), dtype=dtype, device=self.device)  # noqa: E731
            self.body_pos_w = t(body_chunks["pos"])
            self.body_quat_w = t(body_chunks["quat"])
            self.body_lin_vel_w = t(body_chunks["lin"])
            self.body_ang_vel_w = t(body_chunks["ang"])
            del body_chunks
            self.goal_feat = goal_features(self._split(self._ref), q0)
        else:
            from supermimic.kinematics.g1_fk import G1Kinematics

            self._fk = G1Kinematics(self.profile.model(), device=self.device, dtype=dtype)
            self._tracked_idx = self._fk.body_indices(self.body_names)

        self.clip_start = torch.tensor([c.start for c in clips], device=self.device, dtype=torch.long)
        self.clip_len = torch.tensor([c.length for c in clips], device=self.device, dtype=torch.long)
        self.clip_last = self.clip_start + self.clip_len - 1
        self.frame_clip = torch.repeat_interleave(torch.arange(self.num_clips, device=self.device), self.clip_len)
        # Bins of `bin_frames` frames inside each clip (plan D14), numbered globally.
        bins_per_clip = (self.clip_len + self.bin_frames - 1) // self.bin_frames
        self.clip_bin_offset = torch.cumsum(bins_per_clip, 0) - bins_per_clip
        self.num_bins = int(bins_per_clip.sum().item())
        bin_clip = torch.repeat_interleave(torch.arange(self.num_clips, device=self.device), bins_per_clip)
        local_bin = torch.arange(self.num_bins, device=self.device) - self.clip_bin_offset[bin_clip]
        self.bin_start = self.clip_start[bin_clip] + local_bin * self.bin_frames
        self.bin_end = torch.minimum(self.bin_start + self.bin_frames, self.clip_start[bin_clip] + self.clip_len[bin_clip])
        self.bin_clip = bin_clip

    # ------------------------------------------------------------------ per-frame access

    @staticmethod
    def _split(r: torch.Tensor) -> RefState:
        nj = (r.shape[-1] - 13) // 2
        nq = 7 + nj
        return RefState(r[..., :3], r[..., 3:7], r[..., 7:nq], r[..., nq:nq+3], r[..., nq+3:nq+6], r[..., nq+6:])

    def ref(self, frame: torch.Tensor) -> RefState:
        return self._split(self._ref[frame].to(self.dtype))

    def ref71(self, frame: torch.Tensor) -> torch.Tensor:
        return self._ref[frame].to(self.dtype)

    def bodies(self, frame: torch.Tensor) -> RefBodies:
        """Tracked-body link states (world) at the given frames."""
        if not self.compact:
            return RefBodies(self.body_pos_w[frame], self.body_quat_w[frame], self.body_lin_vel_w[frame], self.body_ang_vel_w[frame])
        qpos, qvel = self.ref_qpos_qvel(frame)
        with torch.no_grad():
            out = self._fk(qpos, qvel)
        i = self._tracked_idx
        return RefBodies(out["pos"][..., i, :], out["quat"][..., i, :], out["lin_vel"][..., i, :], out["ang_vel"][..., i, :])

    # Convenience column accessors (full fp32 copies; prefer ``ref()`` in hot paths).
    @property
    def root_pos_w(self) -> torch.Tensor:
        return self._ref[:, 0:3].to(self.dtype)

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self._ref[:, 3:7].to(self.dtype)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._ref[:, 7:self.dims.nq].to(self.dtype)

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        return self._ref[:, self.dims.nq:self.dims.nq+3].to(self.dtype)

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        return self._ref[:, self.dims.nq+3:self.dims.nq+6].to(self.dtype)

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._ref[:, self.dims.nq+6:].to(self.dtype)

    # ------------------------------------------------------------------ queries

    def bin_of_frame(self, frame: torch.Tensor) -> torch.Tensor:
        clip = self.frame_clip[frame]
        return self.clip_bin_offset[clip] + (frame - self.clip_start[clip]) // self.bin_frames

    def last_frame_of(self, frame: torch.Tensor) -> torch.Tensor:
        return self.clip_last[self.frame_clip[frame]]

    def goal_window(self, frame: torch.Tensor, k: int) -> torch.Tensor:
        """(N, k*68) goal features of frames frame..frame+k-1, clamped at the clip end."""
        last = self.last_frame_of(frame)
        offs = torch.arange(k, device=self.device)
        idx = torch.minimum(frame[:, None] + offs[None, :], last[:, None])  # (N,k)
        if not self.compact:
            return self.goal_feat[idx].reshape(frame.shape[0], k * self.goal_frame_dim)
        feat = goal_features(self.ref(idx.reshape(-1)), self.default_joint_pos)
        return feat.reshape(frame.shape[0], k * self.goal_frame_dim)

    def ref_qpos_qvel(self, frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """MuJoCo-layout reference state: qpos (N,36) and qvel (N,35; root ang vel in body frame)."""
        r = self.ref(frame)
        qpos = torch.cat([r.root_pos_w, r.root_quat_w, r.joint_pos], dim=-1)
        ang_b = quat_apply_inverse(r.root_quat_w, r.root_ang_vel_w)
        qvel = torch.cat([r.root_lin_vel_w, ang_b, r.joint_vel], dim=-1)
        return qpos, qvel

    def summary(self) -> str:
        hours = self.num_frames / 50.0 / 3600.0
        mem = self._ref.numel() * self._ref.element_size() / 1e9
        if not self.compact:
            mem += sum(t.numel() * t.element_size() for t in (self.body_pos_w, self.body_quat_w, self.body_lin_vel_w, self.body_ang_vel_w, self.goal_feat)) / 1e9
        return f"MotionLibrary({'compact' if self.compact else 'full'}): {self.num_clips} clips, {self.num_frames} frames ({hours:.3f} h @50Hz), {self.num_bins} bins, {mem:.2f} GB on {self.device}"


def discover_npz(directory: str | Path, limit: int | None = None, names: list[str] | None = None, manifest: str | Path | None = None) -> list[Path]:
    """List npz clips; ``manifest`` (csv with name,keep columns) restricts to keep==1."""
    directory = Path(directory)
    files = sorted(directory.glob("*.npz"))
    if manifest is not None:
        import csv

        with open(manifest) as f:
            keep = {row["name"] for row in csv.DictReader(f) if row.get("keep", "1") in ("1", "True", "true")}
        files = [f for f in files if f.stem in keep]
    if names is not None:
        wanted = set(names)
        files = [f for f in files if f.stem in wanted]
    if limit is not None:
        files = files[:limit]
    return files
