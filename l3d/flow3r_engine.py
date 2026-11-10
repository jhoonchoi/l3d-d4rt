"""Point queries against flow3r (Cong et al., CVPR 2026), with the same interface as ``l3d.engine.QueryEngine``.

flow3r is a dense multi-view model (fine-tuned from Pi3). One forward pass over a clip gives, per frame ``t``, a
local point map ``L_t[h, w] = (x, y, z)`` in camera ``t``, a confidence logit map and a camera-to-world pose
``T_t``. Its flow head maps a frame pair ``(i, j)`` to the NDC position, in frame ``j``, of every pixel of
frame ``i``. A point query ``(u, v, t_s, t_t, t_c)`` is answered by composition:

1. ``(u', v') = flow(t_s -> t_t)(u, v)`` (the query pixel itself when ``t_s == t_t``);
2. ``X = L_{t_t}(u', v')`` (bilinear);
3. ``P = T_{t_c}^{-1} T_{t_t} X``.

Consequences that the Part A analyses should expect (they are properties of the architecture, not bugs):

- the camera frame enters only through the predicted poses, so moving a query between cameras is an exact rigid
  transform: the camera-swap residual and the non-rigid camera-frame error are zero up to float error, and
  C-type error is pose error;
- ``L_{t_t}`` does not depend on ``t_s``, so source swap with GT ``u2`` is zero and all S-type error comes from
  the flow;
- flow3r has no visibility output. ``visibility`` is a label-free proxy: +10 if ``(u', v')`` is inside the
  image, -10 otherwise.

Pixels follow the Part A convention (``u = x / (W - 1)``, ``v = y / (H - 1)``, see ``docs/l3d/EXPERIMENTS.md``);
frames are resized (no crop) with flow3r's own rule (at most ``pixel_limit`` pixels, sides multiples of 14).
"""

from __future__ import annotations

import contextlib
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

PATCH = 14
PIXEL_LIMIT = 255_000
VIS_LOGIT = 10.0
FLOW3R_COMMIT = "6366c77"  # Kidrauh/flow3r main, 2026-02-27


def flow3r_size(height: int, width: int, pixel_limit: int = PIXEL_LIMIT) -> tuple[int, int]:
    """flow3r's resize rule (``flow3r.utils.basic.load_images_as_tensor``): returns ``(H, W)``, multiples of 14."""
    scale = math.sqrt(pixel_limit / (width * height)) if width * height > 0 else 1.0
    w_t, h_t = width * scale, height * scale
    k, m = round(w_t / PATCH), round(h_t / PATCH)
    while (k * PATCH) * (m * PATCH) > pixel_limit:
        if k / m > w_t / h_t:
            k -= 1
        else:
            m -= 1
    return max(1, m) * PATCH, max(1, k) * PATCH


def ndc_to_unit(ndc: torch.Tensor) -> torch.Tensor:
    """flow3r's PyTorch3D-style NDC (``x_ndc = 1 - 2 x / (W - 1)``, same for y) to ``[0, 1]`` coordinates."""
    return (1.0 - ndc) * 0.5


def bilinear(maps: torch.Tensor, k: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Sample ``maps[k]`` (``maps: [K, H, W, C]``) at unit coordinates ``(u, v)`` (corners at 0 and 1), clamped."""
    _, H, W, _ = maps.shape
    x = (u.clamp(0.0, 1.0) * (W - 1)).float()
    y = (v.clamp(0.0, 1.0) * (H - 1)).float()
    x0 = x.floor().long().clamp(0, W - 1)
    y0 = y.floor().long().clamp(0, H - 1)
    x1 = (x0 + 1).clamp(max=W - 1)
    y1 = (y0 + 1).clamp(max=H - 1)
    wx = (x - x0.float()).unsqueeze(-1)
    wy = (y - y0.float()).unsqueeze(-1)
    m = maps.float()
    top = m[k, y0, x0] * (1 - wx) + m[k, y0, x1] * wx
    bot = m[k, y1, x0] * (1 - wx) + m[k, y1, x1] * wx
    return top * (1 - wy) + bot * wy


def import_flow3r(flow3r_root: str | Path) -> Any:
    """Import the ``Flow3r`` class from a clone of https://github.com/Kidrauh/flow3r (pinned in setup_flow3r.sh)."""
    root = str(Path(flow3r_root).resolve())
    if not (Path(root) / "flow3r" / "models" / "flow3r.py").exists():
        raise FileNotFoundError(f"No flow3r clone at {root}; run scripts/l3d/setup_flow3r.sh")
    if root not in sys.path:
        sys.path.insert(0, root)
    from flow3r.models.flow3r import Flow3r  # noqa: PLC0415

    return Flow3r


class Flow3rEngine:
    """Encode a clip once with flow3r, then answer arbitrary point queries (see the module docstring)."""

    def __init__(
        self,
        model: torch.nn.Module | None,
        device: str | torch.device = "cpu",
        amp: str | None = None,
        chunk_size: int = 4096,
        pair_chunk: int = 8,
        pixel_limit: int = PIXEL_LIMIT,
        clip_frames: int = 64,
        model_config: str = "flow3r",
        load_info: dict[str, Any] | None = None,
    ) -> None:
        self.model = model.eval() if model is not None else None
        self.device = torch.device(device)
        self.amp = amp
        self.chunk_size = int(chunk_size)
        self.pair_chunk = int(pair_chunk)
        self.pixel_limit = int(pixel_limit)
        self.clip_frames = int(clip_frames)
        self.model_config = model_config
        self.load_info = load_info or {}
        self.image_hw: tuple[int, int] | None = None
        self.last_encode_seconds = float("nan")
        self.last_decode_seconds = float("nan")
        self._cache: dict[str, Any] | None = None

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str | Path,
        flow3r_root: str | Path = "third_party/flow3r",
        device: str = "auto",
        amp: str | None = "bf16",
        chunk_size: int = 4096,
        pair_chunk: int = 8,
    ) -> "Flow3rEngine":
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        Flow3r = import_flow3r(flow3r_root)
        model = Flow3r()
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        result = model.load_state_dict(state, strict=True)
        model = model.to(device)
        info = {"ckpt": str(ckpt_path), "flow3r_commit": FLOW3R_COMMIT, "strict": True, "result": str(result)}
        return cls(model, device=device, amp=amp, chunk_size=chunk_size, pair_chunk=pair_chunk,
                   model_config=f"flow3r@{FLOW3R_COMMIT}", load_info=info)

    # ------------------------------------------------------------------ helpers
    def _autocast(self) -> contextlib.AbstractContextManager:
        if self.device.type != "cuda" or not self.amp:
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=torch.bfloat16 if self.amp == "bf16" else torch.float16)

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @property
    def num_frames(self) -> int:
        if self._cache is None:
            raise RuntimeError("Call encode() first.")
        return int(self._cache["local"].shape[0])

    def clear(self) -> None:
        self._cache = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _resize(self, video_uint8: np.ndarray) -> torch.Tensor:
        H, W = video_uint8.shape[1:3]
        h, w = flow3r_size(H, W, self.pixel_limit)
        frames = [np.asarray(Image.fromarray(f).resize((w, h), Image.Resampling.LANCZOS)) for f in video_uint8]
        x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
        return x[None].to(self.device)  # [1, T, 3, h, w]

    def _forward(self, imgs: torch.Tensor) -> dict[str, Any]:
        """flow3r's ``forward`` without the flow call, keeping the intermediates the flow head needs."""
        m = self.model
        B, N, _, H, W = imgs.shape
        x = (imgs - m.image_mean) / m.image_std
        hidden = m.encoder(x.reshape(B * N, 3, H, W), is_training=True)
        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]
        hidden, pos = m.decode(hidden, N, H, W)
        point_hidden, point_inter = m.point_decoder(hidden, xpos=pos, return_intermediate=True)
        conf_hidden = m.conf_decoder(hidden, xpos=pos)
        camera_hidden, camera_inter = m.camera_decoder(hidden, xpos=pos, return_intermediate=True)
        ph, pw = H // PATCH, W // PATCH
        with torch.autocast(device_type=self.device.type, enabled=False):
            ret = m.point_head([point_hidden[:, m.patch_start_idx:].float()], (H, W)).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local = torch.cat([xy * z, z], dim=-1)
            conf = m.conf_head([conf_hidden[:, m.patch_start_idx:].float()], (H, W)).reshape(B, N, H, W)
            poses = m.camera_head(camera_hidden[:, m.patch_start_idx:].float(), ph, pw).reshape(B, N, 4, 4)
        return {
            "local": local[0].float(),
            "conf": conf[0].float(),
            "poses": poses[0].double(),
            "point_inter": [t.float() for t in point_inter],
            "camera_inter": [t.float() for t in camera_inter],
            "hw": (H, W),
        }

    def _flow_maps(self, pairs: torch.Tensor) -> torch.Tensor:
        """Flow head for ``pairs [P, 2]`` (source, target): NDC target coordinates ``[P, H, W, 2]``."""
        c = self._cache
        H, W = c["hw"]
        with torch.autocast(device_type=self.device.type, enabled=False):
            out = self.model.flow_head(c["point_inter"], c["camera_inter"], pairs[None], self.model.patch_start_idx,
                                       (H, W), 1, self.num_frames)
        return out[0].float()

    # ------------------------------------------------------------------ public API
    def encode(
        self,
        video_uint8: np.ndarray,
        aspect_wh: float | None = None,
        save_memory_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run flow3r once on ``video_uint8 [T, H, W, 3]`` and cache its outputs. ``aspect_wh`` is unused (no
        aspect token) and ``save_memory_path`` is not supported (the cache is ~1 GB per clip)."""
        del aspect_wh
        if save_memory_path is not None:
            raise NotImplementedError("Flow3rEngine does not save memory; drop --save-memory.")
        video_uint8 = np.asarray(video_uint8)
        if video_uint8.ndim != 4 or video_uint8.shape[-1] != 3:
            raise ValueError(f"Expected video [T, H, W, 3], got {video_uint8.shape}")
        start = time.perf_counter()
        imgs = self._resize(video_uint8)
        with torch.no_grad(), self._autocast():
            self._cache = self._forward(imgs)
        self._sync()
        self.image_hw = self._cache["hw"]
        self.last_encode_seconds = time.perf_counter() - start
        return self._cache

    def decode(
        self,
        u: np.ndarray,
        v: np.ndarray,
        t_src: np.ndarray,
        t_tgt: np.ndarray,
        t_cam: np.ndarray,
        chunk_size: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Answer ``M`` queries. Returns float32 ``xyz_3d [M, 3]``, ``uv_2d [M, 2]``, ``visibility [M]`` (proxy
        logit) and ``confidence [M]`` (logit of the point map at the flowed pixel)."""
        del chunk_size
        c = self._cache
        if c is None:
            raise RuntimeError("Call encode() first.")
        T = self.num_frames
        u_t = torch.as_tensor(np.asarray(u, dtype=np.float32).reshape(-1), device=self.device)
        v_t = torch.as_tensor(np.asarray(v, dtype=np.float32).reshape(-1), device=self.device)
        ts, tt, tc = (torch.as_tensor(np.asarray(a).reshape(-1).astype(np.int64), device=self.device)
                      for a in (t_src, t_tgt, t_cam))
        M = u_t.shape[0]
        if not (v_t.shape[0] == ts.shape[0] == tt.shape[0] == tc.shape[0] == M):
            raise ValueError("u, v, t_src, t_tgt, t_cam must have the same length.")
        for name, arr in (("t_src", ts), ("t_tgt", tt), ("t_cam", tc)):
            if M and (int(arr.min()) < 0 or int(arr.max()) >= T):
                raise ValueError(f"{name} out of range [0, {T})")
        if M and not (torch.isfinite(u_t).all() and torch.isfinite(v_t).all()):
            raise ValueError("u, v must be finite.")

        start = time.perf_counter()
        uo, vo = u_t.clone(), v_t.clone()
        need = torch.nonzero(ts != tt).flatten()
        with torch.no_grad():
            if need.numel():
                pair_id = ts[need] * T + tt[need]
                uniq, inv = torch.unique(pair_id, return_inverse=True)
                pairs = torch.stack([uniq // T, uniq % T], dim=-1)
                for a in range(0, pairs.shape[0], self.pair_chunk):
                    sel = torch.nonzero((inv >= a) & (inv < a + self.pair_chunk)).flatten()
                    if sel.numel() == 0:
                        continue
                    maps = self._flow_maps(pairs[a:a + self.pair_chunk])
                    rows = need[sel]
                    tgt = ndc_to_unit(bilinear(maps, inv[sel] - a, u_t[rows], v_t[rows]))
                    uo[rows], vo[rows] = tgt[:, 0], tgt[:, 1]
            inside = (uo >= 0) & (uo <= 1) & (vo >= 0) & (vo <= 1)
            X = bilinear(c["local"], tt, uo, vo).double()
            conf = bilinear(c["conf"][..., None], tt, uo, vo)[:, 0]
            poses = c["poses"]
            rel = torch.linalg.inv(poses[tc]) @ poses[tt]
            P = (rel[:, :3, :3] @ X[..., None])[..., 0] + rel[:, :3, 3]
        self._sync()
        self.last_decode_seconds = time.perf_counter() - start
        vis = torch.where(inside, torch.full_like(uo, VIS_LOGIT), torch.full_like(uo, -VIS_LOGIT))
        return {
            "xyz_3d": P.float().cpu().numpy(),
            "uv_2d": torch.stack([uo, vo], -1).float().cpu().numpy(),
            "visibility": vis.float().cpu().numpy(),
            "confidence": conf.float().cpu().numpy(),
        }
