"""
Cross-modal metrics for 3D distillation eval: ULIP-I and Uni3D-I.

Uses FlashVDM's bundled ULIP_PointBERT and Uni3DScore as underlying model wrappers.
We provide our own per-sample / per-model loop suitable for cross-family-bench
output structure (predictions/<model>/default/<oid>/mesh_raw.obj).

Adopted from Tencent-Hunyuan/FlashVDM evaluation/{ulip,uni3d}/*.py (2025-07-29).

Verified API (FlashVDM 2025-07-29):
  ULIP():              no constructor args; ckpt at ULIP/...pointbert.pt (relative)
  Uni3DScore():        no constructor args; ckpts at uni3d/{open_clip,model}.pt (rel)
  .sim_img(pc, images): returns (B, N_images) cosine similarity tensor
  Neither applies coordinate normalization → we normalize externally.

NOTE on import paths (5/20 fix):
  - `ulip/ulip_score.py` uses `from ULIP.models.ULIP_models import ULIP_PointBERT`
    → requires cwd=external/ulip/ AND sys.path includes that dir
  - `uni3d/uni3d_score.py` uses `from models.uni3d import ...`
    → requires cwd=external/uni3d/ AND sys.path includes that dir
  So we cd + path-insert *per model load*, not once at scorer __enter__.

References:
  ULIP-1 PointBERT  (arxiv 2212.05171)
  Uni3D-Giant       (arxiv 2310.06773, ICLR'24)
  open_clip EVA02-E-14-plus  (open_clip library)
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import trimesh
from PIL import Image

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# External (FlashVDM bundle) root
# ----------------------------------------------------------------------
DEFAULT_FLASHVDM_EVAL_ROOT = Path(os.environ.get(
    "FLASHVDM_EVAL_ROOT",
    "/gs/fs/tga-koike-shanda2/sk/3d-distill-eval/pipeline/src/evaluation/external",
))

ULIP_N_POINTS = 8192
UNI3D_N_POINTS = 10000


# ----------------------------------------------------------------------
# Per-sample result struct
# ----------------------------------------------------------------------
@dataclass
class CrossModalResult:
    object_id: str
    model: str
    ulip_i: Optional[float] = None
    uni3d_i: Optional[float] = None
    error: Optional[str] = None


# ----------------------------------------------------------------------
# cwd + sys.path switch context (for FlashVDM relative imports)
# ----------------------------------------------------------------------
@contextmanager
def _cwd_and_path(path: Path):
    """Temporarily cd into `path` and prepend it to sys.path."""
    orig_cwd = os.getcwd()
    path = Path(path).resolve()
    os.chdir(str(path))
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        try:
            sys.path.remove(str(path))
        except ValueError:
            pass
        os.chdir(orig_cwd)


# Top-level package names that BOTH vendored libraries define locally:
#   ULIP imports `from models.pointnet2...` / `from utils import utils`
#   Uni3D imports `import models.uni3d` / `from utils.tokenizer import ...`
# Whichever loads first caches these under sys.modules, so the second
# library's `import models.<x>` resolves to the first one's package and
# raises ModuleNotFoundError. We evict them after each load (see below).
_CONFLICTING_PKGS = ("models", "utils", "data")


def _is_conflicting(module_name: str) -> bool:
    return module_name.split(".", 1)[0] in _CONFLICTING_PKGS


@contextmanager
def _isolated_load(lib_path: Path, extra_paths: tuple[Path, ...] = ()):
    """Load a self-contained vendored library that uses generic top-level
    package names without polluting/colliding with a sibling library that
    uses the same names.

    chdir into `lib_path` and prepend it (plus `extra_paths`) to sys.path
    for the duration, then on exit restore cwd/sys.path and evict the
    conflicting top-level packages the load introduced (restoring any that
    were cached beforehand). The library's already-instantiated model
    objects keep working because they hold references to the loaded classes;
    only the import cache is cleared so the next library imports its own.
    """
    lib_path = Path(lib_path).resolve()
    orig_cwd = os.getcwd()
    added_paths = []
    for p in (lib_path, *extra_paths):
        sp = str(Path(p).resolve())
        if sp not in sys.path:
            sys.path.insert(0, sp)
            added_paths.append(sp)
    # Preserve any pre-existing modules under the conflicting names.
    saved = {name: mod for name, mod in sys.modules.items() if _is_conflicting(name)}
    os.chdir(str(lib_path))
    try:
        yield
    finally:
        os.chdir(orig_cwd)
        for sp in added_paths:
            try:
                sys.path.remove(sp)
            except ValueError:
                pass
        for name in [n for n in sys.modules if _is_conflicting(n)]:
            del sys.modules[name]
        sys.modules.update(saved)


# ----------------------------------------------------------------------
# Combined scorer
# ----------------------------------------------------------------------
class CrossModalScorer:
    """
    Holds ULIP and Uni3D models on a single GPU. Each model is loaded with
    its own cwd + sys.path context (FlashVDM expects different relative paths).
    """

    def __init__(
        self,
        flashvdm_eval_root: Path = DEFAULT_FLASHVDM_EVAL_ROOT,
        device: str = "cuda",
    ):
        self.device = device
        self.flashvdm_eval_root = Path(flashvdm_eval_root)
        if not self.flashvdm_eval_root.exists():
            raise FileNotFoundError(
                f"flashvdm_eval_root does not exist: {self.flashvdm_eval_root}"
            )

        self._ulip = None
        self._uni3d = None

    # --- ULIP-1 PointBERT ---
    def _load_ulip(self):
        if self._ulip is not None:
            return
        ulip_root = self.flashvdm_eval_root / "ulip"
        ulip_inner = ulip_root / "ULIP"  # for inner `from models.pointnet2...` imports
        with _isolated_load(ulip_root, extra_paths=(ulip_inner,)):
            from ulip_score import ULIP
            self._ulip = ULIP()
        logger.info("ULIP-1 PointBERT loaded")

    # --- Uni3D-Giant + EVA02-E-14-plus ---
    def _load_uni3d(self):
        if self._uni3d is not None:
            return
        uni3d_root = self.flashvdm_eval_root / "uni3d"
        with _isolated_load(uni3d_root):
            from uni3d_score import Uni3DScore  # at uni3d_root top-level
            self._uni3d = Uni3DScore()
        logger.info("Uni3D-Giant + EVA02-E-14-plus loaded")

    # --- Point cloud sampling + normalization ---
    @staticmethod
    def _sample_normalized_pc(mesh_path: str | Path, n_points: int) -> torch.Tensor:
        """Load mesh, sample n_points from surface, normalize to unit sphere,
        return (1, n, 3) tensor."""
        from src.geometry.normalize import normalize_to_unit_sphere

        mesh = trimesh.load(str(mesh_path), force="mesh")
        if mesh.faces is None or len(mesh.faces) == 0:
            raise ValueError(f"mesh has no faces: {mesh_path}")
        pts, _ = trimesh.sample.sample_surface(mesh, n_points)
        pts_norm, _ = normalize_to_unit_sphere(pts.astype(np.float32))
        return torch.from_numpy(pts_norm).unsqueeze(0)

    # --- Per-pair score ---
    def score_image(
        self,
        mesh_path: str | Path,
        image_path: str | Path,
        object_id: str,
        model_name: str,
    ) -> CrossModalResult:
        """Compute ULIP-I and Uni3D-I for one (mesh, image) pair."""
        result = CrossModalResult(object_id=object_id, model=model_name)
        try:
            img = Image.open(str(image_path)).convert("RGB")

            # ULIP-I (8192 points)
            self._load_ulip()
            pc_ulip = self._sample_normalized_pc(mesh_path, ULIP_N_POINTS).to(self.device)
            with torch.no_grad():
                # cwd needs to be ulip_root if sim_img internally uses any relative paths
                with _cwd_and_path(self.flashvdm_eval_root / "ulip"):
                    score_u = self._ulip.sim_img(pc_ulip, [img])
            result.ulip_i = float(score_u.view(-1)[0].item())

            # Uni3D-I (10000 points)
            self._load_uni3d()
            pc_uni3d = self._sample_normalized_pc(mesh_path, UNI3D_N_POINTS).to(self.device)
            with torch.no_grad():
                with _cwd_and_path(self.flashvdm_eval_root / "uni3d"):
                    score_n = self._uni3d.sim_img(pc_uni3d, [img])
            result.uni3d_i = float(score_n.view(-1)[0].item())

        except Exception as e:
            result.error = str(e)
            logger.error("score_image failed for %s/%s: %s", model_name, object_id, e)
        return result


# ----------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------
def compute_cross_modal_for_run(
    cfg: dict,
    samples: list,
    output_root: str,
    flashvdm_eval_root: Optional[Path] = None,
    model_cfgs: Optional[list] = None,
    out_csv: Optional[str] = None,
) -> list[CrossModalResult]:
    """Per-sample, per-model ULIP-I + Uni3D-I.

    `model_cfgs` is the (possibly --models-filtered) model list from the
    caller; falls back to cfg["models"] when not provided.
    `out_csv` overrides the output path (default
    <output_root>/metrics/cross_modal.csv); pass a per-shard path when running
    sharded so concurrent workers don't clobber one file.
    """
    import csv

    if model_cfgs is None:
        model_cfgs = cfg.get("models", [])

    scorer = CrossModalScorer(
        flashvdm_eval_root=flashvdm_eval_root or DEFAULT_FLASHVDM_EVAL_ROOT,
    )
    # Load both models up front. If a dependency or checkpoint is missing this
    # raises once, loudly, and aborts the run — instead of score_image()
    # swallowing the same import error per sample and re-attempting the (heavy)
    # load thousands of times, which silently empties the output and pins a CPU.
    scorer._load_ulip()
    scorer._load_uni3d()

    rows: list[CrossModalResult] = []
    for mcfg in model_cfgs:
        model_name = mcfg["name"]
        pred_root = mcfg["predictions_root"]
        mesh_fn = mcfg.get("mesh_filename", "mesh_raw.obj")
        for s in samples:
            mesh_path = Path(pred_root) / s.object_id / mesh_fn
            if not mesh_path.exists():
                rows.append(CrossModalResult(
                    object_id=s.object_id, model=model_name,
                    error=f"mesh not found: {mesh_path}",
                ))
                continue
            res = scorer.score_image(
                mesh_path=mesh_path,
                image_path=s.input_image,
                object_id=s.object_id,
                model_name=model_name,
            )
            rows.append(res)

    out_csv = Path(out_csv) if out_csv else Path(output_root) / "metrics" / "cross_modal.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["object_id", "model", "ulip_i", "uni3d_i", "error"])
        for r in rows:
            wr.writerow([r.object_id, r.model, r.ulip_i, r.uni3d_i, r.error or ""])
    logger.info("Wrote cross-modal metrics: %s (%d rows)", out_csv, len(rows))
    return rows
