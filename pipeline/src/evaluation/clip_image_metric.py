"""
CLIP-I metric for 3D distillation eval.

Measures cosine similarity between the input conditioning image and the
multiview renders of the generated mesh, in CLIP image embedding space.

Backbone: open_clip EVA02-E-14-plus — the same architecture/weights Uni3D-I
uses, chosen so CLIP-I needs no extra checkpoint beyond Uni3D-I's. NOTE: CLIP-I
loads its own open_clip instance and does not share Uni3D-I's bundled copy
(Uni3DScore builds its open_clip internally), so with both metrics enabled the
EVA02 weights are resident twice.

Input  : (input_image, per-view rendered images of mesh)
Output : per-sample float in [-1, 1] (cosine similarity, averaged over N views)

References:
  - LATTICE Table 2 / Hunyuan3D-2.1 Table 1 use ULIP/Uni3D for image-mesh
    similarity. CLIP-I here is an *additional* light-weight metric not
    requiring ULIP/Uni3D bundles. Useful as a sanity baseline.
  - open_clip EVA02-E-14-plus: laion2b_s9b_b144k (~10 GB ckpt)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Defaults (override via constructor)
# ----------------------------------------------------------------------
HF_HOME = Path(os.environ.get("HF_HOME", "/gs/fs/tga-koike-shanda2/sk/hf_cache"))

DEFAULT_OPEN_CLIP_MODEL = "EVA02-E-14-plus"
DEFAULT_OPEN_CLIP_PRETRAINED = "laion2b_s9b_b144k"

# Default view names produced by multiview_renderer
DEFAULT_VIEW_NAMES = ["view_0.png", "view_90.png", "view_180.png", "view_270.png"]


# ----------------------------------------------------------------------
# Per-sample result
# ----------------------------------------------------------------------
@dataclass
class CLIPImageResult:
    object_id: str
    model: str
    clip_i: Optional[float] = None
    n_views: int = 0
    error: Optional[str] = None


# ----------------------------------------------------------------------
# Scorer (single shared open_clip on GPU)
# ----------------------------------------------------------------------
class CLIPImageScorer:
    """
    Wraps open_clip EVA02-E-14-plus image encoder and computes CLIP-I as
    mean cosine similarity between input image and N rendered views.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_OPEN_CLIP_MODEL,
        pretrained: str = DEFAULT_OPEN_CLIP_PRETRAINED,
        device: str = "cuda",
        cache_dir: Optional[Path] = None,
    ):
        import open_clip
        self.device = device
        self.model_name = model_name

        if cache_dir is None:
            cache_dir = HF_HOME / "open_clip"
        cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading open_clip %s (pretrained=%s) ...", model_name, pretrained)
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
            cache_dir=str(cache_dir),
        )
        model.to(device).eval()
        self.model = model
        self.preprocess = preprocess
        logger.info("open_clip %s loaded on %s", model_name, device)

    @torch.no_grad()
    def _encode_image(self, img: Image.Image) -> torch.Tensor:
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return feat.squeeze(0)  # (D,)

    @torch.no_grad()
    def score_pair(
        self,
        input_image_path: str | Path,
        render_dir: str | Path,
        object_id: str,
        model_name: str,
        view_names: Optional[list[str]] = None,
        input_feat: Optional[torch.Tensor] = None,
    ) -> CLIPImageResult:
        """
        Compute CLIP-I for one (input image, multiview render dir) pair.
        Returns mean cosine sim over views.

        `input_feat`, if given, is the pre-encoded input-image embedding;
        pass it to avoid re-encoding the same input image once per model.
        """
        if view_names is None:
            view_names = DEFAULT_VIEW_NAMES
        result = CLIPImageResult(object_id=object_id, model=model_name)
        try:
            if input_feat is None:
                input_img = Image.open(str(input_image_path)).convert("RGB")
                input_feat = self._encode_image(input_img)

            sims = []
            for vn in view_names:
                vp = Path(render_dir) / vn
                if not vp.exists():
                    logger.debug("missing view: %s", vp)
                    continue
                view_img = Image.open(str(vp)).convert("RGB")
                view_feat = self._encode_image(view_img)
                sim = float((input_feat * view_feat).sum().item())
                sims.append(sim)

            if not sims:
                result.error = f"no views found under {render_dir}"
            else:
                result.clip_i = float(np.mean(sims))
                result.n_views = len(sims)
        except Exception as e:
            result.error = str(e)
            logger.error("CLIP-I failed for %s/%s: %s", model_name, object_id, e)
        return result


# ----------------------------------------------------------------------
# Main entry: per (sample, model) loop
# ----------------------------------------------------------------------
def compute_clip_image_for_run(
    cfg: dict,
    samples: list,  # list of Sample objects (object_id, mesh_obj, input_image, ...)
    renders_root: str,  # output_root + "/multiview_renders" produced by FD pipeline
    output_root: str,
    scorer: Optional[CLIPImageScorer] = None,
    model_cfgs: Optional[list] = None,
    view_names: Optional[list[str]] = None,
) -> list[CLIPImageResult]:
    """
    Compute CLIP-I per (sample, model). Renders must already exist under
    <renders_root>/<model_name>/<object_id>/view_*.png (produced by FD
    pipeline's multiview_renderer.render_all_meshes).

    `model_cfgs` is the (possibly --models-filtered) model list from the
    caller; falls back to cfg["models"] when not provided.
    `view_names` selects which per-view render files to score; defaults to
    the renderer's [0, 90, 180, 270] views. Pass it to match a non-default
    azimuth configuration.

    Output: <output_root>/metrics/clip_image.csv
        columns: object_id, model, clip_i, n_views, error
    """
    import csv

    if model_cfgs is None:
        model_cfgs = cfg.get("models", [])

    if scorer is None:
        scorer = CLIPImageScorer()

    # Pre-encode each distinct input image once (reused across all models),
    # keyed by image path. None marks an image that failed to load; score_pair
    # then falls back to loading it and records the per-pair error.
    input_feats: dict[str, Optional[torch.Tensor]] = {}
    for s in samples:
        key = str(s.input_image)
        if key not in input_feats:
            try:
                input_feats[key] = scorer._encode_image(
                    Image.open(key).convert("RGB")
                )
            except Exception as e:
                logger.error("CLIP-I input encode failed for %s: %s", key, e)
                input_feats[key] = None

    rows: list[CLIPImageResult] = []
    for mcfg in model_cfgs:
        model_name = mcfg["name"]
        for s in samples:
            render_dir = Path(renders_root) / model_name / s.object_id
            res = scorer.score_pair(
                input_image_path=s.input_image,
                render_dir=render_dir,
                object_id=s.object_id,
                model_name=model_name,
                view_names=view_names,
                input_feat=input_feats.get(str(s.input_image)),
            )
            rows.append(res)

    out_csv = Path(output_root) / "metrics" / "clip_image.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["object_id", "model", "clip_i", "n_views", "error"])
        for r in rows:
            wr.writerow([r.object_id, r.model, r.clip_i, r.n_views, r.error or ""])
    logger.info("Wrote CLIP-I metrics: %s (%d rows)", out_csv, len(rows))
    return rows
