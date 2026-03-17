# Small-Scale Benchmarking Experiment Plan

Status (2026-03-12): This document is an early comparison design memo. The
authoritative sources for the current implementation are
`experiments/toys4k_baseline/config.yaml`, `pipeline/docs/evaluation_protocol.md`,
and `pipeline/docs/model_notes.md`. In particular, references to SAM-3D-Objects
in this document are not part of the active pipeline.

## 1. Benchmark Selection

### Dataset: Google Scanned Objects (GSO) subset, rendered to images

**Why GSO:**
- Contains real scanned 3D objects with ground-truth meshes, enabling geometry comparison
- Objects are single isolated items on clean backgrounds -- compatible with all single-image models
- Used extensively in prior 3D generation benchmarks (referenced indirectly by TRELLIS and Hunyuan3D evaluation pipelines)
- Objects can be rendered from controlled viewpoints to produce both single-view and multi-view inputs

**Alternative if GSO is unavailable:** Use the **TRELLIS example images** already present at `TRELLIS/assets/example_image/` (41 images covering buildings, creatures, humanoids, vehicles, misc objects). These lack ground-truth geometry but enable end-to-end smoke testing and visual comparison across all models. A hybrid approach is recommended: use TRELLIS example images for smoke testing, and a small GSO subset for quantitative evaluation.

**Inputs provided:** Single RGBA images (background removed) per object; optionally 4-view renders (front/left/back/right) for multiview-capable models.

**Expected outputs:** Textured or untextured 3D meshes (.glb/.obj) per object.

### Proposed subset

- **Smoke test:** 10 objects from `TRELLIS/assets/example_image/` (select 2 per category: building, creature, humanoid, vehicle, misc)
- **Quantitative test:** 20 objects from GSO (or Toys4k via `JeffreyXiang/TRELLIS-500K` HuggingFace metadata -- Toys4k is TRELLIS's designated evaluation set with 3,229 assets)

For Toys4k: download 20 objects using TRELLIS's dataset toolkit:
```bash
cd TRELLIS
python dataset_toolkits/build_metadata.py Toys4k --output_dir datasets/Toys4k
python dataset_toolkits/download.py Toys4k --output_dir datasets/Toys4k --world_size 162
# This downloads ~20 objects (3229/162 ~ 20)
python dataset_toolkits/render_cond.py Toys4k --output_dir datasets/Toys4k --num_views 4
```

### Model applicability

| Model | Single-image | Multi-view | Notes |
|-------|-------------|------------|-------|
| TRELLIS | Yes | Yes (tuning-free) | Primary target |
| Hunyuan3D | Yes | Yes (2mv variant) | Primary target |
| SAM-3D-Objects | Yes (+ mask) | No | Needs object mask |
| Shaper | No | Yes (+ point cloud) | Requires SLAM data -- **excluded from shared benchmark** |
| CAST | Yes | No | Pipeline wrapping other models -- **excluded or tested separately** |

**Decision:** Focus quantitative comparison on **TRELLIS, Hunyuan3D, and SAM-3D-Objects** (the three models accepting single images). Shaper is excluded because it requires Aria SLAM data incompatible with rendered images. CAST is excluded because it is a meta-pipeline calling external APIs (Tripo3D, TRELLIS, Hunyuan3D) rather than a standalone model.

---

## 2. Input Protocol

### SV-1: Single View (all three models)

| Property | Specification |
|----------|--------------|
| Format | RGBA PNG, white or transparent background |
| Resolution | 512x512 (native for Hunyuan3D; TRELLIS preprocessor resizes to 518x518; SAM-3D accepts variable) |
| Background | Removed via `rembg` (u2net), stored as alpha channel |
| Camera | Front view, elevation ~20-30 degrees, normalized object scale (fills ~80% of frame) |
| Normalization | Object centered, longest axis fills frame with 15-20% border |

**Preprocessing script (shared across models):**
```python
from PIL import Image
from rembg import remove

def prepare_sv1(image_path, output_path, size=512):
    img = Image.open(image_path).convert("RGBA")
    img = remove(img)  # background removal
    # Center-crop and resize to size x size
    img.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    offset = ((size - img.width) // 2, (size - img.height) // 2)
    canvas.paste(img, offset)
    canvas.save(output_path)
```

### MV-4: Four Views (TRELLIS multi-image, Hunyuan3D-2mv)

| Property | Specification |
|----------|--------------|
| Views | Front (0), Left (90), Back (180), Right (270) at elevation 20 degrees |
| Format | Same as SV-1 per view |
| Naming | `{object_id}/front.png`, `left.png`, `back.png`, `right.png` |

Rendered using TRELLIS dataset toolkit (`render_cond.py`) or Blender script for GSO objects.

### Per-model input mapping

**TRELLIS:**
```python
image = Image.open("object_id/front.png")
outputs = pipeline.run(image, seed=42, ...)
# Multi-view:
images = [Image.open(f"object_id/{v}.png") for v in ["front","left","back","right"]]
outputs = pipeline.run_multi_image(images, seed=42, mode="stochastic", ...)
```

**Hunyuan3D:**
```python
image = Image.open("object_id/front.png")
mesh = pipeline(image=image, num_inference_steps=50, guidance_scale=5.0, generator=torch.manual_seed(42))
# Multi-view:
mesh = pipeline(image={"front": front, "left": left, "back": back}, ...)
```

**SAM-3D-Objects:**
```python
from notebook.inference import Inference, load_image
inference = Inference("checkpoints/hf/pipeline.yaml", compile=False)
image = load_image("object_id/front.png")  # RGB
# Mask: full-object mask from alpha channel
mask = np.array(Image.open("object_id/front.png").split()[-1]) > 128
output = inference(image, mask, seed=42)
```

---

## 3. Parameter Test (Inference Ablation)

### 3.1 TRELLIS parameter grid

Source: `trellis/pipelines/trellis_image_to_3d.py`, `example.py`

| Parameter | Default | Test values |
|-----------|---------|-------------|
| `ss_steps` (sparse structure sampling steps) | 12 | {4, 8, 12, 25, 50} |
| `slat_steps` (structured latent sampling steps) | 12 | {4, 8, 12, 25, 50} |
| `ss_cfg_strength` | 7.5 | {3.0, 5.0, 7.5, 10.0} |
| `slat_cfg_strength` | 3.0 | {1.5, 3.0, 5.0, 7.5} |
| seed | 42 | fixed |

**Primary sweep (steps, 5 configs):** Vary `ss_steps = slat_steps` together: {4, 8, 12, 25, 50}. Keep cfg at defaults.

**Secondary sweep (guidance, 4 configs):** Fix steps=12. Vary `ss_cfg_strength` in {3.0, 7.5} x `slat_cfg_strength` in {1.5, 3.0}: 4 configs.

**Total: 9 configs x 10 objects = 90 runs**

```python
# Example run
outputs = pipeline.run(
    image, seed=42,
    sparse_structure_sampler_params={"steps": steps, "cfg_strength": ss_cfg},
    slat_sampler_params={"steps": steps, "cfg_strength": slat_cfg},
    formats=["mesh", "gaussian"],
)
```

### 3.2 Hunyuan3D parameter grid

Source: `hy3dgen/shapegen/pipelines.py`, `examples/shape_gen.py`

| Parameter | Default | Test values |
|-----------|---------|-------------|
| `num_inference_steps` | 50 | {5, 10, 20, 30, 50} |
| `guidance_scale` | 5.0 | {2.5, 5.0, 7.5} |
| `octree_resolution` | 384 | {256, 384} |
| seed | 42 | fixed via `torch.manual_seed(42)` |

**Primary sweep (steps, 5 configs):** Vary `num_inference_steps`: {5, 10, 20, 30, 50}. Keep guidance=5.0, octree=384.

**Secondary sweep (guidance x resolution, 4 configs):** Fix steps=50. Vary `guidance_scale` in {2.5, 7.5} x `octree_resolution` in {256, 384}.

**Total: 9 configs x 10 objects = 90 runs**

```python
mesh = pipeline(
    image=image,
    num_inference_steps=steps,
    guidance_scale=guidance,
    octree_resolution=octree_res,
    generator=torch.manual_seed(42),
    output_type='trimesh',
)[0]
```

### 3.3 SAM-3D-Objects parameter grid

Source: `sam3d_objects/pipeline/inference_pipeline.py`, `notebook/inference.py`

| Parameter | Default | Test values |
|-----------|---------|-------------|
| `ss_inference_steps` | 25 | {8, 15, 25, 50} |
| `slat_inference_steps` | 25 | {8, 15, 25, 50} |
| `ss_cfg_strength` | 7.0 | {3.0, 7.0, 10.0} |
| `slat_cfg_strength` | 5.0 | {3.0, 5.0, 7.0} |
| seed | 42 | fixed |

**Primary sweep (steps, 4 configs):** Vary `ss_inference_steps = slat_inference_steps` together: {8, 15, 25, 50}. Keep cfg at defaults.

**Secondary sweep (guidance, 4 configs):** Fix steps=25. Vary `ss_cfg_strength` in {3.0, 7.0} x `slat_cfg_strength` in {3.0, 5.0}.

**Total: 8 configs x 10 objects = 80 runs**

Note: SAM-3D-Objects parameters are set at `InferencePipelinePointMap` construction time, not per-call. Each config requires re-instantiating the pipeline or modifying the pipeline.yaml.

### 3.4 Total experiment size

| Model | Configs | Objects | Total runs |
|-------|---------|---------|------------|
| TRELLIS | 9 | 10 | 90 |
| Hunyuan3D | 9 | 10 | 90 |
| SAM-3D-Objects | 8 | 10 | 80 |
| **Total** | | | **260** |

---

## 4. Execution Procedure

### 4.1 Environment setup

Each model requires a separate conda/mamba environment due to conflicting dependencies.

```bash
# TRELLIS (requires: spconv, flash-attn, kaolin, nvdiffrast)
mamba create -n trellis python=3.10
mamba activate trellis
cd TRELLIS && pip install -r requirements.txt
# Install spconv, flash-attn, kaolin per TRELLIS INSTALL.md

# Hunyuan3D (requires: diffusers, rembg, xatlas)
mamba create -n hunyuan3d python=3.10
mamba activate hunyuan3d
cd Hunyuan3D-2 && pip install -r requirements.txt

# SAM-3D-Objects (requires: pytorch3d, kaolin, hydra, moge)
mamba env create -f sam-3d-objects/environments/default.yml
mamba activate sam3d-objects
cd sam-3d-objects && pip install -e '.[dev]' && pip install -e '.[p3d]' && pip install -e '.[inference]'
```

Download checkpoints:
```bash
# TRELLIS -- auto-downloads from HuggingFace on first run
# Hunyuan3D -- auto-downloads from tencent/Hunyuan3D-2
# SAM-3D-Objects -- gated access
cd sam-3d-objects && hf download --repo-type model --local-dir checkpoints/hf facebook/sam-3d-objects
```

### 4.2 Dataset preparation

```bash
mkdir -p benchmark/inputs/sv1 benchmark/inputs/mv4 benchmark/outputs

# Option A: Use TRELLIS example images (smoke test)
for f in TRELLIS/assets/example_image/typical_*.png; do
    python preprocess.py --input "$f" --output "benchmark/inputs/sv1/$(basename $f)" --size 512
done
# Select 10 representative images

# Option B: Download and render Toys4k subset (quantitative)
cd TRELLIS
python dataset_toolkits/build_metadata.py Toys4k --output_dir datasets/Toys4k
python dataset_toolkits/download.py Toys4k --output_dir datasets/Toys4k --world_size 162
python dataset_toolkits/render_cond.py Toys4k --output_dir datasets/Toys4k --num_views 4
# Copy rendered front views to benchmark/inputs/sv1/
```

### 4.3 Running inference

Create a driver script per model. Example for TRELLIS:

```python
# run_trellis.py
import time, json, torch, os
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils
from PIL import Image

pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()

CONFIGS = [
    {"ss_steps": 4,  "slat_steps": 4,  "ss_cfg": 7.5, "slat_cfg": 3.0},
    {"ss_steps": 8,  "slat_steps": 8,  "ss_cfg": 7.5, "slat_cfg": 3.0},
    {"ss_steps": 12, "slat_steps": 12, "ss_cfg": 7.5, "slat_cfg": 3.0},
    {"ss_steps": 25, "slat_steps": 25, "ss_cfg": 7.5, "slat_cfg": 3.0},
    {"ss_steps": 50, "slat_steps": 50, "ss_cfg": 7.5, "slat_cfg": 3.0},
    {"ss_steps": 12, "slat_steps": 12, "ss_cfg": 3.0, "slat_cfg": 1.5},
    {"ss_steps": 12, "slat_steps": 12, "ss_cfg": 3.0, "slat_cfg": 3.0},
    {"ss_steps": 12, "slat_steps": 12, "ss_cfg": 7.5, "slat_cfg": 1.5},
    {"ss_steps": 12, "slat_steps": 12, "ss_cfg": 7.5, "slat_cfg": 3.0},  # duplicate = default baseline
]

images = sorted(os.listdir("benchmark/inputs/sv1/"))[:10]

for cfg in CONFIGS:
    cfg_name = f"ss{cfg['ss_steps']}_slat{cfg['slat_steps']}_sscfg{cfg['ss_cfg']}_slatcfg{cfg['slat_cfg']}"
    for img_name in images:
        obj_id = os.path.splitext(img_name)[0]
        out_dir = f"benchmark/outputs/trellis/{cfg_name}/{obj_id}"
        os.makedirs(out_dir, exist_ok=True)

        image = Image.open(f"benchmark/inputs/sv1/{img_name}")
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        outputs = pipeline.run(
            image, seed=42,
            sparse_structure_sampler_params={"steps": cfg["ss_steps"], "cfg_strength": cfg["ss_cfg"]},
            slat_sampler_params={"steps": cfg["slat_steps"], "cfg_strength": cfg["slat_cfg"]},
            formats=["mesh", "gaussian"],
        )

        runtime = time.time() - t0
        peak_mem = torch.cuda.max_memory_allocated() / 1e9

        glb = postprocessing_utils.to_glb(outputs["gaussian"][0], outputs["mesh"][0], simplify=0.95)
        glb.export(f"{out_dir}/mesh.glb")
        outputs["gaussian"][0].save_ply(f"{out_dir}/gaussian.ply")

        json.dump({
            "model": "trellis", "config": cfg, "object_id": obj_id,
            "runtime_sec": runtime, "peak_gpu_gb": peak_mem, "seed": 42,
        }, open(f"{out_dir}/meta.json", "w"), indent=2)
```

Analogous scripts for Hunyuan3D (`run_hunyuan.py`) and SAM-3D-Objects (`run_sam3d.py`).

### 4.4 Collecting outputs

All outputs go to:
```
benchmark/outputs/{model}/{config_name}/{object_id}/
    mesh.glb          # exported mesh
    gaussian.ply      # gaussian splat (if available)
    meta.json         # runtime, memory, config, seed
```

### 4.5 Logging runtime and GPU usage

Each `meta.json` records:
- `runtime_sec`: wall-clock time from input to mesh export
- `peak_gpu_gb`: `torch.cuda.max_memory_allocated()` in GB
- Full config parameters
- Seed value
- Model name and variant

Additionally, use `nvidia-smi` polling in background:
```bash
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 1 > benchmark/gpu_log.csv &
```

### 4.6 Reproducibility

- **Seeds:** Fixed at 42 for all runs. Set via `torch.manual_seed(42)` before each inference call.
- **Environment:** Record `pip freeze > requirements_frozen.txt` per environment.
- **Hardware:** Record GPU model, driver version, CUDA version in a `benchmark/hardware.json`.
- **Determinism:** Set `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`.

---

## 5. Evaluation Protocol

### 5.1 Geometry quality (requires ground-truth mesh)

Available only for Toys4k/GSO objects with GT meshes. Use existing implementations.

**Chamfer Distance (CD):**
```python
# Bilateral Chamfer Distance between predicted and GT point clouds
# Sample 10,000 points from each mesh surface
import trimesh
pred_pts = trimesh.load("mesh.glb").sample(10000)
gt_pts = trimesh.load("gt.glb").sample(10000)
# CD = mean(min_dist(pred->gt)) + mean(min_dist(gt->pred))
```

**F-score (F1 at threshold tau):**
```python
# F-score at tau = 0.01 (1% of bounding box diagonal)
# Precision = fraction of pred points within tau of GT
# Recall = fraction of GT points within tau of pred
# F1 = 2 * P * R / (P + R)
```

**Volume IoU (if voxelizable):**
```python
# Voxelize both meshes at 64^3, compute intersection/union
```

### 5.2 Rendering quality (no GT mesh needed)

For all objects including those without GT. Render the predicted mesh from held-out viewpoints and compare to rendered GT or input views.

Use TRELLIS's existing loss utilities (`trellis/utils/loss_utils.py`):

**PSNR** (line 34): `trellis.utils.loss_utils.psnr(img1, img2)`

**SSIM** (line 39): `trellis.utils.loss_utils.ssim(img1, img2)`

**LPIPS** (line 73): `trellis.utils.loss_utils.lpips(img1, img2)`

**Procedure:**
1. Render predicted mesh from 8 canonical viewpoints (azimuths 0/45/90/.../315, elevation 20)
2. Render GT mesh (or use GT renders) from same viewpoints
3. Compute PSNR, SSIM, LPIPS per view, average across views

For models without texture (geometry-only outputs), compare only silhouette IoU and depth maps.

### 5.3 Efficiency metrics

| Metric | Source |
|--------|--------|
| Wall-clock runtime (seconds) | `time.time()` around pipeline call |
| Peak GPU memory (GB) | `torch.cuda.max_memory_allocated()` |
| Sampling steps | Config parameter |

### 5.4 Visual quality (no GT needed)

For smoke testing without GT:
- Render 360-degree turntable video of each output
- Side-by-side grid: input image | front render | side render | back render
- Manual inspection checklist: geometry completeness, texture quality, artifacts

### 5.5 Evaluation script outline

```python
# evaluate.py
import json, os, trimesh
import numpy as np
from trellis.utils.loss_utils import psnr, ssim, lpips

def chamfer_distance(pts1, pts2):
    from scipy.spatial import cKDTree
    tree1, tree2 = cKDTree(pts1), cKDTree(pts2)
    d1, _ = tree1.query(pts2)
    d2, _ = tree2.query(pts1)
    return d1.mean() + d2.mean()

def f_score(pts1, pts2, tau=0.01):
    from scipy.spatial import cKDTree
    tree1, tree2 = cKDTree(pts1), cKDTree(pts2)
    d1, _ = tree1.query(pts2)
    d2, _ = tree2.query(pts1)
    precision = (d2 < tau).mean()
    recall = (d1 < tau).mean()
    if precision + recall == 0: return 0
    return 2 * precision * recall / (precision + recall)

for model in ["trellis", "hunyuan3d", "sam3d"]:
    for config_dir in sorted(os.listdir(f"benchmark/outputs/{model}")):
        for obj_dir in sorted(os.listdir(f"benchmark/outputs/{model}/{config_dir}")):
            pred_path = f"benchmark/outputs/{model}/{config_dir}/{obj_dir}/mesh.glb"
            gt_path = f"benchmark/gt_meshes/{obj_dir}.glb"
            # ... compute metrics, save to metrics.json
```

---

## 6. Result Logging

### Directory structure

```
benchmark/
  inputs/
    sv1/
      {object_id}.png
    mv4/
      {object_id}/
        front.png
        left.png
        back.png
        right.png
  gt_meshes/
    {object_id}.glb
  outputs/
    {model}/
      {config_name}/
        {object_id}/
          mesh.glb
          gaussian.ply          # if available
          meta.json             # runtime, memory, config
          metrics.json          # CD, F-score, PSNR, SSIM, LPIPS
          renders/
            view_000.png        # rendered from canonical viewpoints
            view_045.png
            ...
  summary/
    results_table.csv           # all metrics in tabular format
    parameter_sensitivity.csv   # steps vs. quality curves
    hardware.json               # GPU, driver, CUDA info
    environments/               # pip freeze per model
```

### meta.json schema

```json
{
  "model": "trellis",
  "model_variant": "TRELLIS-image-large",
  "config": {
    "ss_steps": 12,
    "slat_steps": 12,
    "ss_cfg_strength": 7.5,
    "slat_cfg_strength": 3.0
  },
  "object_id": "typical_misc_lantern",
  "seed": 42,
  "runtime_sec": 14.3,
  "peak_gpu_gb": 8.7,
  "output_files": ["mesh.glb", "gaussian.ply"],
  "timestamp": "2026-03-05T12:00:00Z"
}
```

### metrics.json schema

```json
{
  "chamfer_distance": 0.0123,
  "f_score_001": 0.87,
  "volume_iou": 0.72,
  "psnr_mean": 24.5,
  "ssim_mean": 0.91,
  "lpips_mean": 0.08,
  "num_vertices": 45000,
  "num_faces": 90000
}
```

### results_table.csv

```
model,config,object_id,cd,f_score,psnr,ssim,lpips,runtime_sec,peak_gpu_gb,steps
trellis,ss12_slat12_default,lantern,0.012,0.87,24.5,0.91,0.08,14.3,8.7,12
hunyuan3d,steps50_cfg5.0,lantern,0.015,0.82,23.1,0.88,0.11,45.2,6.1,50
sam3d,ss25_slat25_default,lantern,0.010,0.90,25.1,0.93,0.07,32.1,28.4,25
```

---

## 7. Comparison Strategy

### 7.1 Controlling for compute

Models have very different GPU memory requirements:

| Model | Min VRAM | Typical runtime (default config) |
|-------|----------|----------------------------------|
| TRELLIS | 16 GB | ~15s (12 steps) |
| Hunyuan3D | 6 GB (shape only) | ~30-60s (50 steps) |
| SAM-3D-Objects | 32 GB | ~30-40s (25+25 steps) |

**Fair comparison strategy:**
1. **Fixed-budget comparison:** Compare at similar total sampling steps (e.g., all at ~25 total steps). For TRELLIS: 12+12=24. For Hunyuan3D: 25. For SAM-3D: 12+12=24.
2. **Best-quality comparison:** Each model at its default/recommended settings.
3. **Pareto frontier:** Plot quality (CD or F-score) vs. runtime. Identify which model offers the best quality-speed tradeoff.

### 7.2 Interpreting parameter sweeps

For each model, produce:
1. **Steps vs. quality curve:** X-axis = sampling steps, Y-axis = CD / F-score. Identifies diminishing returns threshold.
2. **Steps vs. runtime curve:** X-axis = sampling steps, Y-axis = seconds. Shows scaling behavior.
3. **Guidance sensitivity plot:** Quality at different CFG strengths with fixed steps. Identifies optimal guidance range.

**Key questions to answer:**
- At what step count does quality plateau for each model?
- Is there a "sweet spot" step count for fast inference with minimal quality loss?
- How sensitive is each model to guidance strength?

### 7.3 Identifying the best distillation candidate

Score each model on:

| Criterion | Weight | How to measure |
|-----------|--------|----------------|
| Output quality at default settings | 30% | CD, F-score, PSNR on benchmark |
| Quality retention at low steps | 20% | Quality ratio: (quality@8steps / quality@default) |
| Latent accessibility | 15% | Can we extract and manipulate latent codes? (binary from code review) |
| Inference efficiency | 10% | Runtime and GPU memory |
| License compatibility | 15% | Can outputs be used for distillation? (binary from license review) |
| Training code availability | 10% | Can we retrain/finetune? (binary) |

**Already known from architecture survey:**

| Criterion | TRELLIS | Hunyuan3D | SAM-3D-Objects |
|-----------|---------|-----------|----------------|
| Latent accessible | Yes (8-ch sparse) | Yes (vectset) | Yes (two-level SLaT) |
| License OK for distillation | Yes (MIT) | **No** (Section 5b blocks it) | Likely yes (SAM License) |
| Training code | Yes | No (this repo) | No |

**Hunyuan3D is pre-disqualified for distillation** due to license restrictions, but still included in the quality benchmark for reference.

### 7.4 Decision framework

After the experiment, rank models by:

1. **If distillation is the goal:** Exclude Hunyuan3D. Between TRELLIS and SAM-3D-Objects, prefer whichever shows better quality retention at low step counts (indicating the latent space is well-structured for few-step generation) AND has accessible training infrastructure.

2. **If pure quality matters:** Rank by CD/F-score at default settings across the 10-20 test objects.

3. **If efficiency matters:** Rank by quality-per-second (F-score / runtime) at various operating points.

---

## Appendix: Quick-Start Commands

### Smoke test (10 objects, default configs only)

```bash
# TRELLIS
conda activate trellis
python run_trellis.py --config default_only --objects 10

# Hunyuan3D
conda activate hunyuan3d
python run_hunyuan.py --config default_only --objects 10

# SAM-3D-Objects
conda activate sam3d-objects
python run_sam3d.py --config default_only --objects 10
```

### Full ablation (10 objects, all configs)

```bash
# ~90 runs per model, expect ~2-4 hours per model on single A100
python run_trellis.py --config all --objects 10
python run_hunyuan.py --config all --objects 10
python run_sam3d.py --config all --objects 10

# Evaluate
python evaluate.py --gt_dir benchmark/gt_meshes --output_dir benchmark/outputs
python summarize.py  # produces results_table.csv and plots
```
