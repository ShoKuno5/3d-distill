# Toys4k Quantitative Evaluation Plan

Status (2026-03-12): This document is a planning memo for full-scale evaluation.
The currently implemented pipeline is geometry-only subset evaluation. The
authoritative sources are `experiments/toys4k_baseline/config.yaml`,
`pipeline/docs/evaluation_protocol.md`, and `pipeline/docs/model_notes.md`.
Sections describing FD/KD, LPIPS/PSNR, and SAM-3D are not implemented or
not part of the current pipeline.

## Goal

Run a quantitative benchmark of TRELLIS, Hunyuan3D-2, and SAM-3D-Objects on the Toys4k dataset, aligning as closely as possible with the evaluation protocol in the TRELLIS paper ("Structured 3D Latents for Scalable and Versatile 3D Generation", arXiv:2412.01506).

The primary purpose is to produce comparable numbers across the three models for both **generation quality** (distributional metrics: FD, KD) and **per-object reconstruction fidelity** (PSNR, LPIPS, Chamfer Distance, F-score), using Toys4k as the test set — exactly as in the TRELLIS paper.

---

## Trellis Paper Settings (What We Know)

### Facts (from paper + repo)

| Setting | Value | Source |
|---------|-------|--------|
| Evaluation dataset | Toys4k, 3,229 objects (filtered at aesthetic ≥ 4.5) | `DATASET.md`, paper §5 |
| Toys4k excluded from training | Yes — explicitly stated | Paper: "not part of our training set or those of the compared methods" |
| Distributional metrics | FD (Fréchet Distance), KD (Kernel Distance) × 100 | Table 2 |
| Feature extractors for FD/KD | Inception-v3, DINOv2, PointNet++ | Table 2 columns |
| Per-object reconstruction metrics | PSNR, LPIPS (appearance); CD, F-score, PSNR-N, LPIPS-N (geometry) | Table 1 |
| CLIP score | Reported for text-to-3D | Table 2 |
| Inference steps (generation) | 50 | Paper §5: "sampling steps to 50" |
| CFG strength (generation) | 3.0 | Paper §5: "CFG strength is set to 3" |
| Training rendering | 150 views per asset, 512×512, Blender CYCLES | `render.py` |
| Training rendering camera | r=2.0, FOV=40°, Hammersley sphere sampling | `render.py:_render()` |
| Condition image rendering | 1024×1024, variable FOV 10-70°, augmented camera | `render_cond.py` |
| Output types evaluated | Gaussians for appearance, meshes for geometry | Paper §5 |
| Object normalization | Fit in unit cube centered at origin (Blender script) | `blender_script/render.py:normalize_scene()` |
| Mesh export from render | Triangulated PLY via `--save_mesh` | `blender_script/render.py` |
| Model variants reported | Large (1.1B), XL (2B) | Paper Table 2 |

### Inferences (not explicitly stated in paper, derived from code/convention)

| Setting | Inferred Value | Reasoning |
|---------|---------------|-----------|
| Evaluation rendering resolution | 512×512 | Matches training renders and default in `render_utils.py` |
| Number of rendered views for FD/KD | 150 per object | Same as training rendering pipeline (render.py default) |
| Evaluation camera config | r=2.0, FOV=40°, Hammersley sequence | Consistent across all render scripts |
| FD/KD image pool | 150 views × 3,229 objects = ~484K images | Standard practice for distributional metrics |
| Point cloud sampling for CD | 10,000 points (convention) | Standard in 3D generation benchmarks |
| F-score threshold | τ = 0.01 (1% of bbox diagonal) | Standard; matches `experiment_plan.md` |
| ICP alignment before CD | Unknown — likely no | Paper does not mention alignment; models should produce centered outputs |
| Scale normalization | Objects normalized to unit cube before comparison | Blender script does this; models should respect similar normalization |

### Uncertainties

1. **Exact number of views for FD/KD computation**: Paper says "render 150 images per asset" in training context; unclear if same count used for eval. Could be fewer (e.g., 20-30) for efficiency.
2. **Image resolution for FD/KD**: Could be 512 (training default) or 256 (common for FID computation). Inception-v3 natively uses 299×299, DINOv2 uses 224×224 or 518×518.
3. **PointNet++ FD details**: Unclear whether point clouds are extracted from meshes or from rendered depth maps.
4. **Exact reconstruction evaluation protocol**: Table 1 (reconstruction fidelity) may use different camera views than Table 2 (generation quality). Reconstruction is likely evaluated on held-out views from the 150 training renders.
5. **Full Table 2 image-to-3D numbers**: The paper HTML truncated some columns; we have text-to-3D numbers but image-to-3D may differ.

---

## Experiment Design

### Scope

We will run **image-to-3D** inference on all 3,229 Toys4k objects across three models:
- TRELLIS (TRELLIS-image-large)
- Hunyuan3D-2
- SAM-3D-Objects

We compute:
1. **Distributional metrics** (FD, KD with Inception-v3 and DINOv2 features) on rendered multi-view images
2. **Per-object appearance metrics** (PSNR, LPIPS) comparing rendered views of predictions vs. ground-truth renders
3. **Per-object geometry metrics** (Chamfer Distance, F-score) comparing predicted meshes vs. ground-truth meshes
4. **Efficiency metrics** (runtime, peak VRAM)

### Input Preparation

For each Toys4k object, we need:
- **Condition image**: A single rendered view to feed as input to all three models
- **Ground-truth multi-view renders**: 150 views at 512×512 for metric computation
- **Ground-truth mesh**: For Chamfer distance / F-score

The TRELLIS dataset toolkit handles rendering via Blender. The condition image will be one of the `render_cond.py` outputs (or we render a canonical front view specifically).

### Model Parameters

All models use their recommended/default inference settings:

| Model | Key Parameters |
|-------|---------------|
| TRELLIS | ss_steps=50, slat_steps=50, ss_cfg=7.5, slat_cfg=3.0, seed=42 |
| Hunyuan3D | num_inference_steps=50, guidance_scale=7.5, octree_resolution=256, seed=42 |
| SAM-3D | Official defaults, seed=42 |

**Note on TRELLIS steps**: The paper states "sampling steps to 50" for generation evaluation, not the default 12 used for interactive demos. We should run **both** 12-step (default) and 50-step (paper eval) for TRELLIS to enable fair comparison.

---

## Toys4k Dataset Acquisition and Preparation

### Step 1: Build metadata

```bash
cd /mnt/workspace/kuno/distillation/TRELLIS
PYTHONPATH=dataset_toolkits python dataset_toolkits/build_metadata.py Toys4k \
    --output_dir /mnt/workspace/kuno/distillation/datasets/Toys4k
```

This downloads `Toys4k.csv` from HuggingFace (`JeffreyXiang/TRELLIS-500K`) and creates `metadata.csv`.

### Step 2: Download 3D assets

Toys4k requires **manual download** of `toys4k_blend_files.zip` from:
https://github.com/rehg-lab/lowshot-shapebias/tree/main/toys4k

```bash
# Manual step: download toys4k_blend_files.zip
mkdir -p /mnt/workspace/kuno/distillation/datasets/Toys4k/raw
# Place toys4k_blend_files.zip in the raw/ directory

cd /mnt/workspace/kuno/distillation/TRELLIS
PYTHONPATH=dataset_toolkits python dataset_toolkits/download.py Toys4k \
    --output_dir /mnt/workspace/kuno/distillation/datasets/Toys4k
```

This extracts .blend files and verifies SHA256 checksums against the metadata.

### Step 3: Update metadata after download

```bash
cd /mnt/workspace/kuno/distillation/TRELLIS
PYTHONPATH=dataset_toolkits python dataset_toolkits/build_metadata.py Toys4k \
    --output_dir /mnt/workspace/kuno/distillation/datasets/Toys4k
```

### Step 4: Render ground-truth multi-view images (150 views, 512×512)

```bash
cd /mnt/workspace/kuno/distillation/TRELLIS
PYTHONPATH=dataset_toolkits python dataset_toolkits/render.py Toys4k \
    --output_dir /mnt/workspace/kuno/distillation/datasets/Toys4k \
    --num_views 150 \
    --max_workers 8
```

This uses Blender 3.0.1 CYCLES to render 150 Hammersley-sampled views per object at 512×512, with `--save_mesh` producing triangulated PLY ground-truth meshes. Camera: r=2.0, FOV=40°.

Output structure per object:
```
datasets/Toys4k/renders/{sha256}/
    000.png - 149.png    # 150 RGBA renders
    transforms.json       # camera parameters per view
    mesh.ply             # triangulated ground-truth mesh
```

**Time estimate**: ~30s per object × 3,229 objects = ~27 hours on 8 workers. Can parallelize with `--rank`/`--world_size`.

### Step 5: Render condition images (single input views)

```bash
cd /mnt/workspace/kuno/distillation/TRELLIS
PYTHONPATH=dataset_toolkits python dataset_toolkits/render_cond.py Toys4k \
    --output_dir /mnt/workspace/kuno/distillation/datasets/Toys4k \
    --num_views 1 \
    --max_workers 8
```

**Important**: We render just 1 condition view per object (not the default 24). This single view serves as the input image for all three models. The view is randomly sampled from the Hammersley sequence with random FOV (10-70°), at 1024×1024.

**Alternative approach**: Instead of using `render_cond.py` (which uses random FOV augmentation), we could use a canonical front view from the 150 training renders (view 0 from `render.py`). This is simpler and more reproducible. The TRELLIS paper likely uses the augmented condition images since `render_cond.py` is their official conditioning pipeline.

**Recommended**: Use `render_cond.py` with `--num_views 1` to match the TRELLIS paper's conditioning augmentation, then resize to 512×512 (with background removal via rembg) for models that expect that resolution.

Output structure:
```
datasets/Toys4k/renders_cond/{sha256}/
    000.png              # 1024×1024 RGBA condition image
    transforms.json      # camera parameters
```

---

## Preprocessing

### Input image preparation per model

All three models expect RGBA input with removed background. The Blender renders already have transparent backgrounds (film_transparent=True), so rembg is not needed.

| Model | Expected Input | Preprocessing |
|-------|---------------|---------------|
| TRELLIS | RGBA, auto-resizes to 518×518 internally | Load directly from render |
| Hunyuan3D | RGBA 512×512 | Resize from 1024→512 |
| SAM-3D | RGB + binary mask | Split RGBA → RGB + alpha>128 mask |

```python
# Shared preprocessing
from PIL import Image
import numpy as np

def prepare_input(cond_path, model_name):
    img = Image.open(cond_path).convert("RGBA")

    if model_name == "trellis":
        return img  # TRELLIS handles resizing internally
    elif model_name == "hunyuan3d":
        return img.resize((512, 512), Image.LANCZOS)
    elif model_name == "sam3d":
        img_512 = img.resize((512, 512), Image.LANCZOS)
        arr = np.array(img_512)
        rgb = Image.fromarray(arr[:, :, :3])
        mask = arr[:, :, 3] > 128
        return rgb, mask
```

### Ground-truth mesh normalization

The Blender script normalizes all objects to a unit cube centered at origin. The `transforms.json` records `scale` and `offset` applied. All models also produce outputs in approximately unit-cube space, so direct comparison should be valid **without ICP alignment**.

However, different models may use different coordinate conventions:
- TRELLIS: Z-up (internal), Y-up (GLB export)
- Hunyuan3D: Y-up (GLB), Z-up (internal rendering)
- SAM-3D: Y-up (GLB)

We must normalize all predicted meshes to the same coordinate frame and scale before computing geometry metrics.

**Normalization procedure:**
1. Load predicted mesh
2. Convert to Z-up if needed (apply Y-up→Z-up rotation)
3. Center at origin
4. Scale to fit in unit cube (longest axis = 1)
5. Compare against GT mesh (already normalized by Blender script)

---

## Evaluation Metrics

### 1. Distributional Metrics (FD / KD)

**Goal**: Measure how well the distribution of generated 3D objects matches the distribution of ground-truth objects.

**Procedure**:
1. For each GT object: use the 150 rendered views (already available from Step 4)
2. For each predicted object: render 150 views using the same camera parameters from `transforms.json`
3. Extract features from all rendered images using:
   - **Inception-v3** (resize to 299×299, extract pool3 features, dim=2048)
   - **DINOv2** (resize to 224×224, extract CLS token, dim=768 or 1024)
4. Compute FD and KD between GT feature set and predicted feature set
5. Optionally compute FD_point using PointNet++ features on point clouds

**Image pool**: 150 views × 3,229 objects = 484,350 images per set (GT and per-model predicted).

**Implementation**: Use `torch-fidelity` or `clean-fid` for Inception FD. For DINOv2, use `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')` to extract features, then compute FD/KD manually.

**KD scaling**: Paper reports KD × 100.

### 2. Per-Object Appearance Metrics (PSNR / LPIPS)

**Goal**: For each object, compare rendered views of the predicted 3D output against ground-truth renders.

**Procedure**:
1. For each object, select a subset of views (e.g., 30 randomly sampled from the 150 GT views, excluding the conditioning view)
2. Render the predicted Gaussian splatting from those exact camera poses
3. Compute per-view PSNR and LPIPS, then average

**Implementation**: Use TRELLIS's `render_utils.render_frames()` with the extrinsics/intrinsics from `transforms.json`, plus `loss_utils.psnr()` and `loss_utils.lpips()`.

**Resolution**: 512×512 (matching GT renders).

### 3. Per-Object Geometry Metrics (CD / F-score)

**Goal**: Compare predicted mesh geometry against ground-truth mesh.

**Procedure**:
1. Load GT mesh (PLY from Blender render step)
2. Load predicted mesh (GLB from model inference)
3. Normalize both to unit cube centered at origin
4. Sample 10,000 points uniformly from each mesh surface
5. Compute bilateral Chamfer Distance
6. Compute F-score at τ = 0.01

**Implementation**:
```python
import trimesh
from scipy.spatial import cKDTree

def chamfer_distance(pts1, pts2):
    tree1, tree2 = cKDTree(pts1), cKDTree(pts2)
    d1, _ = tree1.query(pts2)
    d2, _ = tree2.query(pts1)
    return (d1**2).mean() + (d2**2).mean()  # squared CD

def f_score(pts1, pts2, tau=0.01):
    tree1, tree2 = cKDTree(pts1), cKDTree(pts2)
    d1, _ = tree1.query(pts2)
    d2, _ = tree2.query(pts1)
    precision = (d2 < tau).mean()
    recall = (d1 < tau).mean()
    if precision + recall == 0: return 0.0
    return 2 * precision * recall / (precision + recall)
```

### 4. Normal Quality Metrics (PSNR-N / LPIPS-N)

**Procedure**:
1. Render predicted mesh normal maps from the evaluation camera poses
2. Render GT mesh normal maps from the same poses
3. Compute PSNR and LPIPS on normal map images

This requires a normal-map renderer. TRELLIS's `MeshRenderer` renders normals by default. For Hunyuan3D and SAM-3D, we need to render normals from the exported meshes.

### 5. Efficiency Metrics

Collected during inference: runtime_sec, peak_gpu_gb per object.

---

## Pipeline Architecture

### Directory Structure

```
exp/toys4k/
    PLAN.md                         # This document
    scripts/
        prepare_dataset.sh           # Dataset acquisition
        run_trellis.py               # TRELLIS inference on all Toys4k
        run_hunyuan.py               # Hunyuan3D inference on all Toys4k
        run_sam3d.py                  # SAM-3D inference on all Toys4k
        render_predictions.py         # Render predicted outputs from saved camera poses
        compute_metrics.py            # Per-object PSNR/LPIPS/CD/F-score
        compute_distributional.py     # FD/KD computation
        aggregate_results.py          # Combine into summary tables
    configs/
        trellis_default.json
        trellis_paper.json           # 50 steps, CFG 3.0
        hunyuan_default.json
        sam3d_default.json

datasets/Toys4k/
    metadata.csv                     # From build_metadata.py
    raw/
        toys4k_blend_files.zip
        toys4k_blend_files/          # Extracted .blend files
    renders/
        {sha256}/
            000.png ... 149.png      # GT renders
            transforms.json
            mesh.ply                 # GT mesh
    renders_cond/
        {sha256}/
            000.png                  # Condition image
            transforms.json

exp/toys4k/outputs/
    trellis/
        paper_50step/
            {sha256}/
                mesh.glb
                gaussian.ply
                meta.json
        default_12step/
            {sha256}/
                mesh.glb
                gaussian.ply
                meta.json
    hunyuan3d/
        default/
            {sha256}/
                mesh.glb
                meta.json
    sam3d/
        default/
            {sha256}/
                mesh.glb
                gaussian.ply
                meta.json

exp/toys4k/eval/
    renders/
        {model}/{config}/{sha256}/
            color_000.png ... color_149.png    # Rendered predictions
            normal_000.png ... normal_149.png
    metrics/
        {model}/{config}/{sha256}/
            metrics.json
    distributional/
        features/
            gt_inception.npy
            gt_dinov2.npy
            {model}_{config}_inception.npy
            {model}_{config}_dinov2.npy
        fd_kd_results.json
    summary/
        results_table.csv
        per_object_metrics.csv
```

---

## Adaptation of Existing Benchmark Pipeline

The existing smoke test scripts (`run_trellis_smoke.py`, `run_hunyuan_smoke.py`, `run_sam3d_smoke.py`) need these changes for the Toys4k experiment:

1. **Input source**: Change from `exp/benchmark/inputs/sv1/` (10 example images) to `datasets/Toys4k/renders_cond/{sha256}/000.png` (3,229 condition images)
2. **Object ID**: Use SHA256 hash (from metadata.csv) instead of descriptive names
3. **Output directory**: `exp/toys4k/outputs/{model}/{config}/{sha256}/`
4. **Rendering from GT cameras**: Add a separate rendering pass that renders predictions from the exact GT camera poses stored in `transforms.json`
5. **Batching/parallelism**: Add `--rank` / `--world_size` for multi-GPU parallelism across the 3,229 objects
6. **TRELLIS 50-step config**: Add paper-eval config (steps=50, cfg=3.0) alongside the default (steps=12, cfg=7.5/3.0)

The rendering and grid composition code (`render_views.py`, `compose_grid.py`) are not directly needed for quantitative evaluation but may be reused for qualitative spot-checks.

---

## Step-by-Step Execution Procedure

### Phase 0: Validate Existing Setup
- Confirm all three model environments are working (already done per REPORT.md — 30/30 smoke test pass)
- Confirm GPU availability (4× L20X 144GB)

### Phase 1: Dataset Acquisition (~1 day)
1. Build metadata
2. Download `toys4k_blend_files.zip` (manual)
3. Extract and validate

### Phase 2: Ground-Truth Rendering (~1-2 days)
1. Render 150 views per object at 512×512 (with `--save_mesh`)
2. Render 1 condition view per object at 1024×1024
3. Update metadata
4. Verify: spot-check 10 random objects for correct renders

### Phase 3: Model Inference (~2-3 days)
1. Run TRELLIS on all 3,229 objects (50-step paper config + 12-step default)
2. Run Hunyuan3D on all 3,229 objects
3. Run SAM-3D on all 3,229 objects
4. Verify: check success rate, spot-check outputs

### Phase 4: Prediction Rendering (~1-2 days)
1. For each model's outputs, render 150 views from GT camera poses
2. Render normal maps from meshes

### Phase 5: Metric Computation (~0.5 day)
1. Compute per-object PSNR, LPIPS, CD, F-score
2. Extract features (Inception-v3, DINOv2)
3. Compute FD/KD

### Phase 6: Analysis
1. Aggregate results into summary tables
2. Compare against TRELLIS paper numbers
3. Cross-model comparison

---

## Open Questions / Missing Information

### Critical

1. **Toys4k download**: The `toys4k_blend_files.zip` requires manual download from the lowshot-shapebias GitHub repo. It's unclear if this is still hosted / what the download size is. **Mitigation**: Check the GitHub repo first; if unavailable, consider alternative sources or a smaller subset.

2. **Exact FD/KD protocol**: The paper does not specify:
   - How many rendered views per object are used for FD/KD (150? 20? all?)
   - At what resolution features are extracted (native 512? resized to 299 for Inception?)
   - Whether FD is computed on the full 484K image pool or per-object then averaged
   **Mitigation**: Use 150 views per object at 512×512, extract features at each model's native resolution (299 for Inception, 224 for DINOv2). Compute FD on the full pooled set (standard practice).

3. **PointNet++ FD**: The paper reports FD_point. It's unclear whether:
   - Point clouds are sampled from meshes or from depth maps
   - What PointNet++ variant and pretrained weights are used
   **Mitigation**: Sample 10K points from meshes, use PointNet++ pretrained on ShapeNet (from the `point-e` or `pytorch3d` ecosystem). If exact weights are unknown, skip FD_point or note the discrepancy.

4. **Conditioning image selection**: For image-to-3D evaluation, which view is used as the condition?
   - A random view from the 150 training renders?
   - A specially rendered augmented view from `render_cond.py`?
   **Mitigation**: Use `render_cond.py` (1 view) since that's the official conditioning pipeline. Record the view parameters for reproducibility.

5. **TRELLIS paper CFG for image-to-3D**: The paper says "CFG strength is set to 3" but doesn't distinguish between ss_cfg and slat_cfg. The default code has ss_cfg=7.5, slat_cfg=3.0. Possibly the paper uses cfg=3.0 for both.
   **Mitigation**: Run both configurations — (ss_cfg=7.5, slat_cfg=3.0) and (ss_cfg=3.0, slat_cfg=3.0) — and compare.

### Non-Critical

6. **Chamfer Distance variant**: Paper likely reports L2 CD (squared distances). Our `experiment_plan.md` code uses L1 CD (absolute distances). Need to verify which variant to match.

7. **Subset or full**: Running all 3,229 objects through 3 models is expensive. If time-constrained, a random 500-object subset would still produce statistically meaningful FD/KD. But for direct comparison with paper numbers, full set is needed.

8. **Hunyuan3D texture mode**: For appearance metrics (PSNR/LPIPS), Hunyuan3D produces UV-textured meshes while TRELLIS uses Gaussians. The paper evaluates TRELLIS Gaussians for appearance. For Hunyuan3D, we'd render the textured mesh.

9. **Blender 3.0.1 availability**: The render scripts download Blender 3.0.1. If the download link is dead, we'd need to source it elsewhere.

---

## Proposed Implementation Steps

### Step 1: Dataset preparation scripts
Create `exp/toys4k/scripts/prepare_dataset.sh` that wraps the TRELLIS dataset toolkit commands. Include verification checks.

### Step 2: Inference runner scripts
Adapt the three smoke test runners for batch Toys4k inference:
- `run_trellis.py` — iterate over metadata.csv, load condition images, run pipeline, save outputs
- `run_hunyuan.py` — same pattern
- `run_sam3d.py` — same pattern
All scripts should:
- Accept `--rank` / `--world_size` for multi-GPU sharding
- Skip already-completed objects (check for `meta.json` + `mesh.glb`)
- Log failures without stopping
- Save `meta.json` with timing and VRAM

### Step 3: Prediction rendering script
Create `render_predictions.py`:
- Load predicted Gaussians/meshes
- Load GT camera params from `transforms.json`
- Render 150 views per object using TRELLIS renderers (for TRELLIS outputs) or nvdiffrast (for Hunyuan3D/SAM-3D meshes)
- Save renders to `exp/toys4k/eval/renders/`

### Step 4: Per-object metrics script
Create `compute_metrics.py`:
- For each object: load GT renders, load predicted renders
- Compute PSNR, LPIPS per view, average
- Load GT mesh, load predicted mesh, normalize, sample points
- Compute CD, F-score
- Save to `metrics.json`

### Step 5: Distributional metrics script
Create `compute_distributional.py`:
- Load all rendered images (GT + per-model)
- Extract Inception-v3 features (batch processing)
- Extract DINOv2 features (batch processing)
- Compute FD and KD
- Save results

### Step 6: Aggregation and reporting
Create `aggregate_results.py`:
- Read all `metrics.json` files
- Produce `results_table.csv` and `per_object_metrics.csv`
- Print summary table matching TRELLIS paper Table 2 format

---

## Runbook

### Prerequisites
- TRELLIS env: `/mnt/workspace/kuno/distillation/miniconda3/envs/trellis/bin/python`
- Hunyuan3D env: `/mnt/workspace/kuno/distillation/Hunyuan3D-2/venv/bin/python`
- SAM-3D env: `/mnt/workspace/kuno/distillation/sam-3d-objects/.mamba/envs/sam3d-objects/bin/python`
- 4× NVIDIA L20X 144GB GPUs available

### 1. Create directory structure
```bash
mkdir -p /mnt/workspace/kuno/distillation/datasets/Toys4k/raw
mkdir -p /mnt/workspace/kuno/distillation/exp/toys4k/{scripts,configs}
mkdir -p /mnt/workspace/kuno/distillation/exp/toys4k/outputs/{trellis/{paper_50step,default_12step},hunyuan3d/default,sam3d/default}
mkdir -p /mnt/workspace/kuno/distillation/exp/toys4k/eval/{renders,metrics,distributional/features,summary}
```

### 2. Build metadata
```bash
cd /mnt/workspace/kuno/distillation/TRELLIS
PYTHONPATH=dataset_toolkits python dataset_toolkits/build_metadata.py Toys4k \
    --output_dir /mnt/workspace/kuno/distillation/datasets/Toys4k
```

**Verify**: `wc -l /mnt/workspace/kuno/distillation/datasets/Toys4k/metadata.csv` should show ~3,230 lines (header + 3,229 objects).

### 3. Download Toys4k blend files

```bash
# Check the source repo for download instructions:
# https://github.com/rehg-lab/lowshot-shapebias/tree/main/toys4k
#
# Download toys4k_blend_files.zip and place it at:
# /mnt/workspace/kuno/distillation/datasets/Toys4k/raw/toys4k_blend_files.zip
#
# PLACEHOLDER — exact download command depends on hosting location
```

### 4. Extract and validate
```bash
cd /mnt/workspace/kuno/distillation/TRELLIS
PYTHONPATH=dataset_toolkits python dataset_toolkits/download.py Toys4k \
    --output_dir /mnt/workspace/kuno/distillation/datasets/Toys4k

# Update metadata with local paths
PYTHONPATH=dataset_toolkits python dataset_toolkits/build_metadata.py Toys4k \
    --output_dir /mnt/workspace/kuno/distillation/datasets/Toys4k
```

**Verify**: `grep -c "True" /mnt/workspace/kuno/distillation/datasets/Toys4k/metadata.csv` — local_path column should be populated for ~3,229 rows.

### 5. Render ground-truth views (parallelized across workers)
```bash
cd /mnt/workspace/kuno/distillation/TRELLIS

# This needs Blender — will auto-install on first run
# Run with parallelism (e.g., 4 parallel processes for 4 GPUs)
for rank in 0 1 2 3; do
    PYTHONPATH=dataset_toolkits python dataset_toolkits/render.py Toys4k \
        --output_dir /mnt/workspace/kuno/distillation/datasets/Toys4k \
        --num_views 150 \
        --max_workers 8 \
        --rank $rank \
        --world_size 4 &
done
wait

# Merge rendered CSVs and update metadata
PYTHONPATH=dataset_toolkits python dataset_toolkits/build_metadata.py Toys4k \
    --output_dir /mnt/workspace/kuno/distillation/datasets/Toys4k
```

**Verify**:
```bash
# Count rendered objects
ls /mnt/workspace/kuno/distillation/datasets/Toys4k/renders/ | wc -l
# Should be ~3229

# Spot-check: each should have 150 PNGs + transforms.json + mesh.ply
ls /mnt/workspace/kuno/distillation/datasets/Toys4k/renders/$(ls /mnt/workspace/kuno/distillation/datasets/Toys4k/renders/ | head -1)/ | wc -l
# Should be 152 (150 PNGs + transforms.json + mesh.ply)
```

### 6. Render condition images
```bash
cd /mnt/workspace/kuno/distillation/TRELLIS

for rank in 0 1 2 3; do
    PYTHONPATH=dataset_toolkits python dataset_toolkits/render_cond.py Toys4k \
        --output_dir /mnt/workspace/kuno/distillation/datasets/Toys4k \
        --num_views 1 \
        --max_workers 8 \
        --rank $rank \
        --world_size 4 &
done
wait

PYTHONPATH=dataset_toolkits python dataset_toolkits/build_metadata.py Toys4k \
    --output_dir /mnt/workspace/kuno/distillation/datasets/Toys4k
```

**Verify**: Each `renders_cond/{sha256}/` should have `000.png` + `transforms.json`.

### 7. Run TRELLIS inference

```bash
# Paper config (50 steps, cfg 3.0) — GPU 0
cd /mnt/workspace/kuno/distillation/TRELLIS
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
    /mnt/workspace/kuno/distillation/miniconda3/envs/trellis/bin/python \
    /mnt/workspace/kuno/distillation/exp/toys4k/scripts/run_trellis.py \
    --config paper_50step \
    --dataset_dir /mnt/workspace/kuno/distillation/datasets/Toys4k \
    --output_dir /mnt/workspace/kuno/distillation/exp/toys4k/outputs/trellis/paper_50step \
    --rank 0 --world_size 2 &

# Paper config — GPU 1
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
    /mnt/workspace/kuno/distillation/miniconda3/envs/trellis/bin/python \
    /mnt/workspace/kuno/distillation/exp/toys4k/scripts/run_trellis.py \
    --config paper_50step \
    --dataset_dir /mnt/workspace/kuno/distillation/datasets/Toys4k \
    --output_dir /mnt/workspace/kuno/distillation/exp/toys4k/outputs/trellis/paper_50step \
    --rank 1 --world_size 2 &

wait
```

**Estimated time**: ~50s per object (50 steps + postprocessing) × 3,229 objects / 2 GPUs ≈ 22 hours.

Repeat for default 12-step config if desired.

### 8. Run Hunyuan3D inference

```bash
cd /mnt/workspace/kuno/distillation/Hunyuan3D-2
CUDA_VISIBLE_DEVICES=2 \
    /mnt/workspace/kuno/distillation/Hunyuan3D-2/venv/bin/python \
    /mnt/workspace/kuno/distillation/exp/toys4k/scripts/run_hunyuan.py \
    --dataset_dir /mnt/workspace/kuno/distillation/datasets/Toys4k \
    --output_dir /mnt/workspace/kuno/distillation/exp/toys4k/outputs/hunyuan3d/default &
```

**Estimated time**: ~62s per object × 3,229 objects ≈ 56 hours on 1 GPU (can shard across 2 GPUs to halve).

### 9. Run SAM-3D inference

```bash
cd /mnt/workspace/kuno/distillation/sam-3d-objects
CUDA_VISIBLE_DEVICES=3 \
    /mnt/workspace/kuno/distillation/sam-3d-objects/.mamba/envs/sam3d-objects/bin/python \
    /mnt/workspace/kuno/distillation/exp/toys4k/scripts/run_sam3d.py \
    --dataset_dir /mnt/workspace/kuno/distillation/datasets/Toys4k \
    --output_dir /mnt/workspace/kuno/distillation/exp/toys4k/outputs/sam3d/default &
```

**Estimated time**: ~6s per object × 3,229 objects ≈ 5.4 hours on 1 GPU.

### 10. Render predictions from GT cameras

```bash
# Run per model — uses TRELLIS renderers for Gaussians, nvdiffrast for meshes
cd /mnt/workspace/kuno/distillation/TRELLIS
for model in trellis hunyuan3d sam3d; do
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
        /mnt/workspace/kuno/distillation/miniconda3/envs/trellis/bin/python \
        /mnt/workspace/kuno/distillation/exp/toys4k/scripts/render_predictions.py \
        --model $model \
        --dataset_dir /mnt/workspace/kuno/distillation/datasets/Toys4k \
        --predictions_dir /mnt/workspace/kuno/distillation/exp/toys4k/outputs/$model/default \
        --output_dir /mnt/workspace/kuno/distillation/exp/toys4k/eval/renders/$model/default
done
```

**Note**: Rendering predictions from 150 viewpoints per object for 3,229 objects is expensive. Consider:
- Using a subset of views (e.g., 30) for per-object metrics
- Using all 150 views only for distributional metrics

### 11. Compute per-object metrics

```bash
cd /mnt/workspace/kuno/distillation/TRELLIS
PYTHONPATH=. /mnt/workspace/kuno/distillation/miniconda3/envs/trellis/bin/python \
    /mnt/workspace/kuno/distillation/exp/toys4k/scripts/compute_metrics.py \
    --gt_renders_dir /mnt/workspace/kuno/distillation/datasets/Toys4k/renders \
    --pred_renders_dir /mnt/workspace/kuno/distillation/exp/toys4k/eval/renders \
    --gt_meshes_dir /mnt/workspace/kuno/distillation/datasets/Toys4k/renders \
    --pred_meshes_dir /mnt/workspace/kuno/distillation/exp/toys4k/outputs \
    --output_dir /mnt/workspace/kuno/distillation/exp/toys4k/eval/metrics
```

### 12. Compute distributional metrics (FD / KD)

```bash
cd /mnt/workspace/kuno/distillation
# Extract features from GT renders
python exp/toys4k/scripts/compute_distributional.py \
    --gt_renders_dir datasets/Toys4k/renders \
    --pred_renders_dir exp/toys4k/eval/renders \
    --output_dir exp/toys4k/eval/distributional
```

### 13. Aggregate and compare

```bash
python exp/toys4k/scripts/aggregate_results.py \
    --metrics_dir exp/toys4k/eval/metrics \
    --distributional_dir exp/toys4k/eval/distributional \
    --output_dir exp/toys4k/eval/summary
```

**Expected output**: `results_table.csv` with columns matching TRELLIS paper Table 2.

---

## Validation Checks

### Dataset validation
- [ ] `metadata.csv` contains 3,229 rows with non-null `local_path`
- [ ] Each .blend file passes SHA256 verification
- [ ] Each rendered object has exactly 150 PNGs + transforms.json + mesh.ply
- [ ] Rendered images are 512×512 RGBA
- [ ] Condition images are 1024×1024 RGBA
- [ ] `transforms.json` contains valid camera matrices

### Inference validation
- [ ] Each model produces output for ≥95% of objects (some may fail on degenerate geometries)
- [ ] `meta.json` present for every attempted object
- [ ] `mesh.glb` loads in trimesh with >0 vertices
- [ ] TRELLIS: `gaussian.ply` present alongside mesh.glb
- [ ] No OOM errors (L20X has 144GB — should be fine)
- [ ] Seed=42 set before each inference call

### Metric validation
- [ ] Per-object PSNR values in reasonable range (15-35 dB)
- [ ] LPIPS values in [0, 1] range
- [ ] Chamfer Distance > 0 for all objects
- [ ] F-score in [0, 1] range
- [ ] FD values are positive
- [ ] Spot-check: high-PSNR objects should look visually similar to GT

### Cross-check with paper
- [ ] TRELLIS FD/KD values should be in same order of magnitude as paper Table 2
- [ ] If significantly different, investigate rendering or evaluation differences

---

## Possible Failure Points and Mitigations

| Failure | Likelihood | Impact | Mitigation |
|---------|-----------|--------|------------|
| `toys4k_blend_files.zip` unavailable | Medium | Blocking | Check GitHub release page; contact authors; find mirror |
| Blender 3.0.1 download fails | Low | Blocking | Use local Blender install; update `BLENDER_LINK` in render.py |
| Blender crashes on certain .blend files | Medium | Per-object loss | Skip failed objects; report failure rate |
| TRELLIS OOM on some objects | Low | Per-object loss | L20X has 144GB; unlikely. If occurs, reduce batch/flush cache |
| Hunyuan3D slow on 3,229 objects | High | Schedule | ~56h on 1 GPU; shard across 2 GPUs → 28h |
| nvdiffrast rendering failures | Medium | Rendering quality | Fall back to pyrender/trimesh for affected objects |
| Feature extraction OOM (484K images) | Medium | Metric computation | Process in batches (1000 images at a time) |
| Coordinate frame mismatch | High | Wrong metrics | Verify alignment on 5 random objects visually before full run |
| Scale mismatch between predicted/GT | High | Wrong CD/F-score | Normalize both to unit cube before comparison |
| Random seed in `render_cond.py` | Medium | Non-reproducibility | Fix numpy seed before calling `render_cond.py`, or record the exact condition view parameters |

---

## Distinguishing Facts, Inferences, and Recommendations

### Facts (verified from repo or paper)
- Toys4k has 3,229 objects, used as eval set in TRELLIS paper
- TRELLIS paper uses 50 steps, CFG 3.0 for evaluation
- GT rendering uses Blender CYCLES, 512×512, r=2.0, FOV=40°, 150 views
- FD computed with Inception-v3, DINOv2, PointNet++ feature extractors
- Toys4k requires manual download of .blend files
- All objects normalized to unit cube by Blender script

### Inferences (reasonable deductions)
- 150 views per object used for FD/KD (matches training render count)
- Evaluation at 512×512 resolution (consistent across all scripts)
- No ICP alignment before geometry metrics (paper doesn't mention it; objects are pre-normalized)
- 10,000 point samples for CD/F-score (community standard)

### Recommendations (our choices)
- Run TRELLIS at both 12-step (interactive default) and 50-step (paper eval) for completeness
- Use `render_cond.py` for conditioning images to match TRELLIS paper's augmented conditioning
- Use full 3,229-object set for publishable numbers; subset of 500 for debugging
- Start with Inception FD and DINOv2 FD; add PointNet++ FD only if time permits
- Render 30 views for per-object metrics, 150 for distributional metrics (efficiency compromise)
- Fix numpy random seed in render_cond.py to ensure reproducible condition views
