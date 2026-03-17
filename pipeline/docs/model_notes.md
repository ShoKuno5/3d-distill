# Model Notes

## Models Overview

| Model | Version | Family | Notes |
|-------|---------|--------|-------|
| trellis | 1.0 | TRELLIS (Microsoft) | GPU 0 |
| trellis2 | 2.0 | TRELLIS.2 (Microsoft) | GPU 2 |
| hunyuan3d | 2.0 | Hunyuan3D (Tencent) | GPU 1 |
| hunyuan3d21 | 2.1 | Hunyuan3D (Tencent) | GPU 3 |

---

## TRELLIS (v1.0) — Existing

- **Inference entry point**: `trellis.pipelines.TrellisImageTo3DPipeline`
- **Model ID**: `microsoft/TRELLIS-image-large`
- **Input**: RGBA image (any resolution, internally processed)
- **Output**: raw mesh (vertices + faces from SLAT decoder), Gaussian splatting
- **Raw mesh saved as**: `mesh_raw.obj` (before `to_glb` simplification)
- **Seed control**: Yes, via `seed=` parameter
- **GPU/VRAM**: ~10 GB peak (L20X)
- **Environment**: `envs/miniconda3/envs/trellis/bin/python`
- **Run from**: `cd models/trellis && PYTHONPATH=.`
- **Config**: 12 ss_steps, 12 slat_steps, cfg 7.5/3.0
- **Coordinate system**: Model-native (z-up). `to_glb()` applies z-up→y-up rotation.
  Raw mesh is in model-native frame.
- **Deterministic**: Yes with fixed seed + cudnn.deterministic=True
- **Avg runtime**: ~16s per sample
- **Architecture**: SLAT (Structured Latent) — 8-channel sparse voxels, Rectified Flow Transformers

## TRELLIS.2 (v2.0) — New

- **Inference entry point**: `trellis2.pipelines.Trellis2ImageTo3DPipeline`
- **Model ID**: `microsoft/TRELLIS.2-4B`
- **Input**: RGBA image (internally preprocessed with rembg)
- **Output**: `MeshWithVoxel` (vertices + faces as torch tensors, plus O-Voxel data)
- **Raw mesh saved as**: `mesh_raw.obj` (vertices/faces extracted before any post-processing)
- **Seed control**: Yes, via `seed=` parameter in `run()`
- **GPU/VRAM**: ~24 GB+ (recommended); pipeline_type affects memory
- **Environment**: `envs/miniconda3/envs/trellis2/bin/python`
- **Run from**: `cd models/trellis2`
- **Pipeline types**: `512` (fast, lower res), `1024` (high res), `1024_cascade` (default, best quality)
- **Coordinate system**: Model-native
- **Deterministic**: Yes with `torch.manual_seed(seed)` called internally
- **Architecture**: 4B-parameter model, O-Voxel (field-free sparse voxel), vanilla DiTs, 16× spatial downsampling VAE
- **Key deps**: `cumesh`, `o_voxel`, `flex_gemm`, `nvdiffrast`, `nvdiffrec`, `flash-attn`
- **Differences from TRELLIS v1**:
  - 4B vs ~1B params
  - O-Voxel vs SLAT representation
  - Supports open surfaces, non-manifold geometry
  - PBR materials (base color, roughness, metallic, opacity)
  - Higher resolution output (512³/1024³/1536³)
  - Different pipeline module path (`trellis2` vs `trellis`)

## Hunyuan3D-2 (v2.0) — Existing

- **Inference entry point**: `hy3dgen.shapegen.Hunyuan3DDiTFlowMatchingPipeline`
- **Model ID**: `tencent/Hunyuan3D-2`
- **Input**: RGBA image (internally resized)
- **Output**: trimesh.Trimesh (from octree marching cubes)
- **Raw mesh saved as**: `mesh_raw.obj` (before FaceReducer)
- **Seed control**: Yes, via `generator=torch.manual_seed(seed)`
- **GPU/VRAM**: variable, can be high
- **Environment**: `envs/hunyuan3d-venv/bin/python`
- **Run from**: `cd models/hunyuan3d`
- **Config**: 50 inference steps, guidance_scale 5.0, octree_resolution 384
- **Coordinate system**: y-up by default
- **Known issue**: Needs `LD_PRELOAD=/opt/unreal-engine/usr/lib/x86_64-linux-gnu/libOpenGL.so.0`
- **Deterministic**: Approximately (seed-controlled diffusion)
- **Avg runtime**: ~42s per sample

## Hunyuan3D-2.1 (v2.1) — New

- **Inference entry point**: `hy3dshape.pipelines.Hunyuan3DDiTFlowMatchingPipeline`
- **Model ID**: `tencent/Hunyuan3D-2.1`
- **Input**: RGBA image (internally preprocessed; has built-in rembg)
- **Output**: trimesh.Trimesh (from octree marching cubes)
- **Raw mesh saved as**: `mesh_raw.obj` (exported via trimesh for consistent eval)
- **Seed control**: Yes, via `generator=torch.manual_seed(seed)`
- **GPU/VRAM**: ~10 GB shape, ~21 GB texture, ~29 GB total (shape-only for our eval)
- **Environment**: Reuses `envs/hunyuan3d-venv/bin/python`
  (compatible with existing H3D-2 venv; needed `timm` installed additionally)
- **Run from**: `cd models/hunyuan3d21` with `PYTHONPATH=.` and `sys.path.insert('./hy3dshape')`
- **Coordinate system**: Same as v2.0 (y-up)
- **Deterministic**: Approximately (seed-controlled diffusion)
- **Differences from Hunyuan3D v2.0**:
  - Module path: `hy3dshape.pipelines` (not `hy3dgen.shapegen`)
  - Model: 3.3B params for shape (vs unspecified for v2.0)
  - Uses MoE (Mixture of Experts) blocks
  - PBR texture synthesis (not used in geometry-only eval)
  - Default output format: GLB (we extract raw trimesh for consistency)
  - Additional dependency: `timm`
- **Known issue**: Same LD_PRELOAD requirement as v2.0

## SAM-3D-Objects (Meta) — Excluded

SAM-3D-Objects has been removed from the active evaluation pipeline (2026-03-12).
The model repo and environment remain in `models/sam3d/` and `envs/sam3d-mamba/` for reference.
Previous predictions are archived in `results/toys4k/predictions/sam3d/`.

## Common Settings

- **Seed**: 42 (shared across all models)
- **Raw mesh format**: OBJ (vertex positions + face indices only)
- **Input images**: Toys4k blend renders (300×300 PNG, RGB)
  - Converted to RGBA for TRELLIS, TRELLIS.2, Hunyuan3D, and Hunyuan3D-2.1

## Fairness-Relevant Differences Between Old/New Versions

### TRELLIS v1 vs TRELLIS.2
- Different architectures (SLAT vs O-Voxel) — not directly comparable as "same model"
- TRELLIS.2 is 4× larger (4B vs ~1B params)
- TRELLIS.2 supports higher output resolution
- We use `pipeline_type=512` for TRELLIS.2 subset testing (fastest); full eval may use `1024_cascade`
- Output mesh characteristics (topology, vertex count, watertightness) may differ fundamentally

### Hunyuan3D v2.0 vs v2.1
- Same architectural family (DiT + Flow Matching + Octree VAE)
- v2.1 adds MoE blocks and is 3.3B params
- v2.1 was trained with different data/recipe
- API is the same: `pipeline(image=..., generator=...)[0]`
- Same output format (trimesh.Trimesh)
- Differences in quality should reflect genuine model improvements

## Known Issues

- TRELLIS chair_156: poor alignment (CD=0.17), likely topology mismatch
- TRELLIS banana_028: poor match (CD=0.14)
- Hunyuan3D 2.1: banana_028, screwdriver_016, plate_005 have scale/topology issues (alignment scale 0.25-0.59)
- Hunyuan3D 2.0 generally produces the highest quality geometry across samples
