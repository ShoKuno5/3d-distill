# Rendering Toys4k Objects

## Overview

Toys4k `.blend` files contain meshes with materials/textures but **no camera or lights**.
Rendering requires setting up the scene from scratch, following the official Toys4k
renderer approach ([rehg-lab/lowshot-shapebias](https://github.com/rehg-lab/lowshot-shapebias/tree/main/toys4k/renderer)).

## Official Renderer Approach

The official renderer (`generate.py` + `utils.py`) works as follows:

1. Open `empty_scene.blend` (contains a pre-configured camera)
2. Import objects from `.blend` via `bpy.data.libraries.load(path, link=False)` — **not** `bpy.ops.wm.open_mainfile` (which replaces the entire scene)
3. Rotate object -90° on X axis, center at origin via `ORIGIN_GEOMETRY`
4. Rescale to fit in unit cube (factor 0.45)
5. Apply viewpoint rotation to the **object** (not camera): azimuth on Y, elevation on X
6. Camera stays fixed at `(0, 0, 2.2)` looking along -Z
7. Add area light at `(0, 0, 5)`, size 10×10, strength 30, color temp 6000K

### Official Parameters (from `data_generation_parameters.json`)

| Parameter | Value |
|-----------|-------|
| Camera focal length | 50 mm |
| Camera sensor width | 32 mm |
| Camera distance | 2.2 units |
| Default azimuth | 315° |
| Default elevation | 45° |
| Render samples | 10 |
| Denoising | enabled (radius 5) |
| Bounces (max/min) | 2 |
| Transparency bounces | 2 |
| Caustics | disabled |
| Film transparency | enabled |
| Color mode | RGBA |
| World background | white, strength 0.25 |

## Our Implementation

`pipeline/scripts/render_blender_hires.py` reproduces the official approach:

- Uses `bpy.data.libraries.load` to import objects (preserves materials)
- Applies the full official transform chain (rotate, center, scale, viewpoint)
- Cycles GPU rendering with 10 samples + denoising for speed (~4 sec/object)
- Supports manifest-driven mode (reads `input_image` column as output path)

### Usage

```bash
envs/blender-3.6.16-linux-x64/blender --background \
  --python pipeline/scripts/render_blender_hires.py -- \
  --manifest experiments/category_pilot/manifest.csv \
  --blend-root datasets/Toys4k/official/toys4k_blend_files \
  --use-manifest-paths --resolution 512
```

### Parallel Rendering

Blender is single-threaded per invocation. For parallelism, launch multiple
Blender processes with GNU parallel or xargs:

```bash
cat jobs.txt | xargs -P 8 -I {} bash -c '{}'
```

### Performance

| Engine | Samples | Denoising | Time/object | Notes |
|--------|---------|-----------|-------------|-------|
| Cycles GPU | 10 | yes | ~4 sec | Official settings, good quality |
| Cycles CPU | 64 | no | ~6 min | `_blender_render_mesh.py` default (OBJ only) |

## Key Mistakes to Avoid

- **Do NOT use `bpy.ops.wm.open_mainfile`** to open `.blend` files for rendering.
  This replaces the entire scene. Toys4k `.blend` files have no camera/lights,
  resulting in "Cannot render, no camera" errors or black images.
- **Do NOT forget lighting.** Even with a camera, the scene will be black without lights.
- **The camera is fixed; the object rotates.** The official renderer applies azimuth/elevation
  as rotations on the object, not as camera orbit.

## Output Format

Renders are stored at `datasets/Toys4k/renders/<resolution>/<category>/<oid>/`:

```
renders/512/apple/apple_048/
├── image.png          # RGBA render with transparent background
├── depth.png          # Normalized depth (optional)
└── segmentation.png   # Binary silhouette mask (optional)
```

Only `image.png` is required for the evaluation pipeline (model input).
`depth.png` and `segmentation.png` are for reference/visualization only.

## Blender Versions

- **Blender 3.6.16** (`envs/blender-3.6.16-linux-x64/`): Used for all rendering
  (`.blend` input images via `_blender_render_multipass.py`, OBJ mesh visualization
  via `_blender_render_mesh.py`). GPU Cycles + denoising for fast rendering (~1-2s/obj).
- **Blender 2.80** (`envs/blender-2.80-linux-glibc217-x86_64/`): Legacy, no longer used.
