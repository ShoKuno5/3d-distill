#!/usr/bin/env python
"""Sharded multiview renderer for the eval pipeline (FD + CLIP-I inputs).

run_eval.py renders GT + per-model multiview images sequentially in a single
process. For the 1050-sample cross-family benchmark that is ~7k meshes, far too
slow for one GPU. This driver does the SAME rendering but shardable across many
GPUs/nodes: it builds the full list of (mesh -> render-dir) units, slices it by
`samples[shard_index::num_shards]`, and renders its slice. Pin each shard to one
GPU via CUDA_VISIBLE_DEVICES; Blender uses that device.

It reuses the exact config/manifest/prediction resolution and the per-mesh
render_multiview() from the eval code, so the output layout matches what
run_eval.py's FD / CLIP-I steps consume:
    <output_root>/multiview_renders/gt/<oid>/view_<az>.png
    <output_root>/multiview_renders/<model>/<oid>/view_<az>.png

Example (one shard on GPU 0 of 16):
    CUDA_VISIBLE_DEVICES=0 python render_eval_multiview.py \
        --config config_tsubame_1050_eval.yaml --shard-index 0 --num-shards 16
"""
import argparse
import os
import sys
import time
import yaml

from src.data.toys4k import load_manifest, filter_samples
from src.utils.inference_config import resolve_model_paths
from src.evaluation.multiview_renderer import render_multiview, AZIMUTHS, ELEVATION, RESOLUTION


def main():
    parser = argparse.ArgumentParser(description="Sharded eval multiview renderer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Override model list (default: all in config)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0,
                        help="This worker's shard index in [0, num_shards).")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of shards (workers). Default 1 = no sharding.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Render at most this many units in this shard (for smoke tests).")
    parser.add_argument("--skip-gt", action="store_true",
                        help="Skip GT renders (only needed by FD, not CLIP-I).")
    args = parser.parse_args()

    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        print(f"ERROR: invalid sharding: shard_index={args.shard_index}, num_shards={args.num_shards}")
        sys.exit(1)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    resolve_model_paths(cfg)

    output_root = cfg["output_root"]
    renders_dir = os.path.join(output_root, "multiview_renders")

    # Multiview params (same source run_eval reads).
    mv_cfg = cfg.get("metrics", {}).get("frechet_distance", {}).get("multiview", {})
    azimuths = mv_cfg.get("azimuths", list(AZIMUTHS))
    elevation = mv_cfg.get("elevation", ELEVATION)
    resolution = mv_cfg.get("resolution", RESOLUTION)

    # Samples (same loading as run_eval).
    manifest_path = cfg["dataset"].get("test_manifest") or cfg["dataset"]["manifest"]
    max_samples = args.max_samples or cfg["dataset"].get("max_samples")
    samples = load_manifest(manifest_path)
    samples = filter_samples(
        samples,
        max_samples=max_samples,
        sample_ids_file=cfg["dataset"].get("sample_ids_file"),
        category_filter=cfg["dataset"].get("category_filter"),
    )

    model_cfgs = cfg["models"]
    if args.models:
        model_cfgs = [m for m in model_cfgs if m["name"] in args.models]

    # Build the full, deterministically-ordered list of render units so every
    # shard sees the same order and the [shard::num_shards] slices are disjoint.
    # unit = (label, mesh_path, out_dir)
    units = []
    if not args.skip_gt:
        for s in sorted(samples, key=lambda s: s.object_id):
            units.append(("gt", s.mesh_obj, os.path.join(renders_dir, "gt", s.object_id)))
    for mcfg in sorted(model_cfgs, key=lambda m: m["name"]):
        name = mcfg["name"]
        pred_root = mcfg["predictions_root"]
        mesh_fn = mcfg.get("mesh_filename", "mesh_raw.obj")
        for s in sorted(samples, key=lambda s: s.object_id):
            mesh_path = os.path.join(pred_root, s.object_id, mesh_fn)
            units.append((name, mesh_path, os.path.join(renders_dir, name, s.object_id)))

    total = len(units)
    shard_units = units[args.shard_index::args.num_shards]
    if args.limit is not None:
        shard_units = shard_units[:args.limit]

    print(f"[shard {args.shard_index}/{args.num_shards}] {len(shard_units)}/{total} units "
          f"(azimuths={azimuths}, elevation={elevation}, resolution={resolution})",
          flush=True)

    n_ok = n_skip_missing = n_done = n_fail = 0
    t0 = time.time()
    for i, (label, mesh_path, out_dir) in enumerate(shard_units):
        if not os.path.exists(mesh_path):
            n_skip_missing += 1
            continue
        # Skip if all views already exist (render_multiview also checks, but this
        # avoids the makedirs/normalize call entirely).
        if all(os.path.exists(os.path.join(out_dir, f"view_{int(az)}.png")) for az in azimuths):
            n_done += 1
            continue
        try:
            paths = render_multiview(mesh_path, out_dir,
                                     resolution=resolution, azimuths=azimuths, elevation=elevation)
            if len(paths) == len(azimuths):
                n_ok += 1
            else:
                n_fail += 1
                print(f"  [{label}] {os.path.basename(out_dir)}: {len(paths)}/{len(azimuths)} views", flush=True)
        except Exception as e:
            n_fail += 1
            print(f"  ERROR [{label}] {mesh_path}: {e}", flush=True)
        if (i + 1) % 50 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  [shard {args.shard_index}] {i+1}/{len(shard_units)} "
                  f"({rate:.2f} units/s, ok={n_ok} cached={n_done} miss={n_skip_missing} fail={n_fail})",
                  flush=True)

    dt = time.time() - t0
    print(f"[shard {args.shard_index}/{args.num_shards}] done in {dt:.0f}s: "
          f"rendered={n_ok} cached={n_done} missing_mesh={n_skip_missing} failed={n_fail}",
          flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
