#!/usr/bin/env python3
"""Render high-resolution RGBA images from Toys4k .blend files.

Run with Blender in headless mode:
    envs/blender-3.6.16-linux-x64/blender --background --python pipeline/scripts/render_blender_hires.py -- \
        --manifest datasets/Toys4k/subset.csv \
        --blend-root datasets/Toys4k/toys4k_blend_files \
        --resolutions 512,1024 \
        --output-root datasets/Toys4k
"""

import argparse
import csv
import os
import sys


def parse_args():
    # Blender passes everything after '--' to the script
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Render hi-res images from .blend files")
    parser.add_argument("--manifest", required=True, help="Path to subset.csv")
    parser.add_argument("--blend-root", required=True, help="Root dir of .blend files")
    parser.add_argument("--resolutions", default="512,1024", help="Comma-separated resolutions")
    parser.add_argument("--output-root", default=None, help="Output root (default: same as manifest dir)")
    parser.add_argument("--engine", default=None, help="Render engine override: CYCLES or BLENDER_EEVEE (default: use .blend setting)")
    parser.add_argument("--samples", type=int, default=None, help="Override render samples (Cycles only)")
    return parser.parse_args(argv)


def find_blend_file(blend_root, object_id, category):
    """Find the .blend file for an object. Try several naming conventions."""
    candidates = [
        os.path.join(blend_root, category, f"{object_id}.blend"),
        os.path.join(blend_root, f"{object_id}.blend"),
        os.path.join(blend_root, category, object_id, f"{object_id}.blend"),
        os.path.join(blend_root, category, object_id, "scene.blend"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def render_object(blend_path, object_id, resolution, output_path, engine=None, samples=None):
    """Open a .blend file and render at the given resolution."""
    import bpy

    # Open the .blend file
    bpy.ops.wm.open_mainfile(filepath=blend_path)
    scene = bpy.context.scene

    # Set resolution
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100

    # Transparent background (RGBA)
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"

    # Engine override
    if engine:
        scene.render.engine = engine

    # Cycles sample override
    if samples and scene.render.engine == "CYCLES":
        scene.cycles.samples = samples

    # For Cycles, use GPU if available, else CPU
    if scene.render.engine == "CYCLES":
        prefs = bpy.context.preferences.addons.get("cycles")
        if prefs:
            prefs.preferences.compute_device_type = "CUDA"
            prefs.preferences.get_devices()
            for device in prefs.preferences.devices:
                device.use = True
            scene.cycles.device = "GPU"

    # Output path
    scene.render.filepath = output_path

    # Render
    print(f"  Rendering {object_id} at {resolution}x{resolution} -> {output_path}")
    bpy.ops.render.render(write_still=True)
    print(f"  Done: {output_path}")


def main():
    args = parse_args()

    # Read manifest
    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))

    resolutions = [int(r) for r in args.resolutions.split(",")]
    output_root = args.output_root or os.path.dirname(args.manifest)

    print(f"Manifest: {args.manifest} ({len(rows)} objects)")
    print(f"Blend root: {args.blend_root}")
    print(f"Resolutions: {resolutions}")
    print(f"Output root: {output_root}")
    print()

    # Check all .blend files exist before rendering
    missing = []
    blend_map = {}
    for row in rows:
        oid = row["object_id"]
        cat = row["category"]
        bp = find_blend_file(args.blend_root, oid, cat)
        if bp is None:
            missing.append(oid)
        else:
            blend_map[oid] = bp

    if missing:
        print(f"WARNING: Missing .blend files for {len(missing)} objects: {missing}")
        print("These will be skipped.")
    print(f"Found .blend files for {len(blend_map)}/{len(rows)} objects.")
    print()

    # Render each resolution × each object
    for res in resolutions:
        out_dir = os.path.join(output_root, f"toys_blend_renders_{res}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"=== Resolution {res}x{res} -> {out_dir} ===")

        for row in rows:
            oid = row["object_id"]
            if oid not in blend_map:
                continue

            out_path = os.path.join(out_dir, f"{oid}_img.png")
            if os.path.exists(out_path):
                print(f"  Skipping {oid} (already exists)")
                continue

            try:
                render_object(
                    blend_map[oid], oid, res, out_path,
                    engine=args.engine, samples=args.samples,
                )
            except Exception as e:
                print(f"  ERROR rendering {oid}: {e}")

    print()
    print("All renders complete.")


if __name__ == "__main__":
    main()
