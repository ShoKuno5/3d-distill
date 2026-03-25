#!/usr/bin/env python3
"""Generate a stratified 200-sample manifest for the distill_comparison experiment.

Scans Toys4k official directory for objects with valid OBJ + point cloud files,
then selects a stratified subset across categories.

Usage:
    python experiments/distill_comparison/scripts/generate_manifest.py \
        --n-samples 200 \
        --output experiments/distill_comparison/manifest.csv
"""

import argparse
import csv
import os
import random
from collections import defaultdict
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[3]
TOYS4K_ROOT = PROJECT_DIR / "datasets" / "Toys4k"
OFFICIAL_DIR = TOYS4K_ROOT / "official"
RENDERS_DIR = TOYS4K_ROOT / "renders" / "512"

OBJ_DIR = OFFICIAL_DIR / "toys4k_obj_files"
PC_DIR = OFFICIAL_DIR / "toys4k_point_clouds"
BLEND_DIR = OFFICIAL_DIR / "toys4k_blend_files"


def find_valid_objects():
    """Find all objects with valid OBJ mesh and point cloud files."""
    objects_by_category = defaultdict(list)

    if not OBJ_DIR.exists():
        raise FileNotFoundError(f"OBJ directory not found: {OBJ_DIR}")

    for category_dir in sorted(OBJ_DIR.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name

        for obj_dir in sorted(category_dir.iterdir()):
            if not obj_dir.is_dir():
                continue
            object_id = obj_dir.name

            # Check for OBJ mesh
            mesh_path = obj_dir / "mesh.obj"
            if not mesh_path.exists():
                continue

            # Check for point cloud
            pc_path = PC_DIR / category / object_id / "pc10K.npz"
            if not pc_path.exists():
                continue

            # Check for blend file (needed for rendering)
            blend_path = BLEND_DIR / category / object_id / f"{object_id}.blend"
            if not blend_path.exists():
                continue

            # Check for existing 512px render (optional, noted for stats)
            render_path = RENDERS_DIR / category / object_id / "image.png"
            has_render = render_path.exists()

            objects_by_category[category].append({
                "object_id": object_id,
                "category": category,
                "mesh_obj": str(mesh_path),
                "point_cloud": str(pc_path),
                "blend_file": str(blend_path),
                "has_render": has_render,
            })

    return objects_by_category


def stratified_sample(objects_by_category, n_samples, seed=42):
    """Select n_samples objects, stratified across categories."""
    rng = random.Random(seed)

    categories = sorted(objects_by_category.keys())
    n_categories = len(categories)
    total_available = sum(len(v) for v in objects_by_category.values())

    print(f"Found {total_available} valid objects across {n_categories} categories")

    if total_available == 0:
        print("ERROR: No valid objects found")
        return []

    if n_samples > total_available:
        print(f"WARNING: Requested {n_samples} but only {total_available} available")
        n_samples = total_available

    # Base allocation: floor(n_samples / n_categories) per category
    base_per_cat = n_samples // n_categories
    remainder = n_samples - base_per_cat * n_categories

    # Distribute remainder to categories with most objects
    cat_sizes = [(len(objects_by_category[c]), c) for c in categories]
    cat_sizes.sort(reverse=True)
    extra_cats = {c for _, c in cat_sizes[:remainder]}

    selected = []
    for category in categories:
        pool = list(objects_by_category[category])
        rng.shuffle(pool)
        n_take = base_per_cat + (1 if category in extra_cats else 0)
        n_take = min(n_take, len(pool))
        selected.extend(pool[:n_take])

    # Sort by object_id for reproducibility
    selected.sort(key=lambda x: x["object_id"])
    return selected


def write_manifest(selected, output_path):
    """Write manifest CSV with paths using the official directory layout."""
    fieldnames = ["object_id", "category", "mesh_obj", "point_cloud", "input_image"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for obj in selected:
            # Input image: use 512px render path (will be rendered if missing)
            input_image = str(RENDERS_DIR / obj["category"] / obj["object_id"] / "image.png")
            writer.writerow({
                "object_id": obj["object_id"],
                "category": obj["category"],
                "mesh_obj": obj["mesh_obj"],
                "point_cloud": obj["point_cloud"],
                "input_image": input_image,
            })

    print(f"Wrote {len(selected)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate stratified Toys4k manifest")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(
        PROJECT_DIR / "experiments" / "distill_comparison" / "manifest.csv"
    ))
    args = parser.parse_args()

    objects_by_category = find_valid_objects()

    selected = stratified_sample(objects_by_category, args.n_samples, seed=args.seed)

    # Stats
    n_with_render = sum(1 for s in selected if s["has_render"])
    n_missing_render = len(selected) - n_with_render
    cats_used = len(set(s["category"] for s in selected))
    print(f"Selected: {len(selected)} samples from {cats_used} categories")
    print(f"  With existing 512px render: {n_with_render}")
    print(f"  Missing render (will be generated): {n_missing_render}")

    write_manifest(selected, args.output)


if __name__ == "__main__":
    main()
