#!/usr/bin/env python3
"""Create train/test manifests for distillation experiment.

Stratified sampling: select ~5 objects per category (target ~500 total),
then split 80/20 within each category for train/test.

Only includes objects that have both:
  - mesh.obj in toys4k_obj_files/
  - pc10K.npz in toys4k_point_clouds/

Usage:
    python distill_methods/scripts/create_manifests.py \
        --target-samples 500 --seed 42
"""

import argparse
import csv
import os
import random
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_DIR / "datasets" / "Toys4k" / "official"
OBJ_DIR = DATASET_ROOT / "toys4k_obj_files"
PC_DIR = DATASET_ROOT / "toys4k_point_clouds"
BLEND_DIR = DATASET_ROOT / "toys4k_blend_files"
RENDERS_DIR = PROJECT_DIR / "datasets" / "Toys4k" / "renders" / "512"
OUTPUT_DIR = PROJECT_DIR / "distill_methods"


def find_valid_objects():
    """Find all objects that have both mesh.obj and pc10K.npz."""
    categories = {}
    for cat_dir in sorted(OBJ_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat = cat_dir.name
        objects = []
        for obj_dir in sorted(cat_dir.iterdir()):
            if not obj_dir.is_dir():
                continue
            oid = obj_dir.name
            mesh_path = obj_dir / "mesh.obj"
            pc_path = PC_DIR / cat / oid / "pc10K.npz"
            if mesh_path.exists() and pc_path.exists():
                objects.append(oid)
        if objects:
            categories[cat] = objects
    return categories


def stratified_sample(categories, target_total, seed):
    """Select ~target_total objects via stratified sampling across categories."""
    rng = random.Random(seed)
    n_cats = len(categories)
    per_cat = max(1, target_total // n_cats)

    selected = {}
    for cat, objects in sorted(categories.items()):
        if len(objects) <= per_cat:
            selected[cat] = list(objects)
        else:
            selected[cat] = sorted(rng.sample(objects, per_cat))

    total = sum(len(v) for v in selected.values())
    print(f"Stratified sampling: {n_cats} categories, {per_cat}/cat target, {total} total selected")
    return selected


def train_test_split(selected, seed):
    """Split each category 80/20 into train/test."""
    rng = random.Random(seed + 1)
    train, test = [], []
    for cat, objects in sorted(selected.items()):
        shuffled = list(objects)
        rng.shuffle(shuffled)
        n_test = max(1, len(shuffled) // 5)
        test.extend((cat, oid) for oid in shuffled[:n_test])
        train.extend((cat, oid) for oid in shuffled[n_test:])
    return train, test


def make_row(cat, oid):
    """Create a manifest row dict."""
    return {
        "object_id": oid,
        "category": cat,
        "mesh_obj": str(OBJ_DIR / cat / oid / "mesh.obj"),
        "point_cloud": str(PC_DIR / cat / oid / "pc10K.npz"),
        "input_image": str(RENDERS_DIR / cat / oid / "image.png"),
    }


def write_manifest(rows, path):
    """Write manifest CSV."""
    fieldnames = ["object_id", "category", "mesh_obj", "point_cloud", "input_image"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cat, oid in sorted(rows):
            writer.writerow(make_row(cat, oid))
    print(f"  Written: {path} ({len(rows)} samples)")


def main():
    parser = argparse.ArgumentParser(description="Create train/test manifests")
    parser.add_argument("--target-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    categories = find_valid_objects()
    total_valid = sum(len(v) for v in categories.values())
    print(f"Found {len(categories)} categories, {total_valid} valid objects")

    selected = stratified_sample(categories, args.target_samples, args.seed)

    train, test = train_test_split(selected, args.seed)
    print(f"Split: {len(train)} train, {len(test)} test")

    write_manifest(train, OUTPUT_DIR / "manifest_train.csv")
    write_manifest(test, OUTPUT_DIR / "manifest_test.csv")

    # Category breakdown
    train_cats = {}
    for cat, oid in train:
        train_cats[cat] = train_cats.get(cat, 0) + 1
    test_cats = {}
    for cat, oid in test:
        test_cats[cat] = test_cats.get(cat, 0) + 1
    print(f"\nPer-category breakdown (first 10):")
    for cat in sorted(train_cats.keys())[:10]:
        print(f"  {cat}: {train_cats.get(cat, 0)} train, {test_cats.get(cat, 0)} test")


if __name__ == "__main__":
    main()
