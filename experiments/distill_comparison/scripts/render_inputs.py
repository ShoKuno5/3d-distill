#!/usr/bin/env python3
"""Render missing 512px input images for the distill_comparison experiment.

Reads the manifest and renders any objects that don't have existing 512px renders
using the standard Blender multipass renderer.

Usage:
    python experiments/distill_comparison/scripts/render_inputs.py \
        --manifest experiments/distill_comparison/manifest.csv
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[3]
BLENDER_SCRIPT = PROJECT_DIR / "pipeline" / "scripts" / "_blender_render_multipass.py"
TOYS4K_ROOT = PROJECT_DIR / "datasets" / "Toys4k"
BLEND_DIR = TOYS4K_ROOT / "official" / "toys4k_blend_files"
RENDERS_DIR = TOYS4K_ROOT / "renders" / "512"

# Try common Blender locations
BLENDER_CANDIDATES = [
    str(PROJECT_DIR / "envs" / "blender-3.6.16-linux-x64" / "blender"),
    "blender",
    "/usr/bin/blender",
    "/opt/blender/blender",
    "/snap/bin/blender",
]


def find_blender():
    """Find a working Blender binary."""
    for candidate in BLENDER_CANDIDATES:
        try:
            result = subprocess.run(
                [candidate, "--version"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                version = result.stdout.strip().split("\n")[0]
                print(f"Found Blender: {candidate} ({version})")
                return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def render_object(blender_bin, category, object_id):
    """Render a single object using the Blender multipass script."""
    blend_path = BLEND_DIR / category / object_id / f"{object_id}.blend"
    output_dir = RENDERS_DIR / category / object_id

    if not blend_path.exists():
        print(f"  SKIP {object_id}: blend file not found at {blend_path}")
        return False

    cmd = [
        blender_bin, "--background", "--python", str(BLENDER_SCRIPT),
        "--", "--input", str(blend_path),
        "--output-dir", str(output_dir),
        "--resolution", "512",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0 and (output_dir / "image.png").exists():
            return True
        else:
            print(f"  FAIL {object_id}: {result.stderr[-200:] if result.stderr else 'unknown error'}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT {object_id}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Render missing 512px input images")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be rendered")
    args = parser.parse_args()

    # Load manifest
    with open(args.manifest) as f:
        samples = list(csv.DictReader(f))

    # Find missing renders
    missing = []
    for s in samples:
        input_image = s["input_image"]
        if not os.path.exists(input_image):
            missing.append(s)

    print(f"Total samples: {len(samples)}")
    print(f"Missing renders: {len(missing)}")

    if not missing:
        print("All renders exist. Nothing to do.")
        return

    if args.dry_run:
        for s in missing:
            print(f"  Would render: {s['object_id']} ({s['category']})")
        return

    # Find Blender
    blender_bin = find_blender()
    if blender_bin is None:
        print("ERROR: Blender not found. Install Blender or add it to PATH.")
        sys.exit(1)

    # Render missing
    success = 0
    failed = 0
    for i, s in enumerate(missing):
        oid = s["object_id"]
        cat = s["category"]
        print(f"[{i+1}/{len(missing)}] Rendering {oid}...")
        if render_object(blender_bin, cat, oid):
            success += 1
        else:
            failed += 1

    print(f"\nDone: {success} rendered, {failed} failed")


if __name__ == "__main__":
    main()
