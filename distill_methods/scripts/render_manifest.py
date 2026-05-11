#!/usr/bin/env python3
"""Batch render input images for a manifest CSV.

Reads a manifest (output of create_manifests_trellis500k.py or any other
script that produces an `input_image` column), finds rows whose input_image
file does not yet exist, and renders them via Blender + _blender_render_mesh.py.

Supports .obj, .glb, .gltf, .ply, .stl source meshes (extension dispatch in
the underlying render script). CUDA GPU rendering is enabled by default in
that script.

Usage:
    python distill_methods/scripts/render_manifest.py \
        --manifest distill_methods/manifests/525_hssd/train.csv \
        --resolution 512

    python distill_methods/scripts/render_manifest.py \
        --manifest distill_methods/manifests/525_hssd/train.csv \
        distill_methods/manifests/525_hssd/test.csv \
        distill_methods/manifests/toys4k_eval_105/test.csv \
        --dry-run
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]

# Resolve Blender + render script paths (overridable via env vars)
BLENDER_BIN_ENV = "BLENDER_BIN"
BLENDER_BIN_DEFAULT = "/inspire/qb-ilm/project/qproject-assement/zhangkaipeng-24043/sk5/.bin/blender"

RENDER_SCRIPT = str(PROJECT_DIR / "pipeline" / "scripts" / "_blender_render_mesh.py")


def collect_missing(manifests: list[Path]) -> list[dict]:
    """Read all rows; deduplicate by object_id; return only those needing render."""
    seen = set()
    rows = []
    for m in manifests:
        with m.open() as f:
            for row in csv.DictReader(f):
                oid = row["object_id"]
                if oid in seen:
                    continue
                seen.add(oid)
                rows.append(row)
    missing = []
    for r in rows:
        if not os.path.exists(r["input_image"]):
            if not os.path.exists(r["mesh_obj"]):
                print(f"WARN: mesh source missing for {r['object_id']}: {r['mesh_obj']}",
                      file=sys.stderr)
                continue
            missing.append(r)
    return rows, missing


def render_one(blender: str, mesh_path: str, output_png: str, resolution: int,
               timeout_sec: int = 180) -> tuple[bool, str]:
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    cmd = [
        blender, "--background",
        "--python", RENDER_SCRIPT,
        "--",
        "--input", mesh_path,
        "--output", output_png,
        "--resolution", str(resolution),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout_sec}s"
    if result.returncode != 0:
        tail = (result.stderr or result.stdout)[-500:]
        return False, tail
    return True, ""


def main():
    parser = argparse.ArgumentParser(description="Render input images for a manifest")
    parser.add_argument("--manifest", nargs="+", required=True,
                        help="One or more manifest CSV paths (rows pooled by object_id)")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print missing list, do not render")
    parser.add_argument("--timeout-sec", type=int, default=180,
                        help="Per-mesh render timeout")
    parser.add_argument("--max-renders", type=int, default=None,
                        help="Stop after N renders (debug)")
    args = parser.parse_args()

    blender = os.environ.get(BLENDER_BIN_ENV, BLENDER_BIN_DEFAULT)
    if not os.path.exists(blender):
        print(f"ERROR: Blender binary not found: {blender}", file=sys.stderr)
        sys.exit(2)

    manifests = [Path(p) for p in args.manifest]
    for m in manifests:
        if not m.exists():
            print(f"ERROR: Manifest not found: {m}", file=sys.stderr)
            sys.exit(2)

    rows, missing = collect_missing(manifests)
    print(f"Manifests: {len(manifests)}")
    print(f"Unique rows (deduped by object_id): {len(rows)}")
    print(f"Already rendered: {len(rows) - len(missing)}")
    print(f"Need rendering: {len(missing)}")
    if args.dry_run:
        for r in missing[:20]:
            print(f"  {r['object_id']:60s} {r['mesh_obj']}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        return

    to_render = missing if args.max_renders is None else missing[:args.max_renders]
    print(f"Rendering {len(to_render)} images @ {args.resolution}x{args.resolution} ...")
    t_start = time.time()
    ok = 0
    fail = 0
    for i, r in enumerate(to_render, 1):
        success, msg = render_one(blender, r["mesh_obj"], r["input_image"],
                                  args.resolution, args.timeout_sec)
        if success:
            ok += 1
        else:
            fail += 1
            print(f"  FAIL [{i}/{len(to_render)}] {r['object_id']}: {msg[:200]}",
                  file=sys.stderr)
        if i % 10 == 0 or i == len(to_render):
            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(to_render) - i) / rate if rate > 0 else 0
            print(f"  [{i}/{len(to_render)}] ok={ok} fail={fail} "
                  f"rate={rate:.2f}/s eta={eta/60:.1f}min")
    print(f"\nDone. ok={ok} fail={fail} elapsed={(time.time()-t_start)/60:.1f}min")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
