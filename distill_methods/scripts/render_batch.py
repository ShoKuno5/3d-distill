#!/usr/bin/env python3
"""Batch render input images for distillation manifests.

Reads manifest_train.csv + manifest_test.csv, finds samples missing renders,
and renders them using Blender in parallel across GPUs.

Usage:
    python distill_methods/scripts/render_batch.py
    python distill_methods/scripts/render_batch.py --dry-run
    python distill_methods/scripts/render_batch.py --num-gpus 2
"""

import argparse
import csv
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from threading import Lock

PROJECT_DIR = Path(__file__).resolve().parents[2]
BLENDER = str(PROJECT_DIR / "envs" / "blender-3.6.16-linux-x64" / "blender")
RENDER_SCRIPT = str(PROJECT_DIR / "pipeline" / "scripts" / "_blender_render_multipass.py")
BLEND_DIR = PROJECT_DIR / "datasets" / "Toys4k" / "official" / "toys4k_blend_files"
MANIFEST_DIR = PROJECT_DIR / "distill_methods"


def load_samples_needing_render():
    """Load all samples from both manifests, return those missing input images."""
    samples = []
    for manifest_name in ["manifest_train.csv", "manifest_test.csv"]:
        manifest_path = MANIFEST_DIR / manifest_name
        if not manifest_path.exists():
            print(f"WARNING: {manifest_path} not found, skipping")
            continue
        with open(manifest_path) as f:
            for row in csv.DictReader(f):
                samples.append(row)

    # Deduplicate by object_id (in case overlap)
    seen = set()
    unique = []
    for s in samples:
        if s["object_id"] not in seen:
            seen.add(s["object_id"])
            unique.append(s)

    # Filter to those needing render
    missing = []
    for s in unique:
        image_path = s["input_image"]
        if os.path.exists(image_path):
            continue
        # Check .blend file exists
        cat = s["category"]
        oid = s["object_id"]
        blend_path = BLEND_DIR / cat / oid / f"{oid}.blend"
        if not blend_path.exists():
            print(f"WARNING: No .blend for {oid}: {blend_path}")
            continue
        missing.append((s, str(blend_path)))

    return unique, missing


BLENDER_LIB = str(PROJECT_DIR / "envs" / "blender-3.6.16-linux-x64" / "lib")


def render_one(blend_path, output_dir, gpu_id, resolution=512):
    """Render a single object using Blender."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["LD_LIBRARY_PATH"] = BLENDER_LIB + ":" + env.get("LD_LIBRARY_PATH", "")
    cmd = [
        BLENDER, "--background", "--python", RENDER_SCRIPT,
        "--", "--input", blend_path, "--output-dir", output_dir,
        "--resolution", str(resolution),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env, timeout=120,
    )
    if result.returncode != 0:
        return False, result.stderr[-500:] if result.stderr else "unknown error"
    return True, ""


def main():
    parser = argparse.ArgumentParser(description="Batch render input images")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    total_samples, missing = load_samples_needing_render()
    print(f"Total unique samples: {len(total_samples)}")
    print(f"Already rendered: {len(total_samples) - len(missing)}")
    print(f"Need rendering: {len(missing)}")

    if args.dry_run or not missing:
        return

    # Round-robin GPU assignment
    gpu_queues = [[] for _ in range(args.num_gpus)]
    for i, (sample, blend_path) in enumerate(missing):
        gpu_queues[i % args.num_gpus].append((sample, blend_path))

    print(f"\nRendering {len(missing)} samples across {args.num_gpus} GPUs...")
    t0 = time.time()

    success_count = 0
    fail_count = 0
    lock = Lock()

    def render_worker(gpu_id, queue):
        nonlocal success_count, fail_count
        for sample, blend_path in queue:
            output_dir = os.path.dirname(sample["input_image"])
            ok, err = render_one(blend_path, output_dir, gpu_id, args.resolution)
            with lock:
                if ok:
                    success_count += 1
                    total_done = success_count + fail_count
                    if total_done % 50 == 0 or total_done == len(missing):
                        elapsed = time.time() - t0
                        print(f"  Progress: {total_done}/{len(missing)} ({elapsed:.0f}s)")
                else:
                    fail_count += 1
                    print(f"  FAILED: {sample['object_id']}: {err[:100]}")

    with ThreadPoolExecutor(max_workers=args.num_gpus) as pool:
        futures = []
        for gpu_id in range(args.num_gpus):
            futures.append(pool.submit(render_worker, gpu_id, gpu_queues[gpu_id]))
        for f in futures:
            f.result()

    elapsed = time.time() - t0
    print(f"\nDone: {success_count} rendered, {fail_count} failed ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
