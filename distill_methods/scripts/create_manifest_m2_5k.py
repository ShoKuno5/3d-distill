#!/usr/bin/env python3
"""Create the M2 5K-balanced training manifest from files actually on TSUBAME.

Bypasses the TRELLIS-500K metadata path resolvers (which assume the
Inspire/Aliyun layout) and instead scans what's physically present on
TSUBAME:

  - HSSD .glb files at /gs/fs/tga-koike-shanda2/sk/data/trellis500k_root/HSSD/raw/objects/
    (sk's local copy, ~14K files)
  - ObjaverseXL_sketchfab .glb files at
    /gs/bs/tga-koike-shanda/yurh/dataset/trellis-500k/ObjaverseXL_sketchfab/raw/hf-objaverse-v1/glbs/<bucket>/
    (yurh team-shared, ~91K files spread over 160 buckets named like
    "000-012"; bucket naming is opaque so we discover files directly)
  - ABO would be similar but the 80 GB tarball isn't extracted yet, so
    skip ABO in M2 v1.

Optional: cross-reference TRELLIS-500K metadata.csv to attach
aesthetic_score and apply a quality floor.

Toys4k disjoint guard still runs (sha256/name lookups against
distill_methods/data/toys4k_uids.txt + toys4k_names.txt).

Output: distill_methods/manifests/<name>/train.csv with columns
(object_id, category, mesh_obj, point_cloud, input_image, aesthetic).

Usage:
    python distill_methods/scripts/create_manifest_m2_5k.py \\
        --hssd-count 1500 \\
        --objaverse-count 3500 \\
        --name 5k_balanced \\
        --seed 42
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DM_DIR = PROJECT_DIR / "distill_methods"
DATA_DIR = DM_DIR / "data"

# Actual data roots on TSUBAME
HSSD_GLB_ROOT = Path(os.environ.get(
    "HSSD_GLB_ROOT",
    "/gs/fs/tga-koike-shanda2/sk/data/trellis500k_root/HSSD/raw/objects",
))
OBJAVERSE_GLB_ROOT = Path(os.environ.get(
    "OBJAVERSE_GLB_ROOT",
    "/gs/bs/tga-koike-shanda/yurh/dataset/trellis-500k/ObjaverseXL_sketchfab/raw/hf-objaverse-v1/glbs",
))

# Optional metadata CSVs for aesthetic filtering
HSSD_META = Path(os.environ.get(
    "HSSD_META",
    "/gs/fs/tga-koike-shanda2/sk/data/trellis500k_meta/HSSD.csv",
))
OBJAVERSE_META = Path(os.environ.get(
    "OBJAVERSE_META",
    "/gs/fs/tga-koike-shanda2/sk/data/trellis500k_meta/ObjaverseXL_sketchfab.csv",
))

# Where derived assets land (per-object renders, encoded latents, etc.)
SK_DATA_ROOT = Path(os.environ.get(
    "SK_DATA_ROOT",
    "/gs/fs/tga-koike-shanda2/sk/data/5k_balanced",
))


def load_toys4k_refs() -> tuple[set[str], set[str]]:
    uids_file = DATA_DIR / "toys4k_uids.txt"
    names_file = DATA_DIR / "toys4k_names.txt"
    if not uids_file.exists() or not names_file.exists():
        raise FileNotFoundError(
            f"Toys4k guard files missing: {uids_file} and {names_file}"
        )
    uids = {ln.strip() for ln in uids_file.read_text().splitlines() if ln.strip()}
    names = {ln.strip() for ln in names_file.read_text().splitlines() if ln.strip()}
    return uids, names


def scan_hssd() -> dict[str, Path]:
    """Return {object_id: Path} for every HSSD .glb actually present.
    object_id is the filename stem (a sha-like hex string)."""
    if not HSSD_GLB_ROOT.exists():
        raise FileNotFoundError(f"HSSD glb root not found: {HSSD_GLB_ROOT}")
    out: dict[str, Path] = {}
    for p in HSSD_GLB_ROOT.rglob("*.glb"):
        out[p.stem] = p
    return out


def scan_objaverse() -> dict[str, Path]:
    """Return {object_id: Path} for every Objaverse-XL Sketchfab .glb."""
    if not OBJAVERSE_GLB_ROOT.exists():
        raise FileNotFoundError(f"Objaverse glb root not found: {OBJAVERSE_GLB_ROOT}")
    out: dict[str, Path] = {}
    for bucket in OBJAVERSE_GLB_ROOT.iterdir():
        if not bucket.is_dir():
            continue
        for p in bucket.glob("*.glb"):
            out[p.stem] = p
    return out


def load_aesthetic_index(meta_path: Path, key_column: str) -> dict[str, float]:
    """Build {key: aesthetic_score} from a TRELLIS-500K metadata CSV.
    `key_column` is which column to use as lookup key: usually
    'sha256' for HSSD (matches file_identifier basename) or a parsed
    URL UUID for Objaverse (file_identifier last segment)."""
    if not meta_path.exists():
        print(f"  [warn] metadata not found, skipping aesthetic filter: {meta_path}")
        return {}
    out: dict[str, float] = {}
    with meta_path.open() as f:
        for row in csv.DictReader(f):
            try:
                aes = float(row["aesthetic_score"])
            except (KeyError, ValueError):
                continue
            if key_column == "file_identifier_stem":
                # HSSD: file_identifier = "objects/X/<stem>.glb"; key = stem
                fid = row.get("file_identifier", "")
                key = Path(fid).stem
            elif key_column == "file_identifier_uuid":
                # Objaverse: file_identifier = "https://sketchfab.com/3d-models/<uuid>"
                fid = row.get("file_identifier", "")
                key = fid.rstrip("/").rsplit("/", 1)[-1]
            else:
                key = row.get(key_column, "")
            if key:
                out[key] = aes
    return out


def pick_subset(
    files: dict[str, Path],
    aesthetics: dict[str, float],
    target: int,
    min_aesthetic: float,
    toys4k_uids: set[str],
    exclude_oids: set[str],
    rng: random.Random,
) -> list[tuple[str, Path, float]]:
    """Return up to `target` (object_id, glb_path, aesthetic) tuples.

    Filters: drop Toys4k overlaps; drop entries below min_aesthetic
    when aesthetics is non-empty; drop entries whose oid is in
    exclude_oids (used to skip already-tried-and-failed samples).
    Sorts by aesthetic descending and randomly samples from the
    top-2x pool for quality + diversity."""
    candidates: list[tuple[str, Path, float]] = []
    drop_toys = 0
    drop_aes = 0
    drop_excl = 0
    for oid, path in files.items():
        if oid in toys4k_uids:
            drop_toys += 1
            continue
        if oid in exclude_oids:
            drop_excl += 1
            continue
        aes = aesthetics.get(oid, 0.0 if aesthetics else float("inf"))
        if aesthetics and aes < min_aesthetic:
            drop_aes += 1
            continue
        candidates.append((oid, path, aes))

    print(f"    files on disk: {len(files)}, after filters: {len(candidates)} "
          f"(dropped {drop_toys} Toys4k, {drop_aes} below aesthetic floor, "
          f"{drop_excl} excluded)")

    if not candidates:
        return []
    if not aesthetics:
        # No aesthetic data available (e.g. Objaverse-Sketchfab where the
        # metadata uses URL UUIDs that don't match the on-disk filenames).
        # Sample uniformly from ALL candidates to avoid filesystem-order bias.
        if len(candidates) <= target:
            return candidates
        return rng.sample(candidates, target)
    # Aesthetic-aware: top 2x pool then random sample for quality + diversity.
    candidates.sort(key=lambda x: -x[2])
    if len(candidates) <= target:
        return candidates
    pool = candidates[: target * 2]
    return rng.sample(pool, target)


def write_manifest(rows: list[tuple[str, str, Path, float]], path: Path) -> None:
    fieldnames = ["object_id", "category", "mesh_obj", "point_cloud", "input_image", "aesthetic"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for oid, category, glb_path, aes in rows:
            writer.writerow({
                "object_id":  oid,
                "category":   category,
                "mesh_obj":   str(glb_path),
                "point_cloud": str(SK_DATA_ROOT / category / oid / "pc10K.npz"),
                "input_image": str(SK_DATA_ROOT / category / oid / "image.png"),
                "aesthetic":  f"{aes:.4f}",
            })
    print(f"  Written: {path} ({len(rows)} samples)")


def main() -> None:
    parser = argparse.ArgumentParser(description="M2 5K-balanced manifest from physical files")
    parser.add_argument("--hssd-count", type=int, default=1500)
    parser.add_argument("--objaverse-count", type=int, default=3500)
    parser.add_argument("--min-aesthetic", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="5k_balanced",
                        help="Output dir: distill_methods/manifests/<name>/train.csv")
    parser.add_argument("--exclude-oids-file", action="append", default=[],
                        help="Path to a text file of object_ids (one per line) to exclude. "
                             "Can be passed multiple times to combine lists. Useful for "
                             "skipping already-tried-and-failed samples and for skipping "
                             "samples that are already in an earlier manifest.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    toys4k_uids, _ = load_toys4k_refs()
    print(f"Toys4k guard: {len(toys4k_uids)} sha256")

    exclude_oids: set[str] = set()
    for fp in args.exclude_oids_file:
        with open(fp) as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    exclude_oids.add(ln)
    if exclude_oids:
        print(f"Exclude list: {len(exclude_oids)} oids loaded from {len(args.exclude_oids_file)} files")
    print(f"HSSD glb root:      {HSSD_GLB_ROOT}")
    print(f"Objaverse glb root: {OBJAVERSE_GLB_ROOT}")
    print(f"Derived assets:     {SK_DATA_ROOT}")
    print(f"Targets:            HSSD={args.hssd_count}, Objaverse={args.objaverse_count}, "
          f"total={args.hssd_count + args.objaverse_count}")
    print(f"Aesthetic floor:    {args.min_aesthetic}")
    print()

    hssd_picked: list = []
    if args.hssd_count > 0:
        print("Scanning HSSD ...")
        hssd_files = scan_hssd()
        hssd_aes = load_aesthetic_index(HSSD_META, "file_identifier_stem")
        print(f"  HSSD aesthetic index: {len(hssd_aes)} entries")
        hssd_picked = pick_subset(hssd_files, hssd_aes, args.hssd_count,
                                  args.min_aesthetic, toys4k_uids, exclude_oids, rng)

    obja_picked: list = []
    if args.objaverse_count > 0:
        print("Scanning Objaverse_sketchfab ...")
        obja_files = scan_objaverse()
        obja_aes = load_aesthetic_index(OBJAVERSE_META, "file_identifier_uuid")
        print(f"  Objaverse aesthetic index: {len(obja_aes)} entries")
        obja_picked = pick_subset(obja_files, obja_aes, args.objaverse_count,
                                  args.min_aesthetic, toys4k_uids, exclude_oids, rng)

    rows = []
    rows.extend((oid, "hssd", p, aes) for oid, p, aes in hssd_picked)
    rows.extend((oid, "objaverse_sketchfab", p, aes) for oid, p, aes in obja_picked)
    print()
    print(f"Total picked: {len(rows)}")

    out_dir = DM_DIR / "manifests" / args.name
    write_manifest(rows, out_dir / "train.csv")


if __name__ == "__main__":
    main()
