#!/usr/bin/env python3
"""Create train/test manifest from TRELLIS-500K (minus Toys4k).

Reads each dataset's metadata.csv (sha256 / file_identifier / aesthetic_score /
captions), resolves the physical .glb path on the Inspire shared mount, applies
hard Toys4k disjoint, optionally filters by aesthetic_score top-N, then writes
two CSVs (train, test) with the same columns the historical Toys4k manifest
used (object_id, category, mesh_obj, point_cloud, input_image).

object_id == sha256 (the TRELLIS-500K canonical UID).
category   == dataset name (abo / hssd / 3d_future / objaverse_xl_github / etc).
point_cloud and input_image columns are filled with placeholder paths under
SK5_DATA_ROOT — they don't exist yet, prepare_training_data.py / Blender will
populate them.

Usage:
    python distill_methods/scripts/create_manifests_trellis500k.py \
        --datasets abo hssd \
        --target-samples 525 \
        --train-test-ratio 4.0 \
        --seed 42

The Toys4k disjoint guard (data/toys4k_uids.txt + toys4k_names.txt) is enforced
here on top of train.py's runtime check — belt-and-suspenders.
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

# Inspire shared (read-only) mount
TRELLIS_ROOT_ENV = "TRELLIS500K_ROOT"
TRELLIS_ROOT_DEFAULT = "/inspire/hdd/global_public/public_datas/TRELLIS-500k"

# sk-train working data root (where renders + pcd derive into)
SK5_DATA_ENV = "SK5_DATA_ROOT"
SK5_DATA_DEFAULT = "/inspire/qb-ilm/project/qproject-assement/zhangkaipeng-24043/sk5/data/trellis500k"


# ---------------------------------------------------------------------------
# Per-dataset path resolvers
# ---------------------------------------------------------------------------

def resolve_abo(trellis_root: Path, file_id: str) -> Path:
    # metadata: "3/B07YBH2SR3.glb"
    # physical: "ABO/raw/3dmodels/original/3/B07YBH2SR3.glb"
    return trellis_root / "ABO" / "raw" / "3dmodels" / "original" / file_id


def resolve_hssd(trellis_root: Path, file_id: str) -> Path:
    # metadata: "objects/3/3e479054...glb"  -- already includes objects/ prefix
    # physical: "HSSD/raw/<that path>"
    return trellis_root / "HSSD" / "raw" / file_id


def resolve_3d_future(trellis_root: Path, file_id: str) -> Path:
    # metadata: "3D-FUTURE-model/127406da-...-773b"  -- inside one of the 4 zips
    # physical: zips are not extracted by default; caller must extract or use
    # in-memory zipfile reader. For manifest purposes, return the IDEAL path
    # under <SK5_DATA_ROOT>/3d_future/<uuid>/raw_model.obj which prepare_data.sh
    # will populate.
    uuid = file_id.split("/", 1)[1] if "/" in file_id else file_id
    return Path(os.environ.get(SK5_DATA_ENV, SK5_DATA_DEFAULT)) / "3d_future" / uuid / "raw_model.obj"


def resolve_objaverse_xl_github(trellis_root: Path, file_id: str) -> Path:
    # metadata: "github/<user>/<repo>/<path-in-repo>"
    return trellis_root / "ObjaverseXL_github" / "raw" / file_id


def resolve_objaverse_xl_sketchfab(trellis_root: Path, file_id: str) -> Path:
    # metadata: depends; usually under hf-objaverse-v1/
    return trellis_root / "ObjaverseXL_sketchfab" / "raw" / file_id


DATASETS = {
    "abo":                    {"dir": "ABO",                    "resolve": resolve_abo},
    "hssd":                   {"dir": "HSSD",                   "resolve": resolve_hssd},
    "3d_future":              {"dir": "3D-FUTURE",              "resolve": resolve_3d_future},
    "objaverse_xl_github":    {"dir": "ObjaverseXL_github",     "resolve": resolve_objaverse_xl_github},
    "objaverse_xl_sketchfab": {"dir": "ObjaverseXL_sketchfab",  "resolve": resolve_objaverse_xl_sketchfab},
}


# ---------------------------------------------------------------------------
# Toys4k disjoint guard (manifest-side; duplicates train.py's runtime check)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Per-dataset metadata load + filter + sample
# ---------------------------------------------------------------------------

def load_metadata(trellis_root: Path, dataset_key: str, toys4k_uids: set[str]):
    """Yield (sha256, file_identifier, aesthetic_score) for every non-Toys4k row."""
    meta = trellis_root / DATASETS[dataset_key]["dir"] / "metadata.csv"
    with meta.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            sha = row["sha256"]
            if sha in toys4k_uids:
                continue  # paranoid — these datasets shouldn't overlap with Toys4k anyway
            try:
                aes = float(row["aesthetic_score"])
            except (KeyError, ValueError):
                aes = 0.0
            yield sha, row["file_identifier"], aes


def sample_per_dataset(trellis_root: Path, dataset_keys: list[str], total_target: int,
                       toys4k_uids: set[str], min_aesthetic: float, seed: int) -> dict[str, list]:
    """Stratify total_target evenly across dataset_keys, top-aesthetic within each."""
    rng = random.Random(seed)
    per_ds_target = total_target // len(dataset_keys)
    selected = {}
    for k in dataset_keys:
        rows = [(sha, fid, aes) for sha, fid, aes in load_metadata(trellis_root, k, toys4k_uids)
                if aes >= min_aesthetic]
        rows.sort(key=lambda r: -r[2])  # highest aesthetic first
        if len(rows) <= per_ds_target:
            picked = rows
        else:
            # take top 2x by aesthetic, then random sample to target (diversity)
            pool = rows[:per_ds_target * 2]
            picked = rng.sample(pool, per_ds_target)
        selected[k] = picked
        print(f"  {k}: {len(rows)} candidates >= aes {min_aesthetic}, picked {len(picked)}")
    return selected


def train_test_split(selected: dict[str, list], train_ratio: float, seed: int):
    rng = random.Random(seed + 1)
    train, test = [], []
    for dataset_key, rows in sorted(selected.items()):
        shuffled = list(rows)
        rng.shuffle(shuffled)
        n_total = len(shuffled)
        n_train = int(n_total * train_ratio / (train_ratio + 1.0))
        train.extend((dataset_key, *r) for r in shuffled[:n_train])
        test.extend((dataset_key, *r) for r in shuffled[n_train:])
    return train, test


def make_row(dataset_key: str, sha: str, file_id: str, aes: float,
             trellis_root: Path, sk5_data_root: Path) -> dict:
    mesh_path = DATASETS[dataset_key]["resolve"](trellis_root, file_id)
    # derived assets land under sk5_data_root, keyed by sha256
    return {
        "object_id":  sha,
        "category":   dataset_key,
        "mesh_obj":   str(mesh_path),
        "point_cloud": str(sk5_data_root / dataset_key / sha / "pc10K.npz"),
        "input_image": str(sk5_data_root / dataset_key / sha / "image.png"),
        "aesthetic":  f"{aes:.4f}",
    }


def write_manifest(rows: list, path: Path, trellis_root: Path, sk5_data_root: Path):
    fieldnames = ["object_id", "category", "mesh_obj", "point_cloud", "input_image", "aesthetic"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for dataset_key, sha, file_id, aes in rows:
            writer.writerow(make_row(dataset_key, sha, file_id, aes, trellis_root, sk5_data_root))
    print(f"  Written: {path} ({len(rows)} samples)")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Create manifests from TRELLIS-500K (minus Toys4k)")
    parser.add_argument("--datasets", nargs="+", required=True,
                        choices=sorted(DATASETS.keys()),
                        help="Which sub-datasets to draw from")
    parser.add_argument("--target-samples", type=int, default=525,
                        help="Total samples (train + test combined), stratified evenly across datasets")
    parser.add_argument("--train-ratio", type=float, default=4.0,
                        help="train:test ratio (4.0 => 80/20)")
    parser.add_argument("--min-aesthetic", type=float, default=4.0,
                        help="Drop anything with aesthetic_score below this")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="525_abo_hssd",
                        help="Manifest set name; output goes to distill_methods/manifests/<name>/{train,test}.csv")
    args = parser.parse_args()

    trellis_root = Path(os.environ.get(TRELLIS_ROOT_ENV, TRELLIS_ROOT_DEFAULT))
    sk5_data_root = Path(os.environ.get(SK5_DATA_ENV, SK5_DATA_DEFAULT))
    assert trellis_root.exists(), f"TRELLIS500K root not found: {trellis_root}"

    toys4k_uids, _ = load_toys4k_refs()
    print(f"Loaded Toys4k guard: {len(toys4k_uids)} sha256")
    print(f"TRELLIS500K root:   {trellis_root}")
    print(f"Derived assets root: {sk5_data_root}")
    print(f"Datasets:           {args.datasets}")
    print(f"Target:             {args.target_samples} samples, train:test = {args.train_ratio}:1")
    print(f"Aesthetic floor:    {args.min_aesthetic}")
    print()

    selected = sample_per_dataset(trellis_root, args.datasets, args.target_samples,
                                  toys4k_uids, args.min_aesthetic, args.seed)
    train, test = train_test_split(selected, args.train_ratio, args.seed)
    print(f"Split: {len(train)} train, {len(test)} test")
    print()

    out_dir = DM_DIR / "manifests" / args.name
    write_manifest(train, out_dir / "train.csv", trellis_root, sk5_data_root)
    write_manifest(test,  out_dir / "test.csv",  trellis_root, sk5_data_root)


if __name__ == "__main__":
    main()
