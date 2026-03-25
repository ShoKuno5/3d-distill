#!/usr/bin/env python3
"""Split manifest object IDs into N files for parallel inference."""

import argparse
import csv
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to manifest CSV")
    parser.add_argument("--output-dir", required=True, help="Directory for split files")
    parser.add_argument("--n-splits", type=int, default=2, help="Number of splits")
    args = parser.parse_args()

    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))

    object_ids = [r["object_id"] for r in rows]
    os.makedirs(args.output_dir, exist_ok=True)

    # Round-robin distribution
    splits = [[] for _ in range(args.n_splits)]
    for i, oid in enumerate(object_ids):
        splits[i % args.n_splits].append(oid)

    for i, split in enumerate(splits):
        path = os.path.join(args.output_dir, f"split_{i}.txt")
        with open(path, "w") as f:
            f.write("\n".join(split) + "\n")
        print(f"Split {i}: {len(split)} objects -> {path}")


if __name__ == "__main__":
    main()
