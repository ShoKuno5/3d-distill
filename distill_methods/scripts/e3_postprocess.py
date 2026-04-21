#!/usr/bin/env python3
"""Post-processing for E3: compare teacher CDs at 50/100/200 steps.

Reads:
  - 50-step baseline: runs/20260421_cross_family/metrics/per_sample.csv
  - 100-step:         runs/20260421_manifold_diagnosis/e3_teacher_extended/metrics/per_sample.csv
  - 200-step:         (same CSV — run_eval merges all models for a run_dir)
"""
import argparse
import csv
from pathlib import Path


def load_per_sample(csv_path: Path):
    if not csv_path.exists():
        return []
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cross-family-csv", required=True, type=Path)
    ap.add_argument("--extended-csv", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--samples", nargs="+",
                    default=["helicopter_015", "robot_016", "dinosaur_069"])
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    base = load_per_sample(args.cross_family_csv)
    ext = load_per_sample(args.extended_csv)

    rows = []
    for oid in args.samples:
        for track in ["A", "B"]:
            record = {"object_id": oid, "track": track}
            # 50-step baseline
            for r in base:
                if r["model"] == "teacher_50step" and r["object_id"] == oid and r["track"] == track:
                    record["cd_50"] = float(r["chamfer_distance"]) if r["chamfer_distance"] else None
                    record["f001_50"] = float(r["f_score_001"]) if r["f_score_001"] else None
                    record["scale_50"] = float(r["alignment_scale"]) if r["alignment_scale"] else None
                    break
            # 100-step
            for r in ext:
                if r["model"] == "teacher_100step" and r["object_id"] == oid and r["track"] == track:
                    record["cd_100"] = float(r["chamfer_distance"]) if r["chamfer_distance"] else None
                    record["f001_100"] = float(r["f_score_001"]) if r["f_score_001"] else None
                    record["scale_100"] = float(r["alignment_scale"]) if r["alignment_scale"] else None
                    break
            # 200-step
            for r in ext:
                if r["model"] == "teacher_200step" and r["object_id"] == oid and r["track"] == track:
                    record["cd_200"] = float(r["chamfer_distance"]) if r["chamfer_distance"] else None
                    record["f001_200"] = float(r["f_score_001"]) if r["f_score_001"] else None
                    record["scale_200"] = float(r["alignment_scale"]) if r["alignment_scale"] else None
                    break
            rows.append(record)

    out_csv = args.out_dir / "step_comparison.csv"
    fieldnames = ["object_id", "track",
                  "cd_50", "cd_100", "cd_200",
                  "f001_50", "f001_100", "f001_200",
                  "scale_50", "scale_100", "scale_200"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"{'oid':15s} {'track':>5s} {'CD_50':>10s} {'CD_100':>10s} {'CD_200':>10s}  {'F@1_50':>7s} {'F@1_100':>8s} {'F@1_200':>8s}  {'ratio_50/200':>13s}")
    for r in rows:
        cd50 = r.get("cd_50")
        cd100 = r.get("cd_100")
        cd200 = r.get("cd_200")
        ratio = f"{cd50/cd200:.2f}x" if cd50 and cd200 and cd200 > 0 else "?"
        f50 = r.get("f001_50")
        f100 = r.get("f001_100")
        f200 = r.get("f001_200")
        print(f"{r['object_id']:15s} {r['track']:>5s} "
              f"{cd50 if cd50 is not None else '?':>10} {cd100 if cd100 is not None else '?':>10} {cd200 if cd200 is not None else '?':>10}  "
              f"{f50 if f50 is not None else '?':>7} {f100 if f100 is not None else '?':>8} {f200 if f200 is not None else '?':>8}  "
              f"{ratio:>13s}")

    print(f"\nCSV: {out_csv}")


if __name__ == "__main__":
    main()
