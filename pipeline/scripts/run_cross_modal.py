#!/usr/bin/env python
"""Standalone ULIP-I / Uni3D-I (cross-modal) runner.

run_eval.py only computes cross-modal at the very end, after the full geometric
Track A/B loop — which would needlessly recompute geometry and overwrite
summary.csv. This runner computes ONLY cross-modal: it loads the same config /
manifest / predictions as run_eval, then calls compute_cross_modal_for_run().
Cross-modal needs no renders (point clouds are sampled from the meshes), so it
can run independently of the multiview-render step.

Shardable by sample for intra-node multi-GPU: each shard processes
samples[shard_index::num_shards] for all models and writes its own CSV; merge
the shard CSVs afterwards (see scripts merge or `--out-csv`).

Example (4 GPUs of one node):
    for K in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$K python run_cross_modal.py --config CFG \
        --models teacher_50step flashvdm_dsw cd_4step dmd2_1step mdt_dist trellis2 \
        --shard-index $K --num-shards 4 \
        --out-csv $EVAL_ROOT/metrics/cross_modal_shard$K.csv &
    done; wait
"""
import argparse
import os
import sys
import yaml

from src.data.toys4k import load_manifest, filter_samples
from src.utils.inference_config import resolve_model_paths
from src.evaluation.cross_modal_metric import compute_cross_modal_for_run


def main():
    parser = argparse.ArgumentParser(description="Standalone cross-modal (ULIP-I/Uni3D-I) runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--out-csv", default=None,
                        help="Output CSV path (default: <output_root>/metrics/cross_modal.csv)")
    args = parser.parse_args()

    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        print(f"ERROR: invalid sharding: shard_index={args.shard_index}, num_shards={args.num_shards}")
        sys.exit(1)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    resolve_model_paths(cfg)
    output_root = cfg["output_root"]

    manifest_path = cfg["dataset"].get("test_manifest") or cfg["dataset"]["manifest"]
    max_samples = args.max_samples or cfg["dataset"].get("max_samples")
    samples = load_manifest(manifest_path)
    samples = filter_samples(
        samples,
        max_samples=max_samples,
        sample_ids_file=cfg["dataset"].get("sample_ids_file"),
        category_filter=cfg["dataset"].get("category_filter"),
    )
    # Stable order before sharding so slices are disjoint and reproducible.
    samples = sorted(samples, key=lambda s: s.object_id)
    if args.num_shards > 1:
        before = len(samples)
        samples = samples[args.shard_index::args.num_shards]
        print(f"Shard {args.shard_index}/{args.num_shards}: {len(samples)}/{before} samples", flush=True)

    model_cfgs = cfg["models"]
    if args.models:
        model_cfgs = [m for m in model_cfgs if m["name"] in args.models]

    print(f"cross-modal: {len(samples)} samples x {len(model_cfgs)} models "
          f"({[m['name'] for m in model_cfgs]})", flush=True)

    compute_cross_modal_for_run(
        cfg=cfg, samples=samples, output_root=output_root,
        model_cfgs=model_cfgs, out_csv=args.out_csv,
    )


if __name__ == "__main__":
    main()
