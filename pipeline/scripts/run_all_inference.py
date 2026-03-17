#!/usr/bin/env python3
"""Check inference status for all models.

This script checks which predictions exist and reports status.
For actually running inference, use the model-specific scripts
with their respective environments.

Usage:
    python pipeline/scripts/run_all_inference.py \
        --config experiments/toys4k_baseline/config.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.utils.inference_config import load_and_filter_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    samples = load_and_filter_samples(cfg, max_samples_override=args.max_samples)

    print(f"Checking inference status for {len(samples)} samples\n")

    all_ready = True
    for model_cfg in cfg["models"]:
        name = model_cfg["name"]
        pred_root = model_cfg["predictions_root"]
        mesh_fn = model_cfg["mesh_filename"]

        found = 0
        missing_ids = []
        for s in samples:
            mesh_path = os.path.join(pred_root, s.object_id, mesh_fn)
            if os.path.exists(mesh_path):
                found += 1
            else:
                missing_ids.append(s.object_id)

        status = "READY" if found == len(samples) else "INCOMPLETE"
        print(f"  {name:<12} {found}/{len(samples)} {status}")
        if missing_ids:
            print(f"    Missing: {missing_ids}")
            all_ready = False

    print()
    if all_ready:
        print("All predictions available. Ready to run evaluation:")
        print(f"  python pipeline/scripts/run_eval.py --config {args.config}")
    else:
        print("Some predictions missing. Run the model-specific inference scripts first.")
        print("See pipeline/scripts/run_inference_*.py for details.")


if __name__ == "__main__":
    main()
