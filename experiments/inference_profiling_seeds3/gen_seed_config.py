#!/usr/bin/env python3
"""Generate a per-seed config by patching predictions_root and output_root.

Usage:
    python gen_seed_config.py --config config.yaml --seed 1 --output configs/config_seed_1.yaml
"""

import argparse
import re

import yaml


def main():
    parser = argparse.ArgumentParser(description="Generate per-seed config")
    parser.add_argument("--config", required=True, help="Base config YAML")
    parser.add_argument("--seed", type=int, required=True, help="Seed number")
    parser.add_argument("--output", required=True, help="Output YAML path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed_tag = f"seed_{args.seed}"

    # Patch predictions_root: replace seed_N with the target seed
    for model in cfg["models"]:
        old = model["predictions_root"]
        model["predictions_root"] = re.sub(r"/seed_\d+$", f"/{seed_tag}", old)

    # Patch output_root: replace eval_seed_N with the target seed
    old_output = cfg["output_root"]
    cfg["output_root"] = re.sub(r"/eval_seed_\d+$", f"/eval_{seed_tag}", old_output)

    with open(args.output, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"Generated {args.output} (seed={args.seed})")


if __name__ == "__main__":
    main()
