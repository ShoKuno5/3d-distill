"""Analyze VAE volume logits at each ODE step (no marching cubes).

Runs at both low (64) and high (384) octree resolution to check
whether early-step structure is masked by coarse voxelization.

Usage:
    cd models/hunyuan3d21/hy3dshape
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 ../../../envs/hunyuan3d-venv/bin/python \
      ../../../distill_methods/scripts/_volume_logit_analysis.py [--latent-dir DIR]
"""
import argparse
import numpy as np
import torch
import os

from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-dir", default="../../../results/distill_methods/intermediate_decode")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[64, 384])
    args = parser.parse_args()

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2.1", use_safetensors=False,
    )
    vae = pipeline.vae.to("cuda")
    del pipeline.model, pipeline.conditioner
    torch.cuda.empty_cache()

    steps = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    samples = ["ball_002", "chair_168", "helicopter_015"]

    for res in args.resolutions:
        print(f"\n{'='*60}")
        print(f"  Octree resolution: {res}")
        print(f"{'='*60}")

        for oid in samples:
            print(f"\n--- {oid} ---")
            for step in steps:
                path = os.path.join(args.latent_dir, oid, f"latent_step_{step:02d}.npy")
                if not os.path.exists(path):
                    print(f"  step {step:2d}: MISSING")
                    continue
                z = np.load(path)
                latent = torch.from_numpy(z).unsqueeze(0).to("cuda", dtype=torch.float16)
                with torch.no_grad():
                    decoded = vae(latent / vae.scale_factor)
                    grid = vae.volume_decoder(
                        decoded, vae.geo_decoder, bounds=1.01, num_chunks=8000,
                        octree_resolution=res, enable_pbar=False,
                    )
                g = grid[0].cpu().numpy()
                neg_pct = 100 * (g < 0).mean()
                crossing = (g.min() < 0) and (g.max() > 0)
                print(
                    f"  step {step:2d} t={step/50:.1f}  "
                    f"range=[{g.min():.2f},{g.max():.2f}]  "
                    f"neg%={neg_pct:5.1f}  std={g.std():.3f}  "
                    f"crossing={crossing}"
                )


if __name__ == "__main__":
    main()
