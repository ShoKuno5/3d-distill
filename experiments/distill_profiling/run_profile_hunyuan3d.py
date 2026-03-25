#!/usr/bin/env python3
"""Block-level inference time profiling for Hunyuan3D-2.0 (teacher).

Expands Hunyuan3DDiTFlowMatchingPipeline.__call__() inline with block-level timing.
Saves profile.json + mesh_raw.obj per sample.

Usage (from models/hunyuan3d):
    PYTHONPATH=. LD_PRELOAD=/opt/unreal-engine/usr/lib/x86_64-linux-gnu/libOpenGL.so.0 \
    CUDA_VISIBLE_DEVICES=2 \
    ../../envs/hunyuan3d-venv/bin/python \
    ../../experiments/distill_profiling/run_profile_hunyuan3d.py \
    --config ../../experiments/distill_profiling/config.yaml
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR / "pipeline"))

from src.utils.inference_config import (
    get_inference_params,
    get_model_config,
    load_and_filter_samples,
)


def sync_and_time():
    import torch
    torch.cuda.synchronize()
    return time.perf_counter()


def profile_one_sample(pipeline, image, num_inference_steps, guidance_scale,
                       octree_resolution, num_chunks, generator):
    """Run hunyuan3d v2.0 teacher pipeline with block-level timing."""
    import numpy as np
    import torch
    from tqdm import tqdm

    timings = {}
    device = pipeline.device
    dtype = pipeline.dtype

    do_classifier_free_guidance = guidance_scale >= 0 and not (
        hasattr(pipeline.model, "guidance_embed")
        and pipeline.model.guidance_embed is True
    )

    # --- Prepare image ---
    t0 = sync_and_time()
    cond_inputs = pipeline.prepare_image(image)
    image_tensor = cond_inputs.pop("image")
    timings["prepare_image"] = sync_and_time() - t0

    # --- Encode conditioning ---
    t0 = sync_and_time()
    cond = pipeline.encode_cond(
        image=image_tensor,
        additional_cond_inputs=cond_inputs,
        do_classifier_free_guidance=do_classifier_free_guidance,
        dual_guidance=False,
    )
    timings["encode_cond"] = sync_and_time() - t0

    batch_size = image_tensor.shape[0]

    # --- Prepare timesteps + latents ---
    t0 = sync_and_time()
    sigmas = np.linspace(0, 1, num_inference_steps)
    from hy3dgen.shapegen.pipelines import retrieve_timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        pipeline.scheduler, num_inference_steps, device, sigmas=sigmas
    )
    latents = pipeline.prepare_latents(batch_size, dtype, device, generator)

    guidance = None
    if hasattr(pipeline.model, "guidance_embed") and pipeline.model.guidance_embed is True:
        guidance = torch.tensor(
            [guidance_scale] * batch_size, device=device, dtype=dtype
        )
    timings["prepare_latents"] = sync_and_time() - t0

    # --- Diffusion sampling loop ---
    t0 = sync_and_time()
    for i, t in enumerate(tqdm(timesteps, desc="Diffusion Sampling")):
        if do_classifier_free_guidance:
            latent_model_input = torch.cat([latents] * 2)
        else:
            latent_model_input = latents

        timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
        timestep = timestep / pipeline.scheduler.config.num_train_timesteps
        noise_pred = pipeline.model(
            latent_model_input, timestep, cond, guidance=guidance
        )

        if do_classifier_free_guidance:
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

        outputs = pipeline.scheduler.step(noise_pred, t, latents)
        latents = outputs.prev_sample
    timings["diffusion"] = sync_and_time() - t0

    # --- VAE decode ---
    t0 = sync_and_time()
    scaled_latents = 1.0 / pipeline.vae.scale_factor * latents
    decoded_latents = pipeline.vae(scaled_latents)
    timings["vae_decode"] = sync_and_time() - t0

    # --- Marching cubes ---
    t0 = sync_and_time()
    mesh_outputs = pipeline.vae.latents2mesh(
        decoded_latents,
        bounds=1.01,
        mc_level=0.0,
        num_chunks=num_chunks,
        octree_resolution=octree_resolution,
        mc_algo=None,
        enable_pbar=True,
    )
    timings["latents2mesh"] = sync_and_time() - t0

    # --- Post-process to trimesh ---
    t0 = sync_and_time()
    from hy3dgen.shapegen.pipelines import export_to_trimesh
    mesh_list = export_to_trimesh(mesh_outputs)
    timings["postprocess"] = sync_and_time() - t0

    return mesh_list, timings


def main():
    parser = argparse.ArgumentParser(description="Hunyuan3D-2.0 teacher profiling")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--sample-ids", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.sample_ids:
        cfg["dataset"]["sample_ids_file"] = args.sample_ids

    samples = load_and_filter_samples(cfg, max_samples_override=args.max_samples)
    model_cfg = get_model_config(cfg, "hunyuan3d")
    if model_cfg is None:
        print("ERROR: 'hunyuan3d' not found in config")
        sys.exit(1)

    pred_root = model_cfg["predictions_root"]
    mesh_fn = model_cfg["mesh_filename"]
    inf_params = get_inference_params(cfg, "hunyuan3d")
    num_inference_steps = inf_params["num_inference_steps"]
    guidance_scale = inf_params["guidance_scale"]
    octree_resolution = inf_params["octree_resolution"]
    num_chunks = inf_params.get("num_chunks", 8000)

    print(f"Hunyuan3D-2.0 teacher profiling | steps={num_inference_steps} "
          f"guidance={guidance_scale} octree_res={octree_resolution} "
          f"num_chunks={num_chunks} | {len(samples)} samples")

    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from PIL import Image
    import trimesh
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    print("Loading Hunyuan3D-2.0 pipeline...")
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2")

    # Warmup
    if samples:
        warmup_s = samples[0]
        print(f"  [WARMUP] {warmup_s.object_id}")
        warmup_img = Image.open(warmup_s.input_image).convert("RGBA")
        generator = torch.manual_seed(args.seed)
        with torch.inference_mode():
            profile_one_sample(
                pipeline, warmup_img, num_inference_steps, guidance_scale,
                octree_resolution, num_chunks, generator,
            )
        torch.cuda.empty_cache()
        print("  [WARMUP] done")

    for i, s in enumerate(samples):
        out_dir = os.path.join(pred_root, s.object_id)
        os.makedirs(out_dir, exist_ok=True)

        print(f"  [{i + 1}/{len(samples)}] {s.object_id}")

        try:
            image = Image.open(s.input_image).convert("RGBA")
            torch.cuda.reset_peak_memory_stats()
            generator = torch.manual_seed(args.seed)

            t_total_start = sync_and_time()
            with torch.inference_mode():
                mesh_list, block_timings = profile_one_sample(
                    pipeline, image, num_inference_steps, guidance_scale,
                    octree_resolution, num_chunks, generator,
                )
            t_total = sync_and_time() - t_total_start

            mesh = mesh_list[0]
            peak_mem = torch.cuda.max_memory_allocated() / 1e9

            # Save mesh
            mesh.export(os.path.join(out_dir, mesh_fn))

            profile = {
                "object_id": s.object_id,
                "category": s.category,
                "model": "hunyuan3d",
                "inference_params": {
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "octree_resolution": octree_resolution,
                },
                "seed": args.seed,
                "blocks": {k: {"wall_sec": round(v, 4)} for k, v in block_timings.items()},
                "blocks_sum_sec": round(sum(block_timings.values()), 4),
                "total_sec": round(t_total, 4),
                "peak_gpu_gb": round(peak_mem, 2),
                "raw_vertices": int(len(mesh.vertices)),
                "raw_faces": int(len(mesh.faces)),
            }

            with open(os.path.join(out_dir, "profile.json"), "w") as f:
                json.dump(profile, f, indent=2)

            block_str = " | ".join(f"{k}={v:.2f}s" for k, v in block_timings.items())
            print(f"    total={t_total:.2f}s | {block_str}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"    FAILED: {e}")
            profile = {
                "object_id": s.object_id,
                "model": "hunyuan3d",
                "status": "failed",
                "error": str(e),
            }
            with open(os.path.join(out_dir, "profile.json"), "w") as f:
                json.dump(profile, f, indent=2)


if __name__ == "__main__":
    main()
