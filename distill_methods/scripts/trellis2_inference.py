"""TRELLIS.2 inference adapter for Toys4k eval pipeline.

Loads microsoft/TRELLIS.2-4B and runs inference on each sample in the
Toys4k test manifest. Outputs mesh.obj per object_id under
<output_root>/predictions/trellis2/default/<oid>/mesh.obj
(matching our cross-family-bench config convention).
"""
from __future__ import annotations

import csv
import os
import sys
import time
import traceback
from pathlib import Path

# Repo: /gs/fs/tga-koike-shanda2/sk/models/trellis2 cloned earlier (must be on PYTHONPATH).

def load_pipeline():
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()
    return pipeline


def export_mesh(mesh, mesh_path: str):
    """Try several export strategies."""
    # 1) trimesh-compatible .export(...)
    if hasattr(mesh, "export"):
        try:
            mesh.export(mesh_path)
            return "trimesh.export"
        except Exception as e1:
            print(f"  trimesh.export failed: {e1}")
    # 2) raw vertices + faces via trimesh
    try:
        import trimesh
        verts = getattr(mesh, "vertices", None)
        faces = getattr(mesh, "faces", None)
        if verts is not None and faces is not None:
            import numpy as np
            tm = trimesh.Trimesh(
                vertices=verts.cpu().numpy() if hasattr(verts, "cpu") else np.asarray(verts),
                faces=faces.cpu().numpy() if hasattr(faces, "cpu") else np.asarray(faces),
            )
            tm.export(mesh_path)
            return "trimesh.Trimesh"
    except Exception as e2:
        print(f"  manual trimesh export failed: {e2}")
    raise RuntimeError(f"all export strategies failed for {mesh_path}")


def main():
    manifest_path = os.environ.get(
        "TRELLIS2_MANIFEST",
        "/gs/fs/tga-koike-shanda2/sk/3d-distill/distill_methods/manifests/toys4k_eval_105/test.csv",
    )
    output_root = os.environ.get(
        "TRELLIS2_OUTPUT",
        "/gs/fs/tga-koike-shanda2/sk/scratch/distill_methods/m2_cd_tsubame_20260519_1629/predictions/trellis2/default",
    )
    # Optional intra-node sharding (one worker per GPU, samples sliced by index).
    shard_index = int(os.environ.get("TRELLIS2_SHARD_INDEX", "0"))
    num_shards = int(os.environ.get("TRELLIS2_NUM_SHARDS", "1"))
    if num_shards < 1 or not (0 <= shard_index < num_shards):
        raise SystemExit(f"invalid sharding: shard_index={shard_index}, num_shards={num_shards}")

    Path(output_root).mkdir(parents=True, exist_ok=True)

    from PIL import Image

    with open(manifest_path) as f:
        samples = list(csv.DictReader(f))
    print(f"manifest: {manifest_path}, n={len(samples)}")
    if num_shards > 1:
        before = len(samples)
        samples = samples[shard_index::num_shards]
        print(f"shard {shard_index}/{num_shards}: {len(samples)}/{before} samples")

    # Filter to missing predictions
    todo = []
    existing = 0
    for s in samples:
        oid = s["object_id"]
        mesh_path = Path(output_root) / oid / "mesh.obj"
        if mesh_path.exists() and mesh_path.stat().st_size > 0:
            existing += 1
        else:
            todo.append(s)
    print(f"existing: {existing}, todo: {len(todo)}")

    if not todo:
        print("nothing to do")
        return

    print("loading TRELLIS.2-4B pipeline...")
    t0 = time.time()
    pipeline = load_pipeline()
    print(f"  loaded in {time.time()-t0:.1f}s")

    n_ok, n_fail = 0, 0
    for i, s in enumerate(todo, 1):
        oid = s["object_id"]
        out_dir = Path(output_root) / oid
        out_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = out_dir / "mesh.obj"
        t = time.time()
        try:
            img = Image.open(s["input_image"]).convert("RGB")
            outputs = pipeline.run(img)
            mesh = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            method = export_mesh(mesh, str(mesh_path))
            dt = time.time() - t
            print(f"  [{i}/{len(todo)}] OK {oid}: {dt:.1f}s via {method}")
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"  [{i}/{len(todo)}] FAIL {oid}: {e}")
            traceback.print_exc(limit=3)

    print(f"Done. ok={n_ok}, fail={n_fail}")


if __name__ == "__main__":
    main()
