"""Toys4k dataset loading and subset management."""

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class Sample:
    object_id: str
    category: str
    mesh_obj: str
    point_cloud: str
    input_image: str


def load_manifest(manifest_path: str) -> list[Sample]:
    """Load Toys4k manifest CSV."""
    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))
    return [
        Sample(
            object_id=r["object_id"],
            category=r["category"],
            mesh_obj=r["mesh_obj"],
            point_cloud=r["point_cloud"],
            input_image=r["input_image"],
        )
        for r in rows
    ]


def filter_samples(
    samples: list[Sample],
    max_samples: Optional[int] = None,
    sample_ids_file: Optional[str] = None,
    category_filter: Optional[list[str]] = None,
) -> list[Sample]:
    """Apply subset filters to sample list."""
    if sample_ids_file and os.path.exists(sample_ids_file):
        with open(sample_ids_file) as f:
            ids = {line.strip() for line in f if line.strip()}
        samples = [s for s in samples if s.object_id in ids]

    if category_filter:
        samples = [s for s in samples if s.category in category_filter]

    if max_samples and max_samples > 0:
        samples = samples[:max_samples]

    return samples


def load_gt_pointcloud(pc_path: str) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Load GT point cloud and normals from Toys4k npz.

    Returns:
        (points, normals) where normals may be None.
        Points shape: (N, 3), float64.
    """
    data = np.load(pc_path, allow_pickle=True)

    # Toys4k format: data['dct'].item() is a dict with 'pc' and 'normals'
    if "dct" in data:
        dct = data["dct"].item()
        if isinstance(dct, dict) and "pc" in dct:
            pts = np.array(dct["pc"], dtype=np.float64)
            normals = None
            if "normals" in dct:
                normals = np.array(dct["normals"], dtype=np.float64)
            return pts, normals

    # Fallback for other formats
    for key in ["points", "pts", "point_cloud", "pc"]:
        if key in data:
            pts = np.array(data[key], dtype=np.float64)
            return pts, None

    arr = data[data.files[0]]
    if arr.dtype == object:
        arr = arr.item()
        if isinstance(arr, dict):
            pts = np.array(arr.get("pc", list(arr.values())[0]), dtype=np.float64)
            normals = np.array(arr["normals"], dtype=np.float64) if "normals" in arr else None
            return pts, normals

    return np.array(arr, dtype=np.float64), None


def save_sample_ids(samples: list[Sample], path: str) -> None:
    """Save list of sample IDs used in this run."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(f"{s.object_id}\n")
