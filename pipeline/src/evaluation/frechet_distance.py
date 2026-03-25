"""Frechet Distance computation on multiview renders.

Computes FD using InceptionV3 or DINOv2 feature extractors.
FD is a distributional metric: one scalar per model (not per-sample).

Usage:
    from pipeline.src.evaluation.frechet_distance import extract_features, compute_fd

    gt_feats = extract_features(gt_image_paths, model='inception_v3')
    pred_feats = extract_features(pred_image_paths, model='inception_v3')
    fd = compute_fd(gt_feats, pred_feats)
"""

from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ImagePathDataset(Dataset):
    """Dataset that loads images from file paths."""

    def __init__(self, image_paths: list[str], transform):
        self.paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def _get_inception_model():
    """Load InceptionV3 and return model + transform.

    Returns pool3 features (2048-dim) before the final classification layer.
    """
    from torchvision.models import inception_v3, Inception_V3_Weights

    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights)
    # Remove the final FC layer to get pool3 features
    model.fc = nn.Identity()
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(299, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return model, transform, 2048


def _get_dinov2_model():
    """Load DINOv2 ViT-L/14 with registers and return model + transform.

    Returns CLS token features (1024-dim).
    """
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return model, transform, 1024


def extract_features(
    image_paths: list[str],
    model_name: Literal["inception_v3", "dinov2"] = "inception_v3",
    batch_size: int = 32,
    device: str = "cuda",
    num_workers: int = 4,
) -> np.ndarray:
    """Extract features from images using the specified model.

    Args:
        image_paths: List of image file paths.
        model_name: Feature extractor to use.
        batch_size: Batch size for feature extraction.
        device: Device to run on.
        num_workers: DataLoader workers.

    Returns:
        (N, D) feature array where D depends on the model.
    """
    if model_name == "inception_v3":
        model, transform, feat_dim = _get_inception_model()
    elif model_name == "dinov2":
        model, transform, feat_dim = _get_dinov2_model()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)

    dataset = ImagePathDataset(image_paths, transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    features = np.empty((len(image_paths), feat_dim), dtype=np.float64)
    idx = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            feats = model(batch)
            if isinstance(feats, tuple):
                feats = feats[0]
            feats = feats.cpu().numpy().astype(np.float64)
            features[idx:idx + len(feats)] = feats
            idx += len(feats)

    return features


def compute_fd(features_pred: np.ndarray, features_gt: np.ndarray) -> float:
    """Compute Frechet Distance between two feature distributions.

    FD = ||mu_pred - mu_gt||^2 + Tr(C_pred + C_gt - 2*(C_pred @ C_gt)^{1/2})

    Args:
        features_pred: (N, D) predicted feature array.
        features_gt: (M, D) ground truth feature array.

    Returns:
        Scalar FD value.
    """
    mu_pred = features_pred.mean(axis=0)
    mu_gt = features_gt.mean(axis=0)

    sigma_pred = np.cov(features_pred, rowvar=False)
    sigma_gt = np.cov(features_gt, rowvar=False)

    diff = mu_pred - mu_gt
    diff_sq = diff @ diff

    # Matrix square root of sigma_pred @ sigma_gt
    covmean, _ = linalg.sqrtm(sigma_pred @ sigma_gt, disp=False)

    # Handle numerical instability (small imaginary components)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError("Imaginary component in matrix square root is too large")
        covmean = covmean.real

    fd = diff_sq + np.trace(sigma_pred + sigma_gt - 2.0 * covmean)
    return float(fd)


def compute_fd_for_model(
    pred_render_dir: str,
    gt_render_dir: str,
    model_name: Literal["inception_v3", "dinov2"] = "inception_v3",
    batch_size: int = 32,
    device: str = "cuda",
) -> float:
    """Compute FD between predicted and GT multiview renders for a model.

    Expects directory structure:
        {render_dir}/{object_id}/view_{az}.png

    Args:
        pred_render_dir: Directory containing predicted mesh renders.
        gt_render_dir: Directory containing GT mesh renders.
        model_name: Feature extractor to use.
        batch_size: Batch size.
        device: Device.

    Returns:
        FD scalar.
    """
    pred_dir = Path(pred_render_dir)
    gt_dir = Path(gt_render_dir)

    pred_paths = sorted(str(p) for p in pred_dir.rglob("view_*.png"))
    gt_paths = sorted(str(p) for p in gt_dir.rglob("view_*.png"))

    if not pred_paths:
        raise ValueError(f"No prediction renders found in {pred_render_dir}")
    if not gt_paths:
        raise ValueError(f"No GT renders found in {gt_render_dir}")

    print(f"  FD ({model_name}): {len(pred_paths)} pred images, {len(gt_paths)} GT images")

    pred_features = extract_features(pred_paths, model_name, batch_size, device)
    gt_features = extract_features(gt_paths, model_name, batch_size, device)

    return compute_fd(pred_features, gt_features)
