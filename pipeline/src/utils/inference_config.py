"""Shared helpers for inference: config loading and sample filtering."""

from typing import Any, Dict, List, Optional


# Default inference settings per model (matches official model defaults).
DEFAULTS = {
    "trellis": {
        "ss_steps": 25,
        "slat_steps": 25,
        "ss_cfg": 5.0,
        "slat_cfg": 5.0,
    },
    "trellis2": {
        "pipeline_type": "1024_cascade",
    },
    "hunyuan3d": {
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
        "octree_resolution": 384,
    },
    "hunyuan3d21": {
        "num_inference_steps": 50,
        "guidance_scale": 5.0,    # match official demo default (FlowMatching pipeline)
        "octree_resolution": 384,
    },
    "mdt_dist": {
        "ss_steps": 2,
        "slat_steps": 2,
        "ss_cfg": 5.0,
        "slat_cfg": 1.0,
        "ss_cfg_interval": [0.5, 1.0],
        "slat_cfg_interval": [0.5, 1.0],
        "rescale_t": 1.0,
    },
    "flashvdm": {
        "num_inference_steps": 5,
        "octree_resolution": 380,
        "num_chunks": 200000,
    },
}


def get_model_config(cfg: dict, model_name: str) -> Optional[dict]:
    """Find a model entry by name in the top-level 'models' list."""
    for m in cfg.get("models", []):
        if m["name"] == model_name:
            return m
    return None


def get_inference_params(cfg: dict, model_name: str) -> Dict[str, Any]:
    """Return inference parameters for *model_name*.

    Looks for ``models[name].inference_params`` in *cfg*.
    Falls back to built-in defaults for any missing keys.
    """
    model_cfg = get_model_config(cfg, model_name)
    configured = {}
    if model_cfg is not None:
        configured = dict(model_cfg.get("inference_params", {}))

    defaults = DEFAULTS.get(model_name, {})
    # Merge: configured values override defaults
    merged = {**defaults, **configured}
    return merged


def load_and_filter_samples(
    cfg: dict,
    max_samples_override: Optional[int] = None,
    manifest_key: str = "manifest",
) -> List:
    """Load manifest and apply all configured filters consistently.

    Uses the same filtering logic as run_eval.py:
    max_samples, sample_ids_file, and category_filter.

    Args:
        cfg: Parsed YAML config dict.
        max_samples_override: CLI --max-samples value (overrides config).
        manifest_key: Which manifest to load from dataset config
            (e.g. "manifest" for training, "test_manifest" for evaluation).

    Returns:
        Filtered list of Sample objects.
    """
    from src.data.toys4k import load_manifest, filter_samples

    max_samples = max_samples_override or cfg["dataset"].get("max_samples")
    manifest_path = cfg["dataset"].get(manifest_key) or cfg["dataset"]["manifest"]
    samples = load_manifest(manifest_path)
    samples = filter_samples(
        samples,
        max_samples=max_samples,
        sample_ids_file=cfg["dataset"].get("sample_ids_file"),
        category_filter=cfg["dataset"].get("category_filter"),
    )
    return samples
