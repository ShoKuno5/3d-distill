#!/usr/bin/env python3
"""Diag-1 + Diag-2: Test discriminator architecture capacity on VecSet features.

Generates real (GT latent) and fake (student 1-step) features via mu_fake,
then trains fresh mean-pooling vs token-wise classifiers to measure
the information loss from mean pooling.

Usage:
    cd /mnt/workspace/kuno/distillation/models/hunyuan3d21/hy3dshape
    PYTHONPATH=.:../../../distill_methods/src CUDA_VISIBLE_DEVICES=0 \
        ../../../envs/hunyuan3d-venv/bin/python -u \
        ../../../distill_methods/scripts/diagnose_discriminator.py \
        --config ../../../distill_methods/config.yaml \
        --num-samples 50
"""

import argparse
import copy
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]


def load_training_data(data_dir: str, num_samples: int):
    """Load pre-encoded GT latents + conditions from training data."""
    files = sorted(Path(data_dir).glob("*.npz"))[:num_samples]
    latents, conds = [], []
    for f in files:
        data = np.load(f)
        latents.append(torch.from_numpy(data["latent"].astype(np.float32)))
        conds.append(torch.from_numpy(data["image_cond"].astype(np.float32)))
    return torch.stack(latents), torch.stack(conds)


def setup_model(config, ckpt_dir_override=None):
    """Load teacher, student (with fake_score adapter), and register hooks."""
    from base_distiller import _load_pipeline
    from peft import PeftModel, LoraConfig, get_peft_model
    from dmd1_distillation import FakeScoreAdapter

    model_cfg = config["training"]["model"]
    lora_cfg = model_cfg["lora"]
    device = torch.device("cuda")

    # Load pipeline
    logger.info("Loading pipeline from %s ...", model_cfg["pretrained"])
    teacher, pipeline = _load_pipeline(model_cfg["pretrained"])
    teacher.to(device).to(torch.bfloat16).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Build student with LoRA
    student = copy.deepcopy(teacher)
    student.to(device)
    for p in student.parameters():
        p.requires_grad_(True)

    peft_config = LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
    )
    student = get_peft_model(student, peft_config)

    # Load DMD2 checkpoint (student LoRA)
    ckpt_dir = ckpt_dir_override or (config["output_root"] + "/checkpoints/dmd2")
    ckpt_steps = sorted([
        d for d in os.listdir(ckpt_dir)
        if os.path.isdir(os.path.join(ckpt_dir, d)) and d.startswith("step_")
    ], key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0)
    if ckpt_steps:
        ckpt_path = os.path.join(ckpt_dir, ckpt_steps[-1])
        logger.info("Loading student from %s", ckpt_path)
        base = copy.deepcopy(student.base_model.model)
        student = PeftModel.from_pretrained(base, ckpt_path, is_trainable=False)
    student.eval()

    # Add fake_score adapter
    fake_adapter = FakeScoreAdapter(student, lora_cfg)
    # Load fake_score weights if available
    if ckpt_steps:
        fs_path = os.path.join(ckpt_dir, ckpt_steps[-1], "fake_score")
        if os.path.exists(fs_path):
            logger.info("Loading fake_score from %s", fs_path)
            student.load_adapter(fs_path, adapter_name="fake_score")

    # Register feature hooks on student blocks
    features_store = {}
    student_dit = student.base_model.model if hasattr(student, "base_model") else student
    if hasattr(student_dit, "single_blocks") and len(student_dit.single_blocks) > 0:
        blocks = student_dit.single_blocks
    elif hasattr(student_dit, "blocks"):
        blocks = student_dit.blocks
    else:
        raise RuntimeError("Cannot find transformer blocks")

    hook_idx = len(blocks) // 2
    logger.info("Hooking block %d/%d", hook_idx, len(blocks))

    def hook_fn(module, input, output):
        features_store["feat"] = output

    blocks[hook_idx].register_forward_hook(hook_fn)

    return student, fake_adapter, features_store, device


def extract_features(student, fake_adapter, features_store, latents, conds,
                     device, batch_size=2, is_real=True):
    """Extract mu_fake features for a set of latents.

    For 'real': diffuse GT latent to random t, run through mu_fake, capture features.
    For 'fake': generate student 1-step from noise, diffuse, run through mu_fake.
    """
    all_feats_pre_pool = []  # [N_total, 4096, D]
    all_feats_post_pool = []  # [N_total, D]

    N = latents.shape[0]
    for i in range(0, N, batch_size):
        b_latent = latents[i:i+batch_size].to(device)
        b_cond = conds[i:i+batch_size].to(device)
        B = b_latent.shape[0]
        contexts = {"main": b_cond}

        if not is_real:
            # Generate fake: student 1-step from noise
            noise = torch.randn_like(b_latent)
            fake_adapter.activate_student()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                v = student(noise, torch.zeros(B, device=device), contexts=contexts)
            x = noise + 1.0 * v  # euler step dt=1
        else:
            x = b_latent

        # Diffuse to random t, run through mu_fake
        t = torch.rand(B, device=device) * 0.998 + 0.001
        eps = torch.randn_like(x)
        t_exp = t.view(-1, *([1] * (x.dim() - 1)))
        x_noised = t_exp * x + (1 - t_exp) * eps

        fake_adapter.activate_fake_score()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            student(x_noised, t, contexts=contexts)

        feat = features_store.get("feat")
        if feat is None:
            logger.warning("Hook failed at batch %d", i)
            continue

        feat = feat.float().detach().cpu()
        all_feats_pre_pool.append(feat)
        all_feats_post_pool.append(feat.mean(dim=1))

        if (i // batch_size) % 10 == 0:
            logger.info("  Extracted %d/%d (%s)", min(i+batch_size, N), N,
                        "real" if is_real else "fake")

    fake_adapter.activate_student()
    return torch.cat(all_feats_pre_pool), torch.cat(all_feats_post_pool)


def run_classification_test(feat_real, feat_fake, label: str):
    """Train logistic regression classifier and report accuracy."""
    N = min(len(feat_real), len(feat_fake))
    X = torch.cat([feat_real[:N], feat_fake[:N]]).numpy()
    y = np.concatenate([np.ones(N), np.zeros(N)])

    # Shuffle
    perm = np.random.RandomState(42).permutation(2 * N)
    X, y = X[perm], y[perm]

    # Split 70/30
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))

    logger.info("  [%s] Train acc: %.3f  Test acc: %.3f  (N=%d)",
                label, train_acc, test_acc, N)
    return test_acc


def analyze_features(feat_real_pre, feat_fake_pre, feat_real_post, feat_fake_post):
    """Diag-2: Analyze feature space information content."""
    logger.info("\n=== Feature Space Analysis ===")

    # Post-pool statistics
    cos_sim_post = nn.functional.cosine_similarity(
        feat_real_post.mean(0, keepdim=True),
        feat_fake_post.mean(0, keepdim=True),
    ).item()
    l2_post = (feat_real_post.mean(0) - feat_fake_post.mean(0)).norm().item()
    logger.info("Post-pool: mean cosine_sim(real, fake) = %.6f, L2 = %.4f",
                cos_sim_post, l2_post)

    # Per-sample cosine similarity (post-pool)
    N = min(len(feat_real_post), len(feat_fake_post))
    cos_per_sample = nn.functional.cosine_similarity(
        feat_real_post[:N], feat_fake_post[:N], dim=1
    )
    logger.info("Post-pool per-sample cosine: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
                cos_per_sample.mean(), cos_per_sample.std(),
                cos_per_sample.min(), cos_per_sample.max())

    # Pre-pool: per-token analysis (sample a few)
    n_check = min(5, len(feat_real_pre))
    for i in range(n_check):
        cos_tokens = nn.functional.cosine_similarity(
            feat_real_pre[i], feat_fake_pre[i], dim=1
        )  # [4096]
        logger.info("Sample %d pre-pool token cosine: mean=%.4f, std=%.4f, min=%.4f",
                    i, cos_tokens.mean(), cos_tokens.std(), cos_tokens.min())

    # Norm distributions
    real_norms = feat_real_pre.norm(dim=-1).mean(dim=0)  # [4096]
    fake_norms = feat_fake_pre.norm(dim=-1).mean(dim=0)
    logger.info("Token norms: real mean=%.2f, fake mean=%.2f",
                real_norms.mean(), fake_norms.mean())

    # Effective dimension via PCA (post-pool)
    from sklearn.decomposition import PCA
    for name, feats in [("real_post", feat_real_post), ("fake_post", feat_fake_post)]:
        pca = PCA(n_components=min(50, feats.shape[0]-1))
        pca.fit(feats.numpy())
        var_ratio = pca.explained_variance_ratio_
        cum_var = np.cumsum(var_ratio)
        eff_dim = np.searchsorted(cum_var, 0.95) + 1
        logger.info("PCA %s: effective_dim(95%%)=%d, top-1=%.3f, top-5=%.3f",
                    name, eff_dim, cum_var[0], cum_var[min(4, len(cum_var)-1)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt-dir", default=None, help="Override checkpoint dir for DMD2")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load data
    data_dir = config["training"]["training_data_dir"]
    logger.info("Loading %d samples from %s", args.num_samples, data_dir)
    latents, conds = load_training_data(data_dir, args.num_samples)
    logger.info("Loaded: latents %s, conds %s", latents.shape, conds.shape)

    # Setup model
    student, fake_adapter, features_store, device = setup_model(config, args.ckpt_dir)

    # Extract features
    logger.info("\n=== Extracting real features ===")
    feat_real_pre, feat_real_post = extract_features(
        student, fake_adapter, features_store, latents, conds,
        device, args.batch_size, is_real=True)
    logger.info("Real features: pre_pool %s, post_pool %s",
                feat_real_pre.shape, feat_real_post.shape)

    logger.info("\n=== Extracting fake features ===")
    feat_fake_pre, feat_fake_post = extract_features(
        student, fake_adapter, features_store, latents, conds,
        device, args.batch_size, is_real=False)
    logger.info("Fake features: pre_pool %s, post_pool %s",
                feat_fake_pre.shape, feat_fake_post.shape)

    # Diag-2: Feature space analysis
    analyze_features(feat_real_pre, feat_fake_pre, feat_real_post, feat_fake_post)

    # Diag-1: Classification tests
    logger.info("\n=== Classification Tests ===")

    # Test 1: Mean-pooled features (current discriminator design)
    acc_mean_pool = run_classification_test(
        feat_real_post, feat_fake_post, "mean_pool (current D)")

    # Test 2: Token-wise — flatten per-token features, train on concatenated
    # Use random subset of tokens to keep feature dim manageable
    n_tokens_sample = 64
    torch.manual_seed(42)
    token_idx = torch.randperm(feat_real_pre.shape[1])[:n_tokens_sample]
    feat_real_tokens = feat_real_pre[:, token_idx].reshape(feat_real_pre.shape[0], -1)
    feat_fake_tokens = feat_fake_pre[:, token_idx].reshape(feat_fake_pre.shape[0], -1)
    acc_token_sample = run_classification_test(
        feat_real_tokens, feat_fake_tokens, "token_sample_64 (token-level info)")

    # Test 3: Per-token variance as features (statistical summary preserving structure)
    feat_real_stats = torch.cat([
        feat_real_pre.mean(dim=1),
        feat_real_pre.std(dim=1),
        feat_real_pre.max(dim=1).values,
        feat_real_pre.min(dim=1).values,
    ], dim=1)
    feat_fake_stats = torch.cat([
        feat_fake_pre.mean(dim=1),
        feat_fake_pre.std(dim=1),
        feat_fake_pre.max(dim=1).values,
        feat_fake_pre.min(dim=1).values,
    ], dim=1)
    acc_stats = run_classification_test(
        feat_real_stats, feat_fake_stats, "token_stats (mean+std+max+min)")

    # Test 4: Random Gaussian noise as fake (baseline — how easy is the task?)
    noise_feats_pre = torch.randn_like(feat_real_pre)
    noise_feats_post = noise_feats_pre.mean(dim=1)
    acc_noise = run_classification_test(
        feat_real_post, noise_feats_post, "random_noise_baseline")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info("  mean_pool (current D design):  %.1f%%", acc_mean_pool * 100)
    logger.info("  token_sample_64:               %.1f%%", acc_token_sample * 100)
    logger.info("  token_stats (mean+std+max+min): %.1f%%", acc_stats * 100)
    logger.info("  random_noise baseline:          %.1f%%", acc_noise * 100)
    logger.info("")
    if acc_mean_pool < 0.7 and acc_token_sample > 0.8:
        logger.info("CONCLUSION: Mean pooling destroys discriminative information.")
        logger.info("  Token-level features CAN distinguish real/fake.")
        logger.info("  → Issue A CONFIRMED: Discriminator architecture is the bottleneck.")
    elif acc_mean_pool > 0.8:
        logger.info("CONCLUSION: Mean pooling retains sufficient information.")
        logger.info("  → Issue A NOT confirmed. Look at training dynamics (Issue B/C).")
    else:
        logger.info("CONCLUSION: Both architectures struggle. Task may be inherently hard.")
        logger.info("  → Further investigation needed.")


if __name__ == "__main__":
    main()
