"""Distribution Matching Distillation (DMD1) for Hunyuan3D-2.1.

Yin et al. (2023): Minimize KL divergence between student and teacher output
distributions using a learned "fake score" network, plus a regression loss
against pre-generated teacher outputs for stability.

Three networks in memory:
  - Teacher (frozen, original DiT)
  - Student (DiT + LoRA)
  - Fake score (DiT + separate LoRA) -- learns score of student distribution

Loss_student = L_distill (KL via score diff) + lambda_reg * L_regress (MSE pairs)
"""

import logging
import os
from glob import glob

import numpy as np
import torch
import torch.nn as nn

from base_distiller import BaseDistiller

logger = logging.getLogger(__name__)


class FakeScoreAdapter:
    """Manages a separate LoRA adapter on the same DiT for the fake score network.

    Instead of loading a third copy of the DiT, we use peft's adapter
    switching to maintain two LoRA adapters (student + fake_score) on one model.
    """

    def __init__(self, student_peft_model, lora_cfg: dict):
        from peft import LoraConfig

        # Add a second adapter named "fake_score"
        fake_config = LoraConfig(
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg["alpha"],
            target_modules=lora_cfg["target_modules"],
        )
        student_peft_model.add_adapter("fake_score", fake_config)
        self.model = student_peft_model
        # Default adapter is "default" (student)

    def activate_student(self):
        self.model.set_adapter("default")

    def activate_fake_score(self):
        self.model.set_adapter("fake_score")

    def fake_score_params(self):
        """Return only fake_score adapter parameters."""
        self.activate_fake_score()
        params = [
            p for n, p in self.model.named_parameters()
            if "fake_score" in n and p.requires_grad
        ]
        self.activate_student()
        return params


class RegressionPairsDataset(torch.utils.data.Dataset):
    """Pre-generated (noise, x_teacher) pairs for DMD1 regression loss."""

    def __init__(self, pairs_dir: str, num_pairs: int):
        # Use explicit index probing to avoid stale NFS/CPFS directory cache
        self.files = sorted([
            os.path.join(pairs_dir, f"pair_{i:06d}.npz")
            for i in range(num_pairs)
            if os.path.isfile(os.path.join(pairs_dir, f"pair_{i:06d}.npz"))
        ])
        if not self.files:
            raise FileNotFoundError(f"No pair files in {pairs_dir}")
        if len(self.files) < num_pairs:
            logger.warning(
                "RegressionPairs: found %d/%d pairs (some indices missing)",
                len(self.files), num_pairs,
            )
        logger.info("RegressionPairs: %d pairs from %s", len(self.files), pairs_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        return {
            "noise": torch.from_numpy(data["noise"].astype(np.float32)),
            "x_teacher": torch.from_numpy(data["x_teacher"].astype(np.float32)),
            "image_cond": torch.from_numpy(data["image_cond"].astype(np.float32)),
        }


class DMD1Distillation(BaseDistiller):
    """DMD1: KL distillation via fake score + regression."""

    def __init__(self, config):
        super().__init__(config)

        dmd1_cfg = config["training"]["methods"]["dmd1"]
        lora_cfg = config["training"]["model"]["lora"]
        self.lambda_reg = dmd1_cfg["lambda_reg"]

        # Setup fake score adapter on the student model
        model = self.student.module if self.is_distributed else self.student
        self.fake_score_adapter = FakeScoreAdapter(model, lora_cfg)

        # Optimizers: one for student, one for fake score
        self._setup_optimizer()

        # Regression pairs dataloader (lazy, set up in train())
        self.pairs_dir = dmd1_cfg["pairs_dir"]
        self._pairs_loader = None
        self._pairs_iter = None

    def _method_name(self) -> str:
        return "dmd1"

    def _setup_optimizer(self):
        opt_cfg = self.config["training"]["optimizer"]
        model = self.student.module if self.is_distributed else self.student

        # Student optimizer (default adapter params)
        self.fake_score_adapter.activate_student()
        student_params = [
            p for n, p in model.named_parameters()
            if "fake_score" not in n and p.requires_grad
        ]
        self.optimizer = torch.optim.AdamW(
            student_params, lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"],
        )

        # Fake score optimizer
        fake_params = self.fake_score_adapter.fake_score_params()
        self.optimizer_fake = torch.optim.AdamW(
            fake_params, lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"],
        )

    # Flag: we handle backward ourselves
    _custom_backward = True

    def _get_pairs_batch(self):
        """Get next batch of regression pairs."""
        if self._pairs_loader is None:
            num_pairs = self.config["training"]["methods"]["dmd1"]["num_pairs"]
            pairs_ds = RegressionPairsDataset(self.pairs_dir, num_pairs)
            self._pairs_loader = torch.utils.data.DataLoader(
                pairs_ds,
                batch_size=self.config["training"]["batch_size"],
                shuffle=True,
                num_workers=2,
                drop_last=True,
                pin_memory=True,
            )
            self._pairs_iter = iter(self._pairs_loader)

        try:
            batch = next(self._pairs_iter)
        except StopIteration:
            self._pairs_iter = iter(self._pairs_loader)
            batch = next(self._pairs_iter)

        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    # ------------------------------------------------------------------
    # Fake score training
    # ------------------------------------------------------------------

    def _train_fake_score(self, batch: dict):
        """Update fake score network via denoising score matching on student outputs.

        1. Generate x_fake = student(noise) in 1-step (no grad)
        2. Diffuse x_fake to random t: x_noised = diffuse(x_fake, t, eps)
        3. Train fake_score to denoise: predict eps from x_noised
        """
        x_data = batch["latent"]
        image_cond = batch["image_cond"]
        B = x_data.shape[0]
        contexts = {"main": image_cond}

        noise = torch.randn_like(x_data)

        # Student 1-step generation (no grad)
        with torch.no_grad():
            self.fake_score_adapter.activate_student()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v_student = self.student_forward(noise, torch.zeros(B, device=self.device), contexts)
            # x_fake = noise + 1.0 * v_student (full step from t=0 to t=1)
            x_fake = self.euler_step(noise, v_student, torch.ones(B, device=self.device))

        # Diffuse x_fake
        t_probe = self.sample_t(B, self.device)
        eps = torch.randn_like(x_fake)
        x_noised = self.diffuse(x_fake, t_probe, eps)

        # Fake score prediction
        self.fake_score_adapter.activate_fake_score()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v_fake = self.student_forward(x_noised, t_probe, contexts)

        # Target: the true velocity for this diffusion (x_fake - eps)
        v_target = self.get_velocity(x_fake, eps)
        loss_fake = nn.functional.mse_loss(v_fake.float(), v_target.float().detach())

        self.optimizer_fake.zero_grad()
        loss_fake.backward()
        torch.nn.utils.clip_grad_norm_(
            self.fake_score_adapter.fake_score_params(),
            self.config["training"]["gradient_clip"],
        )
        self.optimizer_fake.step()

        # Switch back to student
        self.fake_score_adapter.activate_student()
        return loss_fake.detach()

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, step: int) -> dict:
        """DMD1 training step (Yin et al. 2024, Algorithm 1).

        Phase 1: Update fake score via DSM on student outputs.
        Phase 2: Update student with L_distill + lambda_reg * L_regress.
        """
        # --- Phase 1: Update fake score ---
        loss_fake = self._train_fake_score(batch)

        # --- Phase 2: Update student ---
        x_data = batch["latent"]
        image_cond = batch["image_cond"]
        B = x_data.shape[0]
        contexts = {"main": image_cond}
        noise = torch.randn_like(x_data)

        # Student 1-step generation
        self.fake_score_adapter.activate_student()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v_student = self.student_forward(
                noise, torch.zeros(B, device=self.device), contexts,
            )
        x_gen = self.euler_step(noise, v_student, torch.ones(B, device=self.device))

        # KL gradient via score difference (paper Eq. 7, Algorithm 2)
        t_kl = self.sample_t(B, self.device)
        eps_kl = torch.randn_like(x_gen)
        x_gen_noised = self.diffuse(x_gen, t_kl, eps_kl)

        # Denoiser predictions: mu(x_t, t) = x_t + (1-t) * v(x_t, t)
        # Paper uses denoiser outputs (not raw velocity) for score difference.
        t_expand = t_kl.view(-1, *([1] * (x_gen.dim() - 1)))
        with torch.no_grad():
            self.fake_score_adapter.activate_fake_score()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v_fake_at_gen = self.student_forward(x_gen_noised, t_kl, contexts)
            self.fake_score_adapter.activate_student()

            v_teacher_at_gen = self.teacher_forward(x_gen_noised, t_kl, contexts)

            # Convert velocity to denoiser output (paper Eq. 4-5)
            mu_fake = x_gen_noised + (1.0 - t_expand) * v_fake_at_gen
            mu_teacher = x_gen_noised + (1.0 - t_expand) * v_teacher_at_gen

            # Adaptive weighting (paper Eq. 8): normalize by denoising error
            weight_denom = (mu_teacher - x_gen).abs().mean(
                dim=list(range(1, x_gen.dim())), keepdim=True,
            ).clamp(min=1e-6)
            grad_direction = (mu_fake - mu_teacher) / weight_denom

        # Stop-gradient trick (paper Algorithm 2):
        # loss = 0.5 * MSE(x, sg(x - grad)), gradient = grad * dx/dtheta
        target = (x_gen - grad_direction).detach()
        loss_distill = 0.5 * nn.functional.mse_loss(x_gen, target)

        # Regression loss (against pre-generated pairs)
        pairs_batch = self._get_pairs_batch()
        pair_noise = pairs_batch["noise"]
        pair_teacher = pairs_batch["x_teacher"]
        pair_cond = pairs_batch["image_cond"]
        pair_contexts = {"main": pair_cond}

        with torch.autocast("cuda", dtype=torch.bfloat16):
            v_reg = self.student_forward(
                pair_noise, torch.zeros(pair_noise.shape[0], device=self.device),
                pair_contexts,
            )
        x_reg = self.euler_step(
            pair_noise, v_reg, torch.ones(pair_noise.shape[0], device=self.device),
        )
        loss_regress = nn.functional.mse_loss(x_reg, pair_teacher)

        # Total student loss
        loss = loss_distill + self.lambda_reg * loss_regress

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.student.parameters() if p.requires_grad],
            self.config["training"]["gradient_clip"],
        )
        self.optimizer.step()

        return {
            "loss": loss.detach(),
            "loss_distill": loss_distill.detach(),
            "loss_regress": loss_regress.detach(),
            "loss_fake_score": loss_fake,
        }
