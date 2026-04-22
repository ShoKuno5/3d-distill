"""Score Identity Distillation (SiD) for Hunyuan3D-2.1.

Data-free one-step distillation using the score identity:
  ∇log p_θ(x) = E[∇log p_teacher(x|x_θ)]

Only requires the pretrained teacher — no regression pairs (DMD1),
no GAN discriminator (DMD2). Optionally applies Long-Short Guidance
(LSG) for improved CFG handling.

Two trainable networks (both LoRA on the same DiT):
  - Student (default adapter): one-step generator
  - Fake score (fake_score adapter): learns score of student distribution

References:
  - SiD: arxiv 2404.04057 (ICML 2024)
  - SiD-LSG: arxiv 2406.01561 (ICLR 2025)
  - SiD-DiT: arxiv 2509.25127 (flow matching extension)
"""

import logging
import os

import torch
import torch.nn as nn

from base_distiller import BaseDistiller
from dmd1_distillation import FakeScoreAdapter

logger = logging.getLogger(__name__)


class SiDDistillation(BaseDistiller):
    """SiD: score identity distillation with Long-Short Guidance."""

    # Adapter switching means only one of {student, fake_score} contributes
    # to any given backward pass; DDP needs the permissive flag.
    _ddp_find_unused_parameters = True

    def __init__(self, config):
        super().__init__(config)

        sid_cfg = config["training"]["methods"]["sid"]

        self.lsg_teacher_scale = sid_cfg["lsg_teacher_scale"]
        self.lsg_fake_scale = sid_cfg.get("lsg_fake_scale", 0.0)
        self.grad_norm_type = sid_cfg.get("grad_norm_type", "adaptive")

        # fake_score_adapter is created in _add_extra_adapters() during
        # BaseDistiller.__init__, before DDP wrap.

        self._setup_optimizer()

    def _method_name(self) -> str:
        return "sid"

    def _add_extra_adapters(self):
        """Register fake_score LoRA adapter before DDP wrap."""
        lora_cfg = self.config["training"]["model"]["lora"]
        # self.student is still a plain PeftModel here (not yet DDP-wrapped).
        self.fake_score_adapter = FakeScoreAdapter(self.student, lora_cfg)

    # Flag: we handle backward/step ourselves (dual optimizers)
    _custom_backward = True

    def _setup_optimizer(self):
        opt_cfg = self.config["training"]["optimizer"]
        model = self.student.module if self.is_distributed else self.student

        # Student optimizer (default adapter params only)
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

    # ------------------------------------------------------------------
    # Fake score training (DSM on student-generated samples)
    # ------------------------------------------------------------------

    def _train_fake_score(self, batch: dict) -> torch.Tensor:
        """Update fake score network via denoising score matching.

        1. Generate x_fake = student(noise) in 1-step (no grad)
        2. Diffuse x_fake to random t
        3. Train fake_score to predict velocity from noised x_fake
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
                v_student = self.student_forward(
                    noise, torch.zeros(B, device=self.device), contexts,
                )
            x_fake = self.euler_step(noise, v_student, torch.ones(B, device=self.device))

        # Diffuse x_fake to random timestep
        t_probe = self.sample_t(B, self.device)
        eps = torch.randn_like(x_fake)
        x_noised = self.diffuse(x_fake, t_probe, eps)

        # Fake score prediction
        self.fake_score_adapter.activate_fake_score()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v_fake = self.student_forward(x_noised, t_probe, contexts)

        # Target: true velocity for this diffusion
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
        """SiD training step.

        Phase 1: Update fake score via DSM on student outputs.
        Phase 2: Update student via score identity loss (+ LSG).
        """
        # --- Phase 1: Update fake score ---
        loss_fake = self._train_fake_score(batch)

        # --- Phase 2: Score identity loss ---
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

        # Diffuse generated sample to random timestep
        t_kl = self.sample_t(B, self.device)
        eps_kl = torch.randn_like(x_gen)
        x_gen_noised = self.diffuse(x_gen, t_kl, eps_kl)

        # Score queries (no grad)
        t_expand = t_kl.view(-1, *([1] * (x_gen.dim() - 1)))
        with torch.no_grad():
            # Fake score (no CFG / reduced CFG)
            self.fake_score_adapter.activate_fake_score()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v_fake = self.student_forward(x_gen_noised, t_kl, contexts)
            self.fake_score_adapter.activate_student()

            # Teacher with LSG-enhanced CFG
            v_teacher = self.teacher_forward(
                x_gen_noised, t_kl, contexts,
                guidance_scale=self.lsg_teacher_scale,
            )

            # Convert velocity to denoiser output: mu = x_t + (1-t)*v
            mu_fake = x_gen_noised + (1.0 - t_expand) * v_fake
            mu_teacher = x_gen_noised + (1.0 - t_expand) * v_teacher

            # Gradient direction with adaptive weighting
            if self.grad_norm_type == "adaptive":
                weight_denom = (mu_teacher - x_gen).abs().mean(
                    dim=list(range(1, x_gen.dim())), keepdim=True,
                ).clamp(min=1e-6)
                grad_direction = (mu_fake - mu_teacher) / weight_denom
            else:
                grad_direction = mu_fake - mu_teacher

        # Stop-gradient trick: loss = 0.5 * MSE(x, sg(x - grad))
        target = (x_gen - grad_direction).detach()
        loss_sid = 0.5 * nn.functional.mse_loss(x_gen, target)

        # Student backward
        self.optimizer.zero_grad()
        loss_sid.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.student.parameters() if p.requires_grad],
            self.config["training"]["gradient_clip"],
        )
        self.optimizer.step()

        return {
            "loss": loss_sid.detach(),
            "loss_sid": loss_sid.detach(),
            "loss_fake_score": loss_fake,
        }

    # ------------------------------------------------------------------
    # Checkpointing (extends base to save optimizer_fake state)
    # ------------------------------------------------------------------

    def save_checkpoint(self, step):
        """Save base checkpoint + fake_score weights + optimizer_fake state."""
        super().save_checkpoint(step)

        save_dir = os.path.join(self.ckpt_dir, f"step_{step}")

        # PeftModel.save_pretrained() saves only the active adapter, so the
        # fake_score LoRA weights are not in the base checkpoint. Snapshot
        # them explicitly so resume does not silently re-initialize them.
        model = self.student.module if self.is_distributed else self.student
        fake_state = {
            n: p.data.detach().cpu().clone()
            for n, p in model.named_parameters()
            if "fake_score" in n
        }
        torch.save(fake_state, os.path.join(save_dir, "fake_score_adapter.pt"))
        logger.info("Saved fake_score adapter weights (%d tensors)", len(fake_state))

        # Append optimizer_fake state to existing train_state.pt
        train_state_path = os.path.join(save_dir, "train_state.pt")
        train_state = torch.load(train_state_path, map_location="cpu")
        train_state["optimizer_fake"] = self.optimizer_fake.state_dict()
        torch.save(train_state, train_state_path)
        logger.info("Saved optimizer_fake state")

    def load_checkpoint(self, path: str) -> int:
        """Load base checkpoint + fake_score weights + optimizer_fake state."""
        resume_step = super().load_checkpoint(path)

        # super() rebuilt the fake_score adapter with random weights via
        # _add_extra_adapters(). Copy the saved weights into those tensors;
        # optimizer_fake's parameter references stay valid because we are
        # modifying tensor data in place.
        fake_path = os.path.join(path, "fake_score_adapter.pt")
        if os.path.exists(fake_path):
            fake_state = torch.load(fake_path, map_location=self.device)
            model = self.student.module if self.is_distributed else self.student
            loaded = 0
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in fake_state:
                        p.data.copy_(fake_state[n])
                        loaded += 1
            if loaded != len(fake_state):
                logger.warning(
                    "fake_score restore: %d/%d tensors matched by name",
                    loaded, len(fake_state),
                )
            else:
                logger.info("Restored fake_score adapter weights (%d tensors)", loaded)
        else:
            logger.warning(
                "No fake_score_adapter.pt in %s; fake_score resumes from random init",
                path,
            )

        # Restore optimizer_fake state from train_state.pt
        train_state_path = os.path.join(path, "train_state.pt")
        if os.path.exists(train_state_path):
            train_state = torch.load(train_state_path, map_location=self.device)
            if "optimizer_fake" in train_state:
                self.optimizer_fake.load_state_dict(train_state["optimizer_fake"])
                logger.info("Restored optimizer_fake state")

        return resume_step
