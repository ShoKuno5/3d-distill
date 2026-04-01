"""Consistency Distillation (CD) for Hunyuan3D-2.1.

Song et al. (2023): Train a consistency function f(x_t, t) that maps any point
on the ODE trajectory to the same clean data estimate, using the teacher to
compute adjacent points and EMA for the target.

f(x, t) = x + (1-t) * v_student(x, t)   (boundary: f(x_1, 1) = x_1)

Loss = MSE( f_student(x_t, t),  f_EMA(x_{t-h}, t-h) )
"""

import logging

import torch

from base_distiller import BaseDistiller

logger = logging.getLogger(__name__)


class ConsistencyDistillation(BaseDistiller):
    """Consistency distillation with EMA-based target."""

    def __init__(self, config):
        super().__init__(config)
        self._setup_optimizer()

        cd_cfg = config["training"]["methods"]["cd"]
        self.num_timesteps = cd_cfg["num_timesteps"]
        # Step size for teacher ODE
        self.h = 1.0 / self.num_timesteps

    def _method_name(self) -> str:
        return "cd"

    def _setup_optimizer(self):
        cd_cfg = self.config["training"]["methods"]["cd"]
        self.optimizer = torch.optim.AdamW(
            [p for p in self.student.parameters() if p.requires_grad],
            lr=cd_cfg["lr"],
            weight_decay=0.01,
        )

    # ------------------------------------------------------------------
    # Consistency function
    # ------------------------------------------------------------------

    def consistency_fn(self, model_fn, x: torch.Tensor, t: torch.Tensor,
                       contexts: dict) -> torch.Tensor:
        """Evaluate consistency function: f(x, t) = x + (1-t) * v(x, t).

        At t=1 this reduces to x (boundary condition).
        At other times it predicts the clean data estimate.
        """
        v = model_fn(x, t, contexts)
        t_expand = t.view(-1, *([1] * (x.dim() - 1)))
        return x + (1.0 - t_expand) * v

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, step: int) -> dict:
        """CD training step.

        1. Sample t from discrete schedule, avoid t=0 boundary
        2. Create x_t = diffuse(x_data, t, noise)
        3. Teacher 1 Euler step backward: x_{t-h} from x_t
        4. Student consistency: f_student(x_t, t)
        5. EMA consistency target: f_EMA(x_{t-h}, t-h)  (no grad)
        6. Loss = MSE(student, EMA_target)
        """
        x_data = batch["latent"]
        image_cond = batch["image_cond"]
        B = x_data.shape[0]
        contexts = {"main": image_cond}

        noise = torch.randn_like(x_data)
        h = self.h

        # Sample t from (h, 1] -- exclude t=0 to avoid degenerate h-step
        t_indices = torch.randint(1, self.num_timesteps, (B,), device=self.device)
        t = t_indices.float() * h  # t in {h, 2h, ..., 1-h, 1}
        t = t.clamp(h, 1.0)

        # Create noisy sample at t
        x_t = self.diffuse(x_data, t, noise)

        # --- Teacher 1-step backward: get x_{t-h} ---
        # Move backward along the ODE: x_{t-h} = x_t - h * v_teacher(x_t, t)
        # Note: forward ODE goes from 0 to 1, so backward is -h
        v_teacher = self.teacher_forward(x_t, t, contexts)
        x_prev = self.euler_step(x_t, v_teacher, torch.tensor(-h, device=self.device))
        t_prev = (t - h).clamp(0.0, 1.0)

        # --- Student consistency function at (x_t, t) ---
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f_student = self.consistency_fn(self.student_forward, x_t, t, contexts)

        # --- EMA target at (x_{t-h}, t-h) ---
        with torch.no_grad():
            with self.ema_scope():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    f_target = self.consistency_fn(
                        self.student_forward, x_prev, t_prev, contexts,
                    )

        # Consistency loss
        loss = torch.nn.functional.mse_loss(f_student, f_target.detach())

        return {"loss": loss}
