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
        """CD training step (Song et al. 2023, Algorithm 2).

        1. Sample t from discrete schedule: t in {0, h, ..., 1-h}
        2. Create x_t = diffuse(x_data, t, noise)
        3. Teacher 1 Euler step forward: x_{t+h} from x_t (toward data)
        4. Student consistency at noisier point: f_student(x_t, t)
        5. EMA consistency target at cleaner point: f_EMA(x_{t+h}, t+h)
        6. Loss = MSE(student, EMA_target)
        """
        x_data = batch["latent"]
        image_cond = batch["image_cond"]
        B = x_data.shape[0]
        contexts = {"main": image_cond}

        noise = torch.randn_like(x_data)
        h = self.h

        # Sample t in {0, h, 2h, ..., 1-h} — leave room for teacher +h step.
        # Paper: n ~ U{1,N-1}, student at t_{n+1} (noisy), EMA at t_n (clean).
        # In flow-matching convention (t=0 noise, t=1 data) this maps to
        # student at t (noisier) and EMA at t+h (cleaner, closer to boundary).
        # Exclude t=0 (pure noise boundary) — Song et al. 2023 samples n in {1,...,N-1}
        t_indices = torch.randint(1, self.num_timesteps, (B,), device=self.device)
        t = t_indices.float() * h  # t in {h, 2h, ..., 1-h}

        # Create noisy sample at t
        x_t = self.diffuse(x_data, t, noise)

        # --- Teacher 1-step forward: get x_{t+h} (closer to data) ---
        # Paper Eq. (6): x̂_{t_n} = x_{t_{n+1}} + (t_n - t_{n+1}) * Φ(...)
        # In flow-matching: step +h along velocity toward data.
        v_teacher = self.teacher_forward(x_t, t, contexts)
        x_next = self.euler_step(x_t, v_teacher, torch.tensor(h, device=self.device))
        t_next = (t + h).clamp(0.0, 1.0)

        # --- Student consistency at noisier point (x_t, t) ---
        # Paper: f_θ(x_{t_{n+1}}, t_{n+1}) — online model at noisy side
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f_student = self.consistency_fn(self.student_forward, x_t, t, contexts)

        # --- EMA target at cleaner point (x_{t+h}, t+h) ---
        # Paper: f_{θ^-}(x̂_{t_n}, t_n) — EMA at clean side (anchored by boundary)
        with torch.no_grad():
            with self.ema_scope():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    f_target = self.consistency_fn(
                        self.student_forward, x_next, t_next, contexts,
                    )

        # Consistency loss — paper Eq. (7) with d(x,y) = ||x-y||² and λ=1
        loss = torch.nn.functional.mse_loss(f_student, f_target.detach())

        return {"loss": loss}
