"""DMD2: Improved Distribution Matching Distillation for Hunyuan3D-2.1.

Yin et al. (2024): Extends DMD1 by replacing the regression loss with a
GAN discriminator, and adds a replay buffer of teacher outputs.

Components:
  - Teacher (frozen)
  - Student (LoRA)
  - Fake score (separate LoRA adapter, same as DMD1)
  - Discriminator (MLP on DiT intermediate features, ~5M params)
  - Replay buffer (cached teacher outputs)
"""

import logging
from collections import deque

import torch
import torch.nn as nn

from dmd1_distillation import DMD1Distillation, FakeScoreAdapter

logger = logging.getLogger(__name__)


class FeatureDiscriminator(nn.Module):
    """Discriminator operating on DiT intermediate features.

    Extracts features from a specified transformer block via a forward hook,
    applies mean pooling, then a small MLP for real/fake classification.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: [B, N, D] -> [B, 1]"""
        # Mean pool over sequence dimension
        pooled = features.mean(dim=1)  # [B, D]
        return self.mlp(pooled)


class ReplayBuffer:
    """Fixed-size ring buffer of teacher outputs for GAN training."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add(self, x: torch.Tensor):
        """Add a batch of samples. Stores on CPU to save GPU memory."""
        for i in range(x.shape[0]):
            self.buffer.append(x[i].detach().cpu())

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample a random batch from the buffer."""
        indices = torch.randint(0, len(self.buffer), (batch_size,))
        samples = torch.stack([self.buffer[i] for i in indices])
        return samples.to(device)

    def __len__(self):
        return len(self.buffer)


class DMD2Distillation(DMD1Distillation):
    """DMD2: fake score + GAN discriminator (no regression loss)."""

    def __init__(self, config):
        # Skip DMD1.__init__'s pairs setup by calling BaseDistiller.__init__
        # then doing our own setup
        from base_distiller import BaseDistiller
        BaseDistiller.__init__(self, config)

        dmd2_cfg = config["training"]["methods"]["dmd2"]
        lora_cfg = config["training"]["model"]["lora"]

        self.lambda_gan = dmd2_cfg["lambda_gan"]
        self.d_update_ratio = dmd2_cfg["d_update_ratio"]

        # Fake score adapter (shared logic with DMD1)
        model = self.student.module if self.is_distributed else self.student
        self.fake_score_adapter = FakeScoreAdapter(model, lora_cfg)

        # Discriminator
        # DiT hidden dim: determined from the model config
        dit_model = self.teacher
        feature_dim = getattr(dit_model, "hidden_size", 1024)
        self.discriminator = FeatureDiscriminator(feature_dim).to(self.device)
        logger.info(
            "Discriminator: %.2fM params (feature_dim=%d)",
            sum(p.numel() for p in self.discriminator.parameters()) / 1e6,
            feature_dim,
        )

        # Feature extraction hook — use single_blocks (Hunyuan3DDiT) or blocks
        self._features = {}
        teacher_blocks = self._get_hook_blocks(dit_model)
        self._hook_block_idx = len(teacher_blocks) // 2
        self._hook_handle = teacher_blocks[self._hook_block_idx].register_forward_hook(
            self._feature_hook
        )
        logger.info(
            "Teacher feature hook: block %d/%d",
            self._hook_block_idx, len(teacher_blocks),
        )

        # Also register hook on student (unwrap peft to find blocks)
        student_dit = model.base_model.model if hasattr(model, "base_model") else model
        student_blocks = self._get_hook_blocks(student_dit)
        self._student_hook = student_blocks[self._hook_block_idx].register_forward_hook(
            self._student_feature_hook
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(dmd2_cfg["replay_buffer_size"])
        self.replay_warmup = dmd2_cfg["replay_warmup"]

        # No pairs needed (DMD2 removes regression loss)
        self.pairs_dir = None
        self._pairs_loader = None
        self._pairs_iter = None

        self._setup_optimizer_dmd2()

    def _method_name(self) -> str:
        return "dmd2"

    @staticmethod
    def _get_hook_blocks(model):
        """Find the right transformer block list for feature hooks.

        Hunyuan3DDiT has double_blocks + single_blocks (no 'blocks').
        We hook into single_blocks since they output plain tensors,
        while double_blocks output (img, txt) tuples.
        """
        if hasattr(model, "single_blocks") and len(model.single_blocks) > 0:
            return model.single_blocks
        if hasattr(model, "blocks"):
            return model.blocks
        raise RuntimeError(
            f"Cannot find transformer blocks for feature hook on {type(model).__name__}. "
            f"Expected 'single_blocks' or 'blocks' attribute."
        )

    def _feature_hook(self, module, input, output):
        """Hook to capture teacher features."""
        self._features["teacher"] = output

    def _student_feature_hook(self, module, input, output):
        """Hook to capture student features."""
        self._features["student"] = output

    def _setup_optimizer_dmd2(self):
        """Create three optimizers: student, fake_score, discriminator."""
        opt_cfg = self.config["training"]["optimizer"]
        dmd2_cfg = self.config["training"]["methods"]["dmd2"]
        model = self.student.module if self.is_distributed else self.student

        # Student + fake score use global lr
        self.fake_score_adapter.activate_student()
        student_params = [
            p for n, p in model.named_parameters()
            if "fake_score" not in n and p.requires_grad
        ]
        self.optimizer = torch.optim.AdamW(
            student_params, lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"],
        )

        fake_params = self.fake_score_adapter.fake_score_params()
        self.optimizer_fake = torch.optim.AdamW(
            fake_params, lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"],
        )

        # Discriminator: method-specific lr (higher per TTUR)
        self.optimizer_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=dmd2_cfg["lr_d"],
            weight_decay=opt_cfg["weight_decay"],
        )

    _custom_backward = True

    # ------------------------------------------------------------------
    # Replay buffer management
    # ------------------------------------------------------------------

    def _fill_replay_buffer(self, dataloader):
        """Warmup: fill replay buffer with real data samples for D training."""
        logger.info("Warming up replay buffer (%d samples) ...", self.replay_warmup)
        count = 0
        for batch in dataloader:
            if count >= self.replay_warmup:
                break
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            x_data = batch["latent"]
            self.replay_buffer.add(x_data)
            count += x_data.shape[0]
        logger.info("Replay buffer warmed up: %d samples", len(self.replay_buffer))

    # ------------------------------------------------------------------
    # Discriminator training
    # ------------------------------------------------------------------

    def _train_discriminator(self, batch: dict):
        """Update discriminator with non-saturating GAN loss (paper Eq. 4).

        Features extracted via mu_fake (fake score adapter), NOT teacher.
        Paper: "we add a classification branch on top of the bottleneck
        of the fake diffusion denoiser."
        """
        B = batch["latent"].shape[0]
        image_cond = batch["image_cond"]
        contexts = {"main": image_cond}

        # Real samples from replay buffer
        x_real = self.replay_buffer.sample(B, self.device)

        # Fake samples from student (detached)
        noise = torch.randn_like(batch["latent"])
        with torch.no_grad():
            self.fake_score_adapter.activate_student()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v_student = self.student_forward(
                    noise, torch.zeros(B, device=self.device), contexts,
                )
            x_fake = self.euler_step(noise, v_student, torch.ones(B, device=self.device))

        # Extract features via mu_fake at random timestep (paper Section 4.3)
        t_d = self.sample_t(B, self.device)
        self.fake_score_adapter.activate_fake_score()
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                self.student_forward(
                    self.diffuse(x_real, t_d, torch.randn_like(x_real)),
                    t_d, contexts,
                )
            feat_real = self._features.get("student")

            with torch.autocast("cuda", dtype=torch.bfloat16):
                self.student_forward(
                    self.diffuse(x_fake, t_d, torch.randn_like(x_fake)),
                    t_d, contexts,
                )
            feat_fake = self._features.get("student")
        self.fake_score_adapter.activate_student()

        if feat_real is None or feat_fake is None:
            logger.warning("Feature hook failed, skipping D update")
            return torch.tensor(0.0, device=self.device)

        # Non-saturating GAN loss (paper Eq. 4, not hinge)
        d_real = self.discriminator(feat_real.detach().float())
        d_fake = self.discriminator(feat_fake.detach().float())
        loss_d = (
            nn.functional.softplus(-d_real).mean()
            + nn.functional.softplus(d_fake).mean()
        )

        self.optimizer_d.zero_grad()
        loss_d.backward()
        self.optimizer_d.step()

        return loss_d.detach()

    # ------------------------------------------------------------------
    # Training step (overrides DMD1)
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, step: int) -> dict:
        """DMD2 step (Yin et al. 2024).

        TTUR: update fake_score + discriminator K times, then student once.
        Student loss = L_distill (KL via score diff) + lambda_gan * L_GAN.
        """
        batch_size = self.config["training"]["batch_size"]

        # --- Phase 1+2: TTUR — update fake score + D together K times ---
        # Paper: "update mu_fake 5 times per 1 generator update"
        loss_fake = torch.tensor(0.0, device=self.device)
        loss_d = torch.tensor(0.0, device=self.device)
        for _ in range(self.d_update_ratio):
            loss_fake = self._train_fake_score(batch)
            if len(self.replay_buffer) >= batch_size:
                loss_d = self._train_discriminator(batch)

        # --- Phase 3: Update student (KL + GAN) ---
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

        # KL distillation — same fixes as DMD1 (denoiser output, adaptive
        # weighting, MSE stop-grad trick)
        t_kl = self.sample_t(B, self.device)
        eps_kl = torch.randn_like(x_gen)
        x_gen_noised = self.diffuse(x_gen, t_kl, eps_kl)

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

            # Adaptive weighting (paper Eq. 8)
            weight_denom = (mu_teacher - x_gen).abs().mean(
                dim=list(range(1, x_gen.dim())), keepdim=True,
            ).clamp(min=1e-6)
            grad_direction = (mu_fake - mu_teacher) / weight_denom

        # Stop-gradient trick: loss = 0.5 * MSE(x, sg(x - grad))
        target = (x_gen - grad_direction).detach()
        loss_distill = 0.5 * nn.functional.mse_loss(x_gen, target)

        # GAN generator loss — features from mu_fake on noised x_gen
        # Paper: generator minimizes -log(D(feat)) (non-saturating)
        loss_gan = torch.tensor(0.0, device=self.device)
        if len(self.replay_buffer) >= batch_size:
            t_gan = self.sample_t(B, self.device)
            noise_gan = torch.randn_like(x_gen)
            x_gen_noised_gan = self.diffuse(x_gen, t_gan, noise_gan)

            # Run noised x_gen through mu_fake to extract features.
            # Gradients flow: D -> features -> mu_fake(x_gen_noised) -> x_gen -> student.
            # mu_fake params receive spurious gradients but are not in self.optimizer.
            self.fake_score_adapter.activate_fake_score()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                self.student_forward(x_gen_noised_gan, t_gan, contexts)
            feat_gen = self._features.get("student")
            self.fake_score_adapter.activate_student()

            if feat_gen is not None:
                d_gen = self.discriminator(feat_gen.float())
                loss_gan = nn.functional.softplus(-d_gen).mean()

        loss = loss_distill + self.lambda_gan * loss_gan

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.student.parameters() if p.requires_grad],
            self.config["training"]["gradient_clip"],
        )
        self.optimizer.step()

        # Add real data to replay buffer for discriminator
        self.replay_buffer.add(batch["latent"])

        return {
            "loss": loss.detach(),
            "loss_distill": loss_distill.detach(),
            "loss_gan": loss_gan.detach() if isinstance(loss_gan, torch.Tensor) else loss_gan,
            "loss_d": loss_d,
            "loss_fake_score": loss_fake,
        }

    # ------------------------------------------------------------------
    # Override train() to add replay warmup
    # ------------------------------------------------------------------

    def train(self, dataloader, total_steps: int):
        """Override to warm up replay buffer before training."""
        self._fill_replay_buffer(dataloader)
        super().train(dataloader, total_steps)
