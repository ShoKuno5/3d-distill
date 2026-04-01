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
        # HunyuanDiT uses hidden_size (typically 1536 for the 3.3B model)
        dit_model = self.teacher
        if hasattr(dit_model, "hidden_size"):
            feature_dim = dit_model.hidden_size
        else:
            # Fallback: inspect first block output
            feature_dim = 1536
        self.discriminator = FeatureDiscriminator(feature_dim).to(self.device)
        logger.info(
            "Discriminator: %.2fM params (feature_dim=%d)",
            sum(p.numel() for p in self.discriminator.parameters()) / 1e6,
            feature_dim,
        )

        # Feature extraction hook
        self._features = {}
        self._hook_block_idx = len(dit_model.blocks) // 2  # middle block
        self._hook_handle = dit_model.blocks[self._hook_block_idx].register_forward_hook(
            self._feature_hook
        )

        # Also register hook on student
        student_dit = model.base_model.model if hasattr(model, "base_model") else model
        if hasattr(student_dit, "blocks"):
            self._student_hook = student_dit.blocks[self._hook_block_idx].register_forward_hook(
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

    def _feature_hook(self, module, input, output):
        """Hook to capture teacher features."""
        self._features["teacher"] = output

    def _student_feature_hook(self, module, input, output):
        """Hook to capture student features."""
        self._features["student"] = output

    def _setup_optimizer_dmd2(self):
        """Create three optimizers: student, fake_score, discriminator."""
        dmd2_cfg = self.config["training"]["methods"]["dmd2"]
        model = self.student.module if self.is_distributed else self.student

        # Student optimizer
        self.fake_score_adapter.activate_student()
        student_params = [
            p for n, p in model.named_parameters()
            if "fake_score" not in n and p.requires_grad
        ]
        self.optimizer = torch.optim.AdamW(
            student_params, lr=dmd2_cfg["lr_g"], weight_decay=0.01,
        )

        # Fake score optimizer
        fake_params = self.fake_score_adapter.fake_score_params()
        self.optimizer_fake = torch.optim.AdamW(
            fake_params, lr=dmd2_cfg["lr_g"], weight_decay=0.01,
        )

        # Discriminator optimizer (higher LR per TTUR)
        self.optimizer_d = torch.optim.AdamW(
            self.discriminator.parameters(), lr=dmd2_cfg["lr_d"], weight_decay=0.01,
        )

    _custom_backward = True

    # ------------------------------------------------------------------
    # Replay buffer management
    # ------------------------------------------------------------------

    def _fill_replay_buffer(self, dataloader):
        """Warmup: fill replay buffer with teacher outputs."""
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
        """Update discriminator with hinge loss.

        real = teacher samples from replay buffer
        fake = student 1-step outputs (detached)
        """
        B = batch["latent"].shape[0]
        image_cond = batch["image_cond"]
        contexts = {"main": image_cond}

        # Real samples from replay buffer
        x_real = self.replay_buffer.sample(B, self.device)

        # Fake samples from student
        noise = torch.randn_like(batch["latent"])
        with torch.no_grad():
            self.fake_score_adapter.activate_student()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v_student = self.student_forward(
                    noise, torch.zeros(B, device=self.device), contexts,
                )
            x_fake = self.euler_step(noise, v_student, torch.ones(B, device=self.device))

        # Get features via teacher forward (triggers hook)
        t_probe = torch.ones(B, device=self.device) * 0.5
        with torch.no_grad():
            # Run teacher on real to get features
            self.teacher(self.diffuse(x_real, t_probe, torch.randn_like(x_real)),
                         t_probe, contexts=contexts)
        feat_real = self._features.get("teacher")

        with torch.no_grad():
            self.teacher(self.diffuse(x_fake, t_probe, torch.randn_like(x_fake)),
                         t_probe, contexts=contexts)
        feat_fake = self._features.get("teacher")

        if feat_real is None or feat_fake is None:
            logger.warning("Feature hook failed, skipping D update")
            return torch.tensor(0.0, device=self.device)

        # Hinge loss
        d_real = self.discriminator(feat_real.detach().float())
        d_fake = self.discriminator(feat_fake.detach().float())
        loss_d = (
            torch.nn.functional.relu(1.0 - d_real).mean()
            + torch.nn.functional.relu(1.0 + d_fake).mean()
        )

        self.optimizer_d.zero_grad()
        loss_d.backward()
        self.optimizer_d.step()

        return loss_d.detach()

    # ------------------------------------------------------------------
    # Training step (overrides DMD1)
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, step: int) -> dict:
        """DMD2 step: fake_score -> discriminator x d_update_ratio -> student (KL + GAN)."""

        # --- Phase 1: Update fake score (same as DMD1) ---
        loss_fake = self._train_fake_score(batch)

        # --- Phase 2: Update discriminator (TTUR: multiple updates) ---
        loss_d = torch.tensor(0.0, device=self.device)
        if len(self.replay_buffer) >= self.config["training"]["batch_size"]:
            for _ in range(self.d_update_ratio):
                loss_d = self._train_discriminator(batch)

        # --- Phase 3: Update student (KL + GAN, no regression) ---
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

        # KL distillation (same as DMD1)
        t_kl = self.sample_t(B, self.device)
        eps_kl = torch.randn_like(x_gen)
        x_gen_noised = self.diffuse(x_gen, t_kl, eps_kl)

        with torch.no_grad():
            self.fake_score_adapter.activate_fake_score()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v_fake_at_gen = self.student_forward(x_gen_noised, t_kl, contexts)
            self.fake_score_adapter.activate_student()

        v_teacher_at_gen = self.teacher_forward(x_gen_noised, t_kl, contexts)
        grad_kl = (v_fake_at_gen - v_teacher_at_gen).detach()
        loss_distill = (grad_kl * x_gen).sum() / B

        # GAN generator loss
        loss_gan = torch.tensor(0.0, device=self.device)
        if len(self.replay_buffer) >= self.config["training"]["batch_size"]:
            # Get features for generated samples
            t_gan = torch.ones(B, device=self.device) * 0.5
            with torch.no_grad():
                self.teacher(self.diffuse(x_gen.detach(), t_gan, torch.randn_like(x_gen)),
                             t_gan, contexts=contexts)
            feat_gen = self._features.get("teacher")
            if feat_gen is not None:
                d_gen = self.discriminator(feat_gen.float())
                loss_gan = -d_gen.mean()

        loss = loss_distill + self.lambda_gan * loss_gan

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.student.parameters() if p.requires_grad],
            self.config["training"]["gradient_clip"],
        )
        self.optimizer.step()

        # Add current data to replay buffer periodically
        if step % 10 == 0:
            self.replay_buffer.add(x_data)

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
