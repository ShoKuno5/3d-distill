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
import os
from collections import deque
from contextlib import nullcontext

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
    """Fixed-size ring buffer storing (latent, image_cond) pairs for GAN training."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add(self, x: torch.Tensor, cond: torch.Tensor):
        """Add a batch of (latent, conditioning) pairs. Stores on CPU."""
        for i in range(x.shape[0]):
            self.buffer.append((x[i].detach().cpu(), cond[i].detach().cpu()))

    def sample(self, batch_size: int, device: torch.device):
        """Sample a random batch from the buffer. Returns (latents, conds)."""
        indices = torch.randint(0, len(self.buffer), (batch_size,))
        xs = torch.stack([self.buffer[i][0] for i in indices]).to(device)
        conds = torch.stack([self.buffer[i][1] for i in indices]).to(device)
        return xs, conds

    def __len__(self):
        return len(self.buffer)


class DMD2Distillation(DMD1Distillation):
    """DMD2: fake score + GAN discriminator (no regression loss)."""

    def __init__(self, config):
        # Skip DMD1.__init__'s pairs setup by calling BaseDistiller.__init__
        # then doing our own setup. BaseDistiller.__init__ triggers
        # _add_extra_adapters() (inherited from DMD1), which creates
        # fake_score_adapter before the DDP wrap.
        from base_distiller import BaseDistiller
        BaseDistiller.__init__(self, config)

        dmd2_cfg = config["training"]["methods"]["dmd2"]

        self.lambda_gan = dmd2_cfg["lambda_gan"]
        # Inherited _student_generate (from DMD1) reads self.num_inference_steps.
        # DMD2 skips DMD1.__init__, so set it here too. Default 1 (single-step DMD2).
        self.num_inference_steps = dmd2_cfg.get("num_inference_steps", 1)
        self.d_update_ratio = dmd2_cfg["d_update_ratio"]

        # fake_score_adapter was created by inherited _add_extra_adapters()
        # during BaseDistiller.__init__, before DDP wrap.

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

        # Also register the hook on the student. Factored into a method so
        # load_checkpoint can re-bind it after the student module is rebuilt.
        self._register_student_hook()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(dmd2_cfg["replay_buffer_size"])
        self.replay_warmup = dmd2_cfg["replay_warmup"]

        # Optional regression loss (DMD1-style pairs). Enabled via config.
        self.lambda_reg = dmd2_cfg.get("lambda_reg", 0.0)
        if self.lambda_reg > 0:
            dmd1_cfg = config["training"]["methods"]["dmd1"]
            self.pairs_dir = dmd1_cfg["pairs_dir"]
            logger.info("DMD2+regression: lambda_reg=%.3f, pairs=%s",
                        self.lambda_reg, self.pairs_dir)
        else:
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
        """Warmup: fill replay buffer with (latent, image_cond) pairs."""
        logger.info("Warming up replay buffer (%d samples) ...", self.replay_warmup)
        count = 0
        for batch in dataloader:
            if count >= self.replay_warmup:
                break
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            self.replay_buffer.add(batch["latent"], batch["image_cond"])
            count += batch["latent"].shape[0]
        logger.info("Replay buffer warmed up: %d samples", len(self.replay_buffer))

    # ------------------------------------------------------------------
    # Discriminator training
    # ------------------------------------------------------------------

    def _train_discriminator(self, batch: dict):
        """Update discriminator + mu_fake with non-saturating GAN loss.

        Per paper Section 4.6: mu_fake is trained with BOTH DSM (in
        _train_fake_score) AND the GAN classification loss (here).
        Features extracted via mu_fake WITH gradients so D loss also
        improves mu_fake's feature representations.

        Bug fixes vs original implementation:
        - Real samples now use their OWN conditioning from the replay buffer
          (previously used the current batch's conditioning — shortcut learning).
        - mu_fake receives gradients from D loss (previously blocked by no_grad).
        """
        B = batch["latent"].shape[0]
        image_cond = batch["image_cond"]
        contexts_fake = {"main": image_cond}

        # Real samples + their conditioning from replay buffer (Bug 1 fix)
        x_real, cond_real = self.replay_buffer.sample(B, self.device)
        contexts_real = {"main": cond_real}

        # Fake samples from student (detached — no grad to student here)
        noise = torch.randn_like(batch["latent"])
        with torch.no_grad():
            self.fake_score_adapter.activate_student()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v_student = self.student_forward(
                    noise, torch.zeros(B, device=self.device), contexts_fake,
                )
            x_fake = self.euler_step(noise, v_student, torch.ones(B, device=self.device))

        # Extract features via mu_fake WITH gradients (Bug 2 fix).
        # Gradients flow through mu_fake so D loss also trains mu_fake's
        # feature representations, matching the paper's integrated design.
        t_d = self.sample_t(B, self.device)
        self.fake_score_adapter.activate_fake_score()

        self.optimizer_fake.zero_grad()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            self.student_forward(
                self.diffuse(x_real, t_d, torch.randn_like(x_real)),
                t_d, contexts_real,  # Bug 1 fix: correct conditioning
            )
        feat_real = self._features.get("student")

        with torch.autocast("cuda", dtype=torch.bfloat16):
            self.student_forward(
                self.diffuse(x_fake.detach(), t_d, torch.randn_like(x_fake)),
                t_d, contexts_fake,
            )
        feat_fake = self._features.get("student")

        self.fake_score_adapter.activate_student()

        if feat_real is None or feat_fake is None:
            logger.warning("Feature hook failed, skipping D update")
            return torch.tensor(0.0, device=self.device)

        # Non-saturating GAN loss (paper Eq. 4)
        d_real = self.discriminator(feat_real.float())
        d_fake = self.discriminator(feat_fake.float())
        loss_d = (
            nn.functional.softplus(-d_real).mean()
            + nn.functional.softplus(d_fake).mean()
        )

        self.optimizer_d.zero_grad()
        loss_d.backward()
        # Caller wraps this in self.student.no_sync(), so fake_score grads
        # are not auto-reduced by DDP. Discriminator is not DDP-wrapped at
        # all, so its grads are always rank-local and need manual sync too.
        self._sync_grads(self.discriminator.parameters())
        self._sync_grads(self.fake_score_adapter.fake_score_params())
        self.optimizer_d.step()

        # Bug 2 fix: also step mu_fake with GAN classification gradients
        torch.nn.utils.clip_grad_norm_(
            self.fake_score_adapter.fake_score_params(),
            self.config["training"]["gradient_clip"],
        )
        self.optimizer_fake.step()

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
        # NOTE: same batch conditioning reused for all K iterations; fresh noise
        # is generated inside each sub-step. Ideally each iteration would use
        # a fresh batch, but this requires a secondary dataloader iterator.
        #
        # DDP note: each sub-iter does forward+backward through the DDP-wrapped
        # student with the fake_score adapter active. With find_unused_parameters
        # =True, the reducer enters "reduction in progress" after backward and
        # the next forward in the loop trips "Expected to have finished
        # reduction in the prior iteration". Wrap the whole loop in no_sync()
        # so the reducer never activates; sub-iter grads are accumulated
        # locally and synced manually via _sync_grads inside _train_fake_score
        # and _train_discriminator. Phase 3 below ALSO uses no_sync + manual
        # sync because the GAN feature extraction there swaps to fake_score
        # mid-step, which provokes a separate DDP edge case (see Phase 3).
        no_sync = self.student.no_sync if self.is_distributed else nullcontext
        loss_fake = torch.tensor(0.0, device=self.device)
        loss_d = torch.tensor(0.0, device=self.device)
        for _ in range(self.d_update_ratio):
            with no_sync():
                loss_fake = self._train_fake_score(batch)
                if len(self.replay_buffer) >= batch_size:
                    loss_d = self._train_discriminator(batch)

        # --- Phase 3: Update student (KL + GAN) ---
        # Wrap the whole phase in no_sync. DDP arms the reducer inside
        # _post_forward (during forward, not backward), so to bypass
        # DDP's auto reduction the FORWARDS must run inside no_sync —
        # wrapping only loss.backward() is too late. Phase 3 also has a
        # mid-phase adapter swap (student -> fake_score for GAN feature
        # extraction at line 407, then back), which together with
        # find_unused_parameters=True provokes "Encountered gradient
        # which is undefined, but still allreduced by DDP reducer".
        # Student LoRA grads are all-reduced manually after backward.
        # Manual __enter__/__exit__ avoids re-indenting ~80 lines of
        # phase logic. On an unhandled exception in Phase 3 the process
        # dies anyway, so we accept that no_sync state may leak in that
        # rare path.
        phase3_ctx = no_sync()
        phase3_ctx.__enter__()
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

        # Optional regression loss (reuse DMD1's pairs infrastructure)
        loss_regress = torch.tensor(0.0, device=self.device)
        if self.lambda_reg > 0 and self.pairs_dir is not None:
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

        loss = loss_distill + self.lambda_gan * loss_gan + self.lambda_reg * loss_regress

        self.optimizer.zero_grad()
        # Clear fake_score grads to prevent stale GAN gradients from
        # leaking into the next TTUR fake_score update.
        self.optimizer_fake.zero_grad()
        student_params = self.optimizer.param_groups[0]["params"]
        loss.backward()
        # Close the Phase 3 no_sync that was opened before the first forward.
        phase3_ctx.__exit__(None, None, None)
        self._sync_grads(student_params)
        torch.nn.utils.clip_grad_norm_(
            student_params,
            self.config["training"]["gradient_clip"],
        )
        self.optimizer.step()

        # Add real data to replay buffer for discriminator
        self.replay_buffer.add(batch["latent"], batch["image_cond"])

        return {
            "loss": loss.detach(),
            "loss_distill": loss_distill.detach(),
            "loss_gan": loss_gan.detach() if isinstance(loss_gan, torch.Tensor) else loss_gan,
            "loss_regress": loss_regress.detach() if isinstance(loss_regress, torch.Tensor) else loss_regress,
            "loss_d": loss_d,
            "loss_fake_score": loss_fake,
        }

    # ------------------------------------------------------------------
    # Override train() to add replay warmup
    # ------------------------------------------------------------------

    def train(self, dataloader, total_steps: int, resume_step: int = 0):
        """Override to warm up replay buffer before training."""
        self._fill_replay_buffer(dataloader)
        super().train(dataloader, total_steps, resume_step=resume_step)

    # ------------------------------------------------------------------
    # Checkpointing (extends base to save discriminator + extra optimizers)
    # ------------------------------------------------------------------

    def save_checkpoint(self, step):
        """Save base checkpoint + discriminator weights and all optimizer states."""
        super().save_checkpoint(step)

        save_dir = os.path.join(self.ckpt_dir, f"step_{step}")

        # Save discriminator weights
        disc_path = os.path.join(save_dir, "discriminator.pt")
        torch.save(self.discriminator.state_dict(), disc_path)
        logger.info("Saved discriminator weights: %s", disc_path)

        # Append extra optimizer states to existing train_state.pt
        train_state_path = os.path.join(save_dir, "train_state.pt")
        train_state = torch.load(train_state_path, map_location="cpu")
        train_state["optimizer_d"] = self.optimizer_d.state_dict()
        train_state["optimizer_fake"] = self.optimizer_fake.state_dict()
        torch.save(train_state, train_state_path)
        logger.info("Saved optimizer_d and optimizer_fake states")

    def _register_student_hook(self):
        """(Re)register the student feature hook on the current self.student.

        load_checkpoint rebuilds self.student via PeftModel.from_pretrained, so
        the hook bound in __init__ points at the discarded module. Without
        re-binding, self._features['student'] never populates and the
        discriminator/GAN silently no-ops ("Feature hook failed, skipping D
        update").
        """
        if getattr(self, "_student_hook", None) is not None:
            self._student_hook.remove()
        model = self.student.module if self.is_distributed else self.student
        student_dit = model.base_model.model if hasattr(model, "base_model") else model
        student_blocks = self._get_hook_blocks(student_dit)
        self._student_hook = student_blocks[self._hook_block_idx].register_forward_hook(
            self._student_feature_hook
        )

    def load_checkpoint(self, path: str) -> int:
        """Load base checkpoint + discriminator weights and extra optimizer states."""
        resume_step = super().load_checkpoint(path)

        # super() rebuilt self.student (new module) -> re-bind the student
        # feature hook so the discriminator/GAN keeps receiving features.
        self._register_student_hook()

        # Restore discriminator weights (backward compat: skip if missing)
        disc_path = os.path.join(path, "discriminator.pt")
        if os.path.exists(disc_path):
            self.discriminator.load_state_dict(
                torch.load(disc_path, map_location=self.device)
            )
            logger.info("Loaded discriminator weights from %s", disc_path)
        else:
            logger.warning("No discriminator.pt found in %s, using fresh weights", path)

        # Restore extra optimizer states from train_state.pt
        train_state_path = os.path.join(path, "train_state.pt")
        if os.path.exists(train_state_path):
            train_state = torch.load(train_state_path, map_location=self.device)
            if "optimizer_d" in train_state:
                self.optimizer_d.load_state_dict(train_state["optimizer_d"])
                logger.info("Restored optimizer_d state")
            if "optimizer_fake" in train_state:
                self.optimizer_fake.load_state_dict(train_state["optimizer_fake"])
                logger.info("Restored optimizer_fake state")

        return resume_step
