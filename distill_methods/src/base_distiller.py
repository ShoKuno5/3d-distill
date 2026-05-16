"""Base distiller for Hunyuan3D-2.1 flow matching DiT.

All four distillation methods (PD, CD, DMD1, DMD2) inherit from this class.
Provides: model loading, LoRA setup, EMA, flow matching primitives,
teacher CFG forward, training loop skeleton, checkpointing.

Uses pure PyTorch (no PL) for flexibility with multi-optimizer setups (DMD2).
"""

import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import wandb
except ImportError:
    wandb = None

logger = logging.getLogger(__name__)

# Ensure model code is importable
_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "hunyuan3d21" / "hy3dshape"
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))


def _load_pipeline(pretrained_path: str):
    """Load the full Hunyuan3D-2.1 pipeline and return (dit, pipeline).

    Pipeline attributes:
      - pipeline.model: HunyuanDiT (denoiser)
      - pipeline.vae: ShapeVAE (encoder/decoder)
      - pipeline.conditioner: SingleImageEncoder
      - pipeline.scheduler: FlowMatchEulerDiscreteScheduler
    """
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        pretrained_path,
        use_safetensors=False,
    )
    return pipeline.model, pipeline


def _enable_gradient_checkpointing(peft_model):
    """Monkey-patch the inner denoiser's forward to wrap its transformer
    block loop in torch.utils.checkpoint.

    Supports both denoiser architectures shipped with Hunyuan3D-2.1:
      - HunYuanDiTPlain (Hunyuan3D-2.1 production): single
        ``self.blocks`` ModuleList with U-Net-style skip connections.
      - Hunyuan3DDiT (Flux-style alt): split into ``double_blocks`` +
        ``single_blocks``.

    Activations between blocks are recomputed during backward instead of
    stored. Memory drops roughly 5-10x at the cost of ~25-35% extra
    wallclock per training step.

    Adapter-aware: captures the active peft adapter at forward time and
    restores it during checkpoint recomputation. This matters for DMD
    methods that switch adapters between forward and backward — without
    this, recomputation would use the *current* adapter and produce
    gradients for the wrong LoRA matrices.

    Vendored model code at models/hunyuan3d21/hy3dshape/... is not in
    git (.gitignored), so this is a monkey-patch from the distillation
    side rather than a source edit.
    """
    import types
    from torch.utils.checkpoint import checkpoint

    inner = getattr(peft_model, "base_model", peft_model)
    inner = getattr(inner, "model", inner)

    if getattr(inner, "_gc_enabled", False):
        logger.info("Gradient checkpointing already enabled on student DiT")
        return

    arch_name = type(inner).__name__
    if hasattr(inner, "blocks") and not hasattr(inner, "double_blocks"):
        _patch_hunyuan_dit_plain(inner, peft_model, checkpoint)
        inner._gc_enabled = True
        logger.info(
            "Gradient checkpointing enabled on %s (%d blocks, skip-connection)",
            arch_name,
            len(inner.blocks),
        )
    elif hasattr(inner, "double_blocks") and hasattr(inner, "single_blocks"):
        _patch_hunyuan_3d_dit_flux(inner, peft_model, checkpoint)
        inner._gc_enabled = True
        logger.info(
            "Gradient checkpointing enabled on %s (%d double + %d single blocks)",
            arch_name,
            len(inner.double_blocks),
            len(inner.single_blocks),
        )
    else:
        logger.warning(
            "Gradient checkpointing: unrecognized DiT architecture %s — "
            "expected attribute `blocks` or `double_blocks/single_blocks`. "
            "Skipping (will use full activation memory).",
            arch_name,
        )


def _make_adapter_restore_wrapper(peft_model):
    """Return (capture_adapter, restore_in_run) helpers shared by the
    arch-specific patches. Captures the active adapter at the moment of
    call and produces a context manager that restores it during
    checkpoint recomputation only if the current adapter differs.
    """
    captured = peft_model.active_adapter
    if isinstance(captured, list):
        captured = list(captured)

    class _AdapterRestore:
        def __enter__(self):
            self._current = peft_model.active_adapter
            self._need = self._current != captured
            if self._need:
                peft_model.set_adapter(captured)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self._need:
                peft_model.set_adapter(self._current)
            return False

    return _AdapterRestore


def _patch_hunyuan_dit_plain(inner, peft_model, checkpoint):
    """HunYuanDiTPlain has 24 ``self.blocks`` connected with U-Net-style
    skip connections (first half saves outputs, second half pops them).
    Wrap the whole loop in one checkpoint region; the skip_value_list is
    rebuilt internally during recomputation."""
    import types

    def gc_forward(self, x, t, contexts, **kwargs):
        cond = contexts["main"]
        t_emb = self.t_embedder(t, condition=kwargs.get("guidance_cond"))
        x = self.x_embedder(x)
        if self.use_pos_emb:
            pos_embed = self.pos_embed.to(x.dtype)
            x = x + pos_embed
        if self.use_attention_pooling:
            extra_vec = self.pooler(cond, None)
            c = t_emb + self.extra_embedder(extra_vec)
        else:
            c = t_emb
        if self.with_decoupled_ca:
            additional_cond = self.additional_cond_proj(contexts["additional"])
            cond = torch.cat([cond, additional_cond], dim=1)
        x = torch.cat([c, x], dim=1)

        use_ckpt = self.training and torch.is_grad_enabled()
        if use_ckpt:
            Restore = _make_adapter_restore_wrapper(peft_model)

            def run_blocks(x, c, cond):
                with Restore():
                    skip_value_list = []
                    for layer, block in enumerate(self.blocks):
                        skip_value = (
                            None
                            if layer <= self.depth // 2
                            else skip_value_list.pop()
                        )
                        x = block(x, c, cond, skip_value=skip_value)
                        if layer < self.depth // 2:
                            skip_value_list.append(x)
                    return x

            x = checkpoint(run_blocks, x, c, cond, use_reentrant=False)
        else:
            skip_value_list = []
            for layer, block in enumerate(self.blocks):
                skip_value = (
                    None
                    if layer <= self.depth // 2
                    else skip_value_list.pop()
                )
                x = block(x, c, cond, skip_value=skip_value)
                if layer < self.depth // 2:
                    skip_value_list.append(x)

        x = self.final_layer(x)
        return x

    inner.forward = types.MethodType(gc_forward, inner)


def _patch_hunyuan_3d_dit_flux(inner, peft_model, checkpoint):
    """Hunyuan3DDiT (Flux-style) has separate ``double_blocks`` and
    ``single_blocks`` lists. One checkpoint region per list."""
    import types
    from hy3dshape.models.denoisers.hunyuan3ddit import timestep_embedding

    def gc_forward(self, x, t, contexts, **kwargs):
        cond = contexts["main"]
        latent = self.latent_in(x)
        vec = self.time_in(
            timestep_embedding(t, 256, self.time_factor).to(dtype=latent.dtype)
        )
        if self.guidance_embed:
            guidance = kwargs.get("guidance", None)
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            vec = vec + self.guidance_in(
                timestep_embedding(guidance, 256, self.time_factor)
            )
        cond = self.cond_in(cond)
        pe = None

        use_ckpt = self.training and torch.is_grad_enabled()
        if use_ckpt:
            Restore = _make_adapter_restore_wrapper(peft_model)

            def run_double(latent, cond, vec):
                with Restore():
                    for block in self.double_blocks:
                        latent, cond = block(
                            img=latent, txt=cond, vec=vec, pe=None
                        )
                    return latent, cond

            def run_single(latent, vec):
                with Restore():
                    for block in self.single_blocks:
                        latent = block(latent, vec=vec, pe=None)
                    return latent

            latent, cond = checkpoint(
                run_double, latent, cond, vec, use_reentrant=False
            )
            latent = torch.cat((cond, latent), 1)
            latent = checkpoint(run_single, latent, vec, use_reentrant=False)
        else:
            for block in self.double_blocks:
                latent, cond = block(img=latent, txt=cond, vec=vec, pe=pe)
            latent = torch.cat((cond, latent), 1)
            for block in self.single_blocks:
                latent = block(latent, vec=vec, pe=pe)

        latent = latent[:, cond.shape[1]:, ...]
        latent = self.final_layer(latent, vec)
        return latent

    inner.forward = types.MethodType(gc_forward, inner)


def _load_ema(model: nn.Module, decay: float = 0.999, use_num_updates: bool = True):
    """Load LitEma from Hunyuan3D-2.1 codebase."""
    from hy3dshape.utils.ema import LitEma

    return LitEma(model, decay=decay, use_num_updates=use_num_updates)


class BaseDistiller:
    """Common base for all four distillation methods.

    Subclasses must implement ``training_step(batch, step) -> dict``.
    The returned dict must contain a ``'loss'`` key.
    """

    def __init__(self, config):
        self.config = config
        train_cfg = config["training"]
        model_cfg = train_cfg["model"]
        lora_cfg = model_cfg["lora"]
        ema_cfg = model_cfg["ema"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_distributed = dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.is_main = self.rank == 0

        self.guidance_scale = train_cfg["cfg"]["guidance_scale"]

        self.ckpt_dir = os.path.join(
            config["output_root"], "checkpoints", self._method_name()
        )
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # --- Load pipeline (teacher + VAE + conditioner) ---
        logger.info("Loading pipeline from %s ...", model_cfg["pretrained"])
        self.teacher, self._pipeline = _load_pipeline(model_cfg["pretrained"])
        self.teacher.to(self.device)
        # Convert fp16 weights to bf16 to match autocast dtype
        if train_cfg.get("bf16", False):
            self.teacher.to(torch.bfloat16)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # Keep references to VAE and cond model for data prep / inference
        self.vae = self._pipeline.vae
        self.cond_model = self._pipeline.conditioner
        self.z_scale_factor = model_cfg.get("z_scale_factor", 1.0)

        # --- Build student (deepcopy of teacher + LoRA) ---
        logger.info("Building student with LoRA (rank=%d) ...", lora_cfg["rank"])
        import copy
        self.student = copy.deepcopy(self.teacher)
        self.student.to(self.device)
        for p in self.student.parameters():
            p.requires_grad_(True)

        from peft import LoraConfig, get_peft_model

        peft_config = LoraConfig(
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg["alpha"],
            target_modules=lora_cfg["target_modules"],
        )
        self.student = get_peft_model(self.student, peft_config)
        trainable = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.student.parameters())
        logger.info(
            "Student: %d trainable / %d total params (%.2f%%)",
            trainable, total, 100 * trainable / total,
        )

        # --- EMA over trainable (LoRA) parameters ---
        # Note: EMA is created BEFORE _add_extra_adapters() on purpose.
        # Extra adapters (e.g. DMD1/DMD2/SiD's fake_score) are auxiliary
        # score networks, not the generator being distilled, and should
        # not be EMA'd. Keeping EMA initialization above the hook ensures
        # only the student (default) adapter's params are tracked.
        self.ema = _load_ema(
            self.student,
            decay=ema_cfg["decay"],
            use_num_updates=ema_cfg.get("use_num_updates", True),
        )
        self.ema.to(self.device)

        # Subclasses register extra PEFT adapters (e.g. fake_score) before
        # DDP wrap so their params are included in DDP's gradient buckets.
        self._add_extra_adapters()

        # --- Optional: gradient checkpointing on the inner DiT ---
        # Enabled when the subclass sets _use_gradient_checkpointing=True
        # (DMD1/DMD2 do, to make batch=4 fit on H100 96GB) or the config
        # flips training.gradient_checkpointing. Must happen after peft
        # adapters are added and before DDP wrap, so DDP records hooks
        # against the patched forward.
        gc_enabled = (
            getattr(self, "_use_gradient_checkpointing", False)
            or config.get("training", {}).get("gradient_checkpointing", False)
        )
        if gc_enabled:
            _enable_gradient_checkpointing(self.student)

        # --- DDP ---
        if self.is_distributed:
            # find_unused_parameters=True because adapter switching means
            # different subsets of params participate in each backward pass.
            ddp_find_unused = getattr(self, "_ddp_find_unused_parameters", False)
            self.student = DDP(
                self.student,
                device_ids=[self.rank],
                find_unused_parameters=ddp_find_unused,
            )

        # --- Wandb ---
        self._wandb_enabled = False
        wandb_cfg = config.get("wandb")
        if wandb_cfg and wandb is not None and self.is_main:
            self._wandb_enabled = True
            self._wandb_cfg = wandb_cfg

        # Subclass sets up optimizer(s) via _setup_optimizer()

    # ------------------------------------------------------------------
    # Abstract / hooks
    # ------------------------------------------------------------------

    def _method_name(self) -> str:
        """Return short name for checkpoint directory."""
        raise NotImplementedError

    def training_step(self, batch: dict, step: int) -> dict:
        """Compute loss for one step. Must return dict with 'loss' key."""
        raise NotImplementedError

    def _add_extra_adapters(self):
        """Hook called after EMA setup, before DDP wrap. Default no-op.

        Subclasses override this to register extra PEFT adapters on
        ``self.student`` (e.g. DMD1/DMD2/SiD's fake_score). Adapters added
        via ``peft.add_adapter()`` AFTER DDP wraps the module are silently
        excluded from DDP's gradient all-reduce and diverge per-rank, so
        this hook ensures their params live inside the DDP module when
        wrapping happens.
        """
        pass

    def _setup_optimizer(self):
        """Create optimizer from global training.optimizer config.

        Subclasses with multiple optimizers (DMD1, DMD2) override this.
        """
        opt_cfg = self.config["training"]["optimizer"]
        self.optimizer = torch.optim.AdamW(
            [p for p in self.student.parameters() if p.requires_grad],
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg["weight_decay"],
        )

    def _sync_grads(self, params):
        """All-reduce gradients across ranks with AVG.

        Used when DDP's automatic all-reduce is suppressed via
        ``student.no_sync()`` (e.g. DMD1/DMD2 TTUR sub-iterations that
        run multiple forward/backward pairs per training_step), and for
        non-DDP-wrapped modules trained alongside the student (DMD2's
        discriminator). No-op when not distributed.
        """
        if not self.is_distributed:
            return
        for p in params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

    # ------------------------------------------------------------------
    # Flow Matching primitives (ICPlan convention)
    #   x_t = t * x_data + (1-t) * noise
    #   v   = x_data - noise
    #   Euler: x_{t+dt} = x_t + dt * v
    # ------------------------------------------------------------------

    @staticmethod
    def diffuse(x_data: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Interpolate between noise (t=0) and data (t=1)."""
        t = t.view(-1, *([1] * (x_data.dim() - 1)))
        return t * x_data + (1 - t) * noise

    @staticmethod
    def get_velocity(x_data: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Ground truth velocity: v = x_data - noise."""
        return x_data - noise

    @staticmethod
    def euler_step(x_t: torch.Tensor, v: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Single Euler step: x_{t+dt} = x_t + dt * v."""
        if dt.dim() > 0:
            dt = dt.view(-1, *([1] * (x_t.dim() - 1)))
        return x_t + dt * v

    @staticmethod
    def sample_t(batch_size: int, device: torch.device, eps: float = 1e-3) -> torch.Tensor:
        """Sample timesteps uniformly from [eps, 1-eps]."""
        return torch.rand(batch_size, device=device) * (1.0 - 2 * eps) + eps

    # ------------------------------------------------------------------
    # Model forwards
    # ------------------------------------------------------------------

    def teacher_forward(
        self, x_t: torch.Tensor, t: torch.Tensor, contexts: dict,
        guidance_scale: float | None = None,
    ) -> torch.Tensor:
        """Teacher velocity prediction with optional CFG.

        Model signature: model(x, t, contexts=contexts) -> v
        """
        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            v_cond = self.teacher(x_t, t, contexts=contexts)
            if guidance_scale > 1.0:
                null_ctx = self._null_contexts(contexts)
                v_uncond = self.teacher(x_t, t, contexts=null_ctx)
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v = v_cond
        return v

    def student_forward(
        self, x_t: torch.Tensor, t: torch.Tensor, contexts: dict,
    ) -> torch.Tensor:
        """Student velocity prediction (LoRA active).

        Goes through ``self.student`` (not ``.module``) so that DDP's
        ``prepare_for_backward`` hook fires and gradient all-reduce is
        registered for the subsequent backward. Unwrapping DDP here
        causes silent per-rank divergence: grads are computed locally
        but never all-reduced.

        Under ``torch.no_grad()`` DDP's forward skips the reducer state
        update, so no-grad callers (e.g. fake_score queries in the
        student-update phase) incur no extra cost or bookkeeping.
        """
        return self.student(x_t, t, contexts=contexts)

    def _null_contexts(self, contexts: dict) -> dict:
        """Create null (zero) contexts for CFG unconditional branch."""
        return {k: torch.zeros_like(v) for k, v in contexts.items()}

    # ------------------------------------------------------------------
    # EMA helpers
    # ------------------------------------------------------------------

    @contextmanager
    def ema_scope(self):
        """Context manager: temporarily swap student weights with EMA."""
        model = self.student.module if self.is_distributed else self.student
        self.ema.store(model)
        self.ema.copy_to(model)
        try:
            yield
        finally:
            self.ema.restore(model)

    # ------------------------------------------------------------------
    # Wandb
    # ------------------------------------------------------------------

    def set_run_name(self, name: str):
        """Set wandb run name (call before train())."""
        self._run_name = name

    def _init_wandb(self):
        """Initialize wandb run (main process only)."""
        if not self._wandb_enabled:
            return
        cfg = self._wandb_cfg
        method = self._method_name()
        run_name = getattr(self, "_run_name", method)
        # Finish any lingering run before starting a new one
        if wandb.run is not None:
            wandb.finish()
        wandb.init(
            project=cfg["project"],
            entity=cfg.get("entity"),
            name=run_name,
            group=method,
            config=self.config,
        )
        logger.info("Wandb initialized: project=%s, run=%s", cfg["project"], run_name)

    def _finish_wandb(self):
        """Finish wandb run."""
        if self._wandb_enabled and wandb.run is not None:
            wandb.finish()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, dataloader, total_steps: int, resume_step: int = 0):
        """Main training loop. Subclasses implement training_step()."""
        model = self.student.module if self.is_distributed else self.student
        model.train()

        self._init_wandb()

        step = resume_step
        epoch = 0
        log_interval = self.config["training"]["log_interval"]
        save_interval = self.config["training"]["save_interval"]
        grad_clip = self.config["training"]["gradient_clip"]

        logger.info("Starting training for %d steps ...", total_steps)
        while step < total_steps:
            if self.is_distributed:
                dataloader.sampler.set_epoch(epoch)
            for batch in dataloader:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                loss_dict = self.training_step(batch, step)
                loss = loss_dict["loss"]

                # Backward + clip + step are handled here for single-optimizer
                # methods. Multi-optimizer methods (DMD2) override train().
                if not self._custom_backward:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.student.parameters() if p.requires_grad],
                        grad_clip,
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # EMA update
                self.ema(model if not self.is_distributed else self.student.module)

                step += 1
                if step % log_interval == 0 and self.is_main:
                    self._log(step, loss_dict)
                if step % save_interval == 0 and self.is_main:
                    self.save_checkpoint(step)
                if step >= total_steps:
                    break
            epoch += 1

        # Final save
        if self.is_main:
            self.save_checkpoint("final")
        self._finish_wandb()
        logger.info("Training complete (%d steps).", total_steps)

    # Flag for methods that handle backward/step themselves (DMD2)
    _custom_backward = False

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, step):
        """Save LoRA weights, optimizer state, EMA, and step counter."""
        save_dir = os.path.join(self.ckpt_dir, f"step_{step}")
        os.makedirs(save_dir, exist_ok=True)
        model = self.student.module if self.is_distributed else self.student
        model.save_pretrained(save_dir)

        # Save EMA
        ema_path = os.path.join(save_dir, "ema.pt")
        torch.save(self.ema.state_dict(), ema_path)

        # Save optimizer state and step counter for resume
        train_state = {"step": step}
        if hasattr(self, "optimizer") and self.optimizer is not None:
            train_state["optimizer"] = self.optimizer.state_dict()
        torch.save(train_state, os.path.join(save_dir, "train_state.pt"))

        logger.info("Saved checkpoint: %s (step=%s)", save_dir, step)

    def load_checkpoint(self, path: str) -> int:
        """Load LoRA weights, optimizer state, and EMA from checkpoint.

        Returns the step counter from the checkpoint (0 if not saved).

        When the current student is a PeftModel, extracting base_model.model
        leaves peft hooks/attributes that prevent from_pretrained from
        attaching a fresh adapter.  We deepcopy to sever that residual state.
        """
        import copy

        from peft import PeftModel

        model = self.student.module if self.is_distributed else self.student
        if isinstance(model, PeftModel):
            base = copy.deepcopy(model.base_model.model)
        else:
            base = model
        self.student = PeftModel.from_pretrained(base, path, is_trainable=True)
        # Re-register extra adapters (fake_score) on the rebuilt student
        # before DDP wrap, mirroring __init__'s ordering.
        self._add_extra_adapters()
        if self.is_distributed:
            ddp_find_unused = getattr(self, "_ddp_find_unused_parameters", False)
            self.student = DDP(
                self.student,
                device_ids=[self.rank],
                find_unused_parameters=ddp_find_unused,
            )

        # Rebuild optimizer with new model parameters
        self._setup_optimizer()

        # Restore optimizer state and step counter
        resume_step = 0
        train_state_path = os.path.join(path, "train_state.pt")
        if os.path.exists(train_state_path):
            train_state = torch.load(train_state_path, map_location=self.device)
            if "optimizer" in train_state and hasattr(self, "optimizer"):
                self.optimizer.load_state_dict(train_state["optimizer"])
            resume_step = train_state.get("step", 0)
            logger.info("Restored optimizer state, resume from step %d", resume_step)

        ema_path = os.path.join(path, "ema.pt")
        if os.path.exists(ema_path):
            self.ema.load_state_dict(torch.load(ema_path, map_location=self.device))
            logger.info("Loaded EMA from %s", ema_path)

        return resume_step

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, step: int, loss_dict: dict):
        parts = [f"step={step}"]
        wandb_metrics = {"step": step}
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            parts.append(f"{k}={v:.6f}")
            wandb_metrics[k] = v
        logger.info("  ".join(parts))
        if self._wandb_enabled and wandb.run is not None:
            wandb.log(wandb_metrics, step=step)
