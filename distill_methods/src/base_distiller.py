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
        self.ema = _load_ema(
            self.student,
            decay=ema_cfg["decay"],
            use_num_updates=ema_cfg.get("use_num_updates", True),
        )
        self.ema.to(self.device)

        # --- DDP ---
        if self.is_distributed:
            self.student = DDP(
                self.student,
                device_ids=[self.rank],
                find_unused_parameters=False,
            )

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

    def _setup_optimizer(self):
        """Create optimizer(s). Called by subclass __init__."""
        raise NotImplementedError

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
    def sample_t(batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample timesteps uniformly from [0, 1]."""
        return torch.rand(batch_size, device=device)

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
        """Student velocity prediction (LoRA active)."""
        model = self.student.module if self.is_distributed else self.student
        return model(x_t, t, contexts=contexts)

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
        self.ema.store(model.named_parameters())
        self.ema.copy_to(model)
        try:
            yield
        finally:
            self.ema.restore(model.named_parameters())

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, dataloader, total_steps: int):
        """Main training loop. Subclasses implement training_step()."""
        model = self.student.module if self.is_distributed else self.student
        model.train()

        step = 0
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
        logger.info("Training complete (%d steps).", total_steps)

    # Flag for methods that handle backward/step themselves (DMD2)
    _custom_backward = False

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, step):
        """Save LoRA weights only."""
        save_dir = os.path.join(self.ckpt_dir, f"step_{step}")
        os.makedirs(save_dir, exist_ok=True)
        model = self.student.module if self.is_distributed else self.student
        model.save_pretrained(save_dir)

        # Save EMA separately
        ema_path = os.path.join(save_dir, "ema.pt")
        torch.save(self.ema.state_dict(), ema_path)
        logger.info("Saved checkpoint: %s", save_dir)

    def load_checkpoint(self, path: str):
        """Load LoRA weights from checkpoint.

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
        if self.is_distributed:
            self.student = DDP(
                self.student, device_ids=[self.rank], find_unused_parameters=False,
            )

        ema_path = os.path.join(path, "ema.pt")
        if os.path.exists(ema_path):
            self.ema.load_state_dict(torch.load(ema_path, map_location=self.device))
            logger.info("Loaded EMA from %s", ema_path)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @staticmethod
    def _log(step: int, loss_dict: dict):
        parts = [f"step={step}"]
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            parts.append(f"{k}={v:.6f}")
        logger.info("  ".join(parts))
