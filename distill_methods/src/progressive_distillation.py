"""Progressive Distillation (PD) for Hunyuan3D-2.1.

Salimans & Ho (2022): Teacher 2-step -> Student 1-step, repeated across stages.
Stages: 50 -> 25 -> 12 -> 6 steps.

At each stage the teacher performs 2 Euler steps and the student learns to match
the result in a single step. After each stage, the student LoRA is merged into
the base model and a fresh LoRA is initialized for the next stage.
"""

import logging
import os

import torch

from base_distiller import BaseDistiller, _load_ema

logger = logging.getLogger(__name__)


class ProgressiveDistillation(BaseDistiller):
    """Progressive distillation: iteratively halve the number of steps."""

    def __init__(self, config):
        super().__init__(config)
        self._setup_optimizer()

        pd_cfg = config["training"]["methods"]["pd"]
        # Each entry is [teacher_steps, student_steps]
        self.stages = pd_cfg["stages"]  # e.g. [[50,25],[25,12],[12,6]]
        self.current_stage = 0
        self._update_schedule()

    def _method_name(self) -> str:
        return "pd"

    # ------------------------------------------------------------------
    # Stage management
    # ------------------------------------------------------------------

    def set_stage(self, stage_idx: int):
        """Set the current distillation stage."""
        self.current_stage = stage_idx
        self._update_schedule()
        logger.info(
            "PD stage %d: %d-step teacher -> %d-step student (h=%.4f)",
            stage_idx, self.teacher_steps, self.student_steps, self.h,
        )

    def _update_schedule(self):
        """Compute step size from current stage."""
        self.teacher_steps, self.student_steps = self.stages[self.current_stage]
        # Step size for the teacher in this stage
        # Teacher uses teacher_steps steps over [0,1], so dt = 1/teacher_steps
        self.h = 1.0 / self.teacher_steps

    def save_merged_model(self, stage_idx: int):
        """Merge LoRA into base and save the full state dict for next stage."""
        from peft import PeftModel

        model = self.student.module if self.is_distributed else self.student
        if isinstance(model, PeftModel):
            merged = model.merge_and_unload()
        else:
            merged = model
        save_dir = os.path.join(self.ckpt_dir, f"stage_{stage_idx}_merged")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "model.pt")
        torch.save(merged.state_dict(), save_path)
        logger.info("Saved merged model for stage %d: %s", stage_idx, save_path)

    def load_merged_base(self, path: str):
        """Load a merged model state dict into both teacher and student base.

        Used when resuming PD from a previous stage's merged checkpoint.
        This replaces the pretrained weights with the merged weights from
        the previous stage, so that the current stage trains on top of
        the correct base.
        """
        from peft import LoraConfig, get_peft_model

        logger.info("Loading merged base from %s ...", path)
        state_dict = torch.load(path, map_location=self.device)

        # Update teacher
        self.teacher.load_state_dict(state_dict)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # Rebuild student: unwrap LoRA, load merged weights, re-apply fresh LoRA
        model = self.student.module if self.is_distributed else self.student
        base = model.merge_and_unload()
        base.load_state_dict(state_dict)

        lora_cfg = self.config["training"]["model"]["lora"]
        peft_config = LoraConfig(
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg["alpha"],
            target_modules=lora_cfg["target_modules"],
        )
        self.student = get_peft_model(base, peft_config)
        self.student.to(self.device)

        # Reinitialize EMA and optimizer
        ema_cfg = self.config["training"]["model"]["ema"]
        self.ema = _load_ema(
            self.student, decay=ema_cfg["decay"],
            use_num_updates=ema_cfg.get("use_num_updates", True),
        )
        self.ema.to(self.device)
        self._setup_optimizer()

        logger.info("Loaded merged base into teacher + student (fresh LoRA).")

    def advance_stage(self):
        """Merge LoRA into base, reinitialize LoRA, advance to next stage.

        For in-process stage transitions (when all stages run in one process).
        """
        from peft import LoraConfig, get_peft_model

        model = self.student.module if self.is_distributed else self.student

        # Merge current LoRA into base weights
        model = model.merge_and_unload()
        logger.info("Merged LoRA into base model.")

        # Copy merged weights to teacher
        self.teacher.load_state_dict(model.state_dict())
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        logger.info("Updated teacher from merged student.")

        # Reinitialize fresh LoRA on student
        lora_cfg = self.config["training"]["model"]["lora"]
        peft_config = LoraConfig(
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg["alpha"],
            target_modules=lora_cfg["target_modules"],
        )
        self.student = get_peft_model(model, peft_config)
        self.student.to(self.device)

        # Reinitialize EMA and optimizer
        ema_cfg = self.config["training"]["model"]["ema"]
        self.ema = _load_ema(
            self.student, decay=ema_cfg["decay"],
            use_num_updates=ema_cfg.get("use_num_updates", True),
        )
        self.ema.to(self.device)
        self._setup_optimizer()

        # Advance stage index
        self.current_stage += 1
        self._update_schedule()
        logger.info(
            "Advanced to stage %d: %d -> %d steps",
            self.current_stage, self.teacher_steps, self.student_steps,
        )

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, step: int) -> dict:
        """PD training step: teacher 2-step -> student 1-step MSE.

        1. Sample random timestep t from the student schedule
        2. Create x_t = diffuse(x_data, t, noise)
        3. Teacher 2 Euler steps: x_t -> x_mid -> x_tgt
        4. Student 1 Euler step: x_t -> x_pred
        5. Loss = MSE(x_pred, x_tgt)
        """
        x_data = batch["latent"]  # [B, 4096, 64]
        image_cond = batch["image_cond"]  # [B, N, D]
        B = x_data.shape[0]
        contexts = {"main": image_cond}

        noise = torch.randn_like(x_data)
        h = self.h  # step size for teacher

        # Sample t from student schedule: t in {0, 2h, 4h, ...}
        # Student steps are half of teacher, so student dt = 2h
        num_student_positions = self.student_steps
        t_indices = torch.randint(0, num_student_positions, (B,), device=self.device)
        t = t_indices.float() * (2.0 * h)  # student step size = 2h
        t = t.clamp(0.0, 1.0 - 2.0 * h)  # ensure room for 2 teacher steps

        # Create noisy sample at t
        x_t = self.diffuse(x_data, t, noise)

        # --- Teacher 2-step ---
        # Stage 0: teacher is the pretrained model, use CFG.
        # Stage > 0: teacher is the merged student from the previous stage,
        # which was distilled to work without CFG, so use guidance_scale=1.0.
        teacher_cfg = self.guidance_scale if self.current_stage == 0 else 1.0
        v1 = self.teacher_forward(x_t, t, contexts, guidance_scale=teacher_cfg)
        x_mid = self.euler_step(x_t, v1, torch.tensor(h, device=self.device))
        t_mid = t + h
        v2 = self.teacher_forward(x_mid, t_mid, contexts, guidance_scale=teacher_cfg)
        x_tgt = self.euler_step(x_mid, v2, torch.tensor(h, device=self.device))

        # --- Student 1-step (2h) ---
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v_student = self.student_forward(x_t, t, contexts)

        # Loss in velocity space (Salimans & Ho, 2022).
        # The teacher target velocity is the average displacement over 2h:
        #   v_tgt = (x_tgt - x_t) / (2h)
        # Previous implementation used x-space MSE, which scales as (2h)^2
        # and causes implicit loss weighting differences across stages.
        v_tgt = (x_tgt.detach() - x_t) / (2.0 * h)
        loss = torch.nn.functional.mse_loss(v_student, v_tgt)

        return {"loss": loss}
