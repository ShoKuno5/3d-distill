# Known Bugs — Distill Methods

## DMD1: noise shape bug (`generate_dmd1_pairs.py:134`)

```python
latent_shape = data["latent"].shape  # (4096, 64) — no batch dim
noise = torch.randn(B, *latent_shape[1:], ...)  # [B, 64] — WRONG
```

Fix: `torch.randn(B, *latent_shape, ...)` → `[B, 4096, 64]`

## DMD2: GAN loss has no effect on student (`dmd2_distillation.py:294-305`)

Two issues prevent GAN gradients from reaching the student:

1. `x_gen.detach()` at line 300 cuts the gradient path
2. Features are extracted from the **teacher** hook, not the student hook —
   even without detach, the teacher is frozen so gradients wouldn't flow

The student feature hook is registered (line 116-117) but never read.

Fix: for generator loss, run x_gen through the student (not teacher),
use `self._features["student"]`, and remove the `.detach()`.

## DMD2: replay buffer uses static data (`dmd2_distillation.py:185-188`)

`_fill_replay_buffer` and periodic additions (line 319) add `batch["latent"]`
(pre-computed static data). DMD2 paper uses fresh teacher-generated samples.
Low priority — acceptable approximation once training data comes from teacher ODE.

## Config: latent shape documentation

Comments and config reference `[B, 4096, 4]` but actual shape is `[B, 4096, 64]`.
