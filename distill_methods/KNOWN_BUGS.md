# Known Bugs — Distill Methods

## ~~DMD1: noise shape bug (`generate_dmd1_pairs.py:134`)~~ FIXED

~~`latent_shape[1:]` dropped the token dimension. Fixed: now uses `*latent_shape` → `[B, 4096, 64]`.~~

## ~~DMD2: GAN loss has no effect on student (`dmd2_distillation.py:294-305`)~~ FIXED

~~Two issues: `x_gen.detach()` cut gradient path, and features were extracted from frozen teacher hook. Fixed: generator loss now runs x_gen through mu_fake, uses `_features["student"]`, no detach.~~

## ~~DMD2: replay buffer uses static data (`dmd2_distillation.py:185-188`)~~ NOT A BUG

~~Originally reported as "paper uses fresh teacher-generated samples". Re-examined: the replay buffer feeds the discriminator's "real" distribution, so using real data (batch latents) is correct per the paper. Docstring/comment corrected.~~

## ~~Config: latent shape documentation~~ FIXED

~~Comment in `progressive_distillation.py:176` referenced `[B, 4096, 4]`. Fixed to `[B, 4096, 64]`.~~
