# Forward-looking priorities — 2026-04-21

Next-step items organized by axis. See [STATE.md](STATE.md) for the current
picture and [AXIS_INDEX.md](AXIS_INDEX.md) for code/data pointers.

Priority heuristic:
- **✶✶✶** = high priority, low cost
- **✶✶** = high value, medium cost
- **✶**  = high value, high cost (paper-scope-defining)
- **◦**  = nice-to-have / deferred

---

## Axis A — Teacher trajectory next steps

### A1 (H1) — Trellis2 / SLAT trajectory analysis ✶

**Goal**: determine if the narrow-norm-distribution + direction-preservation
claims are VecSet-specific or general to 3D LDMs.

**Approach**: run intermediate decode on TRELLIS 2's sparse voxel pipeline;
extract SLAT latents at matched trajectory positions; compute the analogue
of `analyze_phase3_drift.py` + `analyze_teacher_reference.py`.

**Decision value**: paper-scope-defining.
- VecSet tightness + SLAT spread → **VecSet-specific norm manifold**
- Both tight → **3D LDM universal norm structure**
- SLAT tight + different statistics → comparative study

**Cost**: 3–5 days (non-trivial because SLAT is a 2-stage cascade; need to
decide what counts as the "analogous" latent).

### A2 — Objaverse-scale diversity check ✶✶

**Goal**: does the narrow norm hold on a broader category distribution?

**Approach**: sample ~100 Objaverse objects; run teacher + compute latent
norms; compare against N=105 Toys4k reference.

**Cost**: 1–2 days. Depends on Objaverse access and rendering pipeline.

### A3 — Per-category N > 1 sampling ✶✶

**Goal**: distinguish within-category sample variance from between-category
shape dependence. Currently N=1 per category blocks this.

**Approach**: add 5–10 samples per category for a subset (e.g., 10 categories
× 5 samples = 50 extra inferences); re-aggregate norm stats.

**Cost**: ~1 hour compute + analysis. Cheap, but not paper-scope-critical.

### A4 — Decoder threshold mapping ◦

**Goal**: pinpoint the norm boundary at which teacher VAE transitions from
"uniform -1 SDF" to "valid isosurface". This operationalizes the
norm-based on-manifold-ness story at the decoder level.

**Approach**: synthetic latent perturbation (scale teacher step-50 latent by
varying α, decode, record first α that produces non-saturated output).

**Cost**: half a day. Elegant but not on critical path for paper claims.

---

## Axis B — Distillation comparison next steps

### B1 (H4) — Offline routing validation ✶✶✶

**Goal**: validate the proposed DMD1 → CD fallback using latent norm
z-score, using only existing data (no new GPU compute).

**Approach**: in `e1_postprocess.py`, load `teacher_ref_n105/stats.json`;
for each DMD1 sample, compute z-score; simulate "if z < −3.7, use CD
4-step result (from cross-family run), else use DMD1 1-step result";
compute expected mixed-inference CD and compare against pure DMD1 and
pure CD.

**Predicted outcome**: mixed inference recovers most of CD's success on
failure categories while keeping DMD1's cheap path on success categories.

**Cost**: < 2 hours (pure CSV manipulation; no GPU).

### B2 — Norm axis extension to CD / DMD2 / SiD ✶✶✶

**Goal**: central figure goes from "DMD1 only" to "all 4 student methods
on one axis with teacher reference band".

**Approach**: invoke `diagnose_dmd1_manifold.py` with `--model-name
cd_4step`, `dmd2_1step`, `sid_1step`. Each is ~17 minutes on 1 GPU.
Run in parallel on GPUs 2/3.

**Expected pattern**:
- CD: norm mean near teacher (trajectory-following).
- DMD2: wider spread, possibly some Mode A.
- SiD: all Mode A (explains 0/105 CD success).

**Cost**: ~1 hour (3 × 17 min sequential on 1 GPU, or ~20 min on 3 GPUs).

### B3 — tree_023 outlier visualization ✶✶

**Goal**: identify whether tree_023 is a norm-explained failure or a
separate topology failure.

**Approach**: render the DMD1-produced mesh for tree_023 and compare
visually to ground-truth and teacher's own output. Cheap.

**Cost**: 30 minutes.

### B4 (H2) — AYS schedule CD re-training ✶

**Goal**: test whether non-uniform inference schedule concentrated on the
norm-evolution "elbow" (t ≈ 0.4–0.7) beats the uniform [0.25, 0.5,
0.75, 1.0] that CD currently uses.

**Approach**: re-train CD with sampling distribution skewed toward t
= 0.4–0.7 (e.g., t_max = 0.7 boundary); compare CD vs unchanged CD on
the 105 test samples.

**Cost**: ~1 day (15K training steps + evaluation). Needs 1 GPU.

---

## Cross-axis actions

### X1 — Report refactor: split manifold-diagnosis by axis ✶✶

**Goal**: reorganize `2026-04-21_manifold-diagnosis.html` into two
axis-specific reports going forward. Historical mixed report stays.

**Outcome**:
- New `2026-04-22_teacher-norm-universality.html` — Axis A content (E2
  + E3 + teacher_ref_n105). Core claim: "narrow norm distribution,
  direction preservation, category-spanning tightness".
- New `2026-04-22_distillation-predictor.html` — Axis B content (E1 +
  H4 validation when ready). Core claim: "latent norm z-score predicts
  distillation CD; routing is viable".

**Cost**: 4–6 hours of writing. Best done *after* B1/B2 are in so new
data can be included.

### X2 — Paper scope decision

**Goal**: decide whether to pursue one paper or two.

**Decision triggers**:
- If H1 (A1) says "VecSet-specific": strong case for two papers
  (universality paper + distillation paper).
- If H1 says "universal": one unified paper on 3D LDM norm structure.
- If H1 is partial: one comparative paper with both axes as sections.

**Timing**: **after A1 H1 + B2 norm-axis extension**.
Do *not* write introduction / outline before these are settled.

---

## Suggested execution order

Given current GPU/time budget and decision dependencies:

1. **B1** (no GPU, ~2h) — confirms H4 works, paper implications
2. **B2** (~1h on 3 GPUs) — extends central figure
3. **B3** (30min) — one-shot sanity on outlier
4. **A3** (~1h) — cheap, tightens A claims before H1
5. **A1** (3-5 days) — biggest decision
6. **X1** (split reports) — after B1/B2/A1 data is in
7. **X2** (paper scope) — after A1
8. **A2 / B4** — later, depend on scope decisions

Steps 1–4 are cheap and can run in parallel in under a day, producing a
complete central-figure upgrade and closing most open questions in STATE.md.
