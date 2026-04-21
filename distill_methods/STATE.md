# Current State — 2026-04-21

Snapshot of what's **established**, **uncertain**, or **open** in the two
research threads running in this repo: **trajectory analysis** (what the
VecSet teacher does) and **distillation comparison** (how student methods
fare vs teacher).

---

## Axis A — Teacher trajectory & stepwise decoding

### Established

1. **Direction preserved throughout ODE trajectory** (per-token cos_mean > 0.995
   for every consecutive step pair, all 10 samples × CFG {5.0, 7.5}).
   Source: `runs/20260421_manifold_diagnosis/e2_phase3_drift/per_token_stats_cfg5.csv`.

2. **Phase III (step 35→50) is uniform amplitude scaling**
   (cos 0.989–0.996, per-token norm-ratio gini 0.005–0.011, ratio mean 1.30–1.31).
   Source: same CSV, step pair (35, 50).

3. **Step-50 teacher landing distribution is narrow**:
   mean = 7.7948, std = 0.1149 over N=105 test samples.
   Source: `runs/20260421_manifold_diagnosis/teacher_ref_n105/stats.json`.

4. **Teacher decoder almost always produces valid SDFs** at step 50:
   94 Mode B + 11 unclear + 0 Mode A in N=105.
   Source: `teacher_ref_n105/diagnosis.csv`.

5. **ODE step count 50→200 yields only ~2× CD improvement** on failure-category
   samples; helicopter_015 is the one meaningful beneficiary (2.9×).
   Track B (scale=1.0) is essentially unchanged (±3%).
   Source: `runs/20260421_manifold_diagnosis/e3_teacher_extended/step_comparison.csv`.

6. **Teacher has ~20% systematic scale bias on articulated shapes**
   (helicopter alignment_scale 0.71 → 0.83 → 0.83 at 50/100/200 step; saturates).
   Source: same step_comparison.csv.

7. **Part 1 report "teacher failed on complex shapes" was a track A/B confusion**.
   The cdData embedded in `2026-04-08_combined-analysis.html` showed track-B
   (no-alignment) values (helicopter 0.24, robot 0.33, dinosaur 0.42), while
   track A has these at CD < 1e-2 for all 3. Convergence is real under proper
   alignment.
   Source: `runs/20260421_cross_family/metrics/per_sample.csv` + E3 comparison.

### Uncertain

- **Category-level norm variance**: with 1 sample per category, we see
  ±0.2–0.3 spread (tree = 7.27, flower = 8.09). We cannot distinguish
  "shape-category dependence" from "within-category sample variance"
  without N > 1 per category.

- **The "category-invariant norm schedule"** (original Part 1 claim) was
  based on 3 samples (ball/chair/helicopter) that matched to 3 decimal
  places. N=10 shows similar tightness, but N=105 reveals that category
  means actually spread by ±0.2. The robust claim is "**narrow reference
  distribution**", not "3-digit-identical schedule".

- **tree_023 outlier**: z = −4.58 even for teacher's own output. Is this
  a shape-specific artifact (thin branching structure) or category-wide
  (all tree samples would land low)?

### Open / untested

- **VecSet-specificity vs 3D LDM universality**: whether norm tightness,
  direction preservation, and universal schedule hold on TRELLIS 2's
  sparse-voxel latent (SLAT). → **H1, priority item**.

- **Objaverse-scale diversity**: Toys4k is 105 simple objects. Does the
  narrow norm distribution hold on Objaverse's wider category base?

- **Per-category N > 1 sampling**: is there a way to get multiple samples
  per category from the existing test manifold, or does this require a new
  dataset sample?

- **Intermediate-step decoder behavior (Mode A boundary)**: Part 1 showed
  uniform -1 SDF for t < 0.4 (no isosurface at all). How does this map
  onto the norm-threshold interpretation? At what exact per-token norm
  does the decoder transition from "uniform negative" to "valid
  isosurface"?

---

## Axis B — Distillation comparison & student-vs-teacher

### Established

0. **Cross-family benchmark (9 methods × 105 samples, Track A, CD ×10⁻³, 2026-04-21)**:

   | Model | CD mean | CD median | Failures |
   |---|---|---|---|
   | teacher_50step | 8.5 | 2.5 | 0 |
   | dmd2_1step (fixed) | **9.7** | 3.0 | 0 |
   | trellis (v1 teacher) | 13.5 | 3.9 | 0 |
   | dmd1_1step | 15.1 | 3.3 | 0 |
   | trellis2 | 29.3 | 3.2 | 0 |
   | cd_4step | 31.1 | 2.9 | 0 |
   | pd_6step | 31.9 | 16.5 | 4 |
   | flashvdm | 33.3 | 2.4 | 0 |
   | mdt_dist | 116.3 | 6.4 | 0 |

   Source: `runs/20260421_cross_family/metrics/summary.csv`.
   Key take-aways:
   - **DMD2 bug fix is successful**: 260.9×10⁻³ (Part 2) → 9.7×10⁻³ (27× improvement, now best
     among single-step methods). Part 2 report's "DMD2 is catastrophic" framing is historical.
   - **Architecture-level distillation resilience differs**: H3D-2.1 teacher (8.5) and its
     distillations (9.7–33.3) stay within 4× of teacher. TRELLIS v1 teacher (13.5) →
     MDT-Dist distilled (116.3) is a 9× degradation. VecSet vs sparse-voxel distillation
     behaviors diverge.
   - **Track B CD is 5–15× worse than Track A across all methods**: TRELLIS family shows
     the largest gap (13.5 → 212), suggesting stronger alignment dependence.

1. **DMD1 produces valid SDFs universally** (0/105 Mode A, 92 Mode B + 13 unclear).
   Decoder refusal is not a DMD1 failure mode.
   Source: `runs/20260421_manifold_diagnosis/e1_dmd1_volume_logit/diagnosis.csv`.

2. **DMD1 latent norm is systematically lower than teacher** for all buckets:
   - success (CD < 5e-3, n=62): mean 7.476, z vs N=105 teacher = **−2.77**
   - intermediate (5e-3 ≤ CD < 5e-2, n=36): mean 7.338, z = **−3.97**
   - failure (CD ≥ 5e-2, n=7): mean 7.265, z = **−4.61**
   Source: `e1_dmd1_volume_logit/joined.csv` + `teacher_ref_n105/stats.json`.

3. **Spearman ρ(latent norm, log CD) = −0.530**, p = 6.1 × 10⁻⁹, N=105.
   Norm is a modest-but-robust predictor of distillation CD.
   Source: `e1_dmd1_volume_logit/` (scatter plots).

4. **3/7 DMD1 failures saturate the ICP scale clamp** (alignment_scale = 0.500);
   another 2 are adjacent (0.501, 0.555). Failure bucket has 42.9% clamp
   saturation rate vs 0% in success bucket.
   Source: `joined.csv` scale column.

5. **tree_023 is a teacher-side edge case**: teacher norm = 7.27 (z = −4.58),
   DMD1 further reduces to 6.77 (z = −8.92 vs N=105 ref). Not a pure
   student-only failure; it's teacher-edge-case amplification.
   Source: cross-reference of `teacher_ref_n105/per_sample_norm.csv` and
   `e1_dmd1_volume_logit/joined.csv`.

6. **DMD1 "bimodal failure" framing from Part 1 is misleading**. The CD
   distribution is continuous with a long tail; what looks bimodal is ICP
   scale-clamp saturation kicking in at a specific norm threshold.

7. **Part 2 DMD2 findings remain valid**: replay-buffer conditioning mismatch
   and missing GAN gradient to mu_fake were real bugs. Fixed versions still
   trail CD and DMD1 on CD metric (see cross-family benchmark).
   Source: `reports/2026-04-14_dmd2-bug-analysis.md` + `runs/20260421_cross_family/metrics/summary.csv`.

### Uncertain

- **The norm-under-shoot's cause**: training optimization artifact?
  1-step ODE approximation bias? Structural limitation of flow matching
  distillation? We have no decomposition.

- **Whether CD / DMD2 / SiD show the same norm-CD correlation** as DMD1.
  E1 was DMD1-only. Predictions:
  - CD (trajectory-following): norm closer to teacher
  - DMD2 (GAN): wider spread, possibly some Mode A
  - SiD (adversarial alone): strong Mode A (consistent with 0/105 SiD CD success)
  → **B2, priority item**.

- **Scale-clamp saturation vs latent norm**: correlated (Spearman 0.35)
  but not fully explanatory. There are non-saturated failures
  (car_020 scale=0.637, octopus_010 scale=0.555) that still have low
  norm. Direction axis may also matter.

### Open / untested

- **H4 routing threshold**: works offline? E1 data has everything needed
  to validate on existing results without new compute. → **B1,
  immediate item**.

- **CD / DMD2 / SiD norm distribution**: cheap to run (same script as E1
  with different --model-name). → **B2**.

- **tree_023 mesh visualization**: verify whether it's a topology failure
  beyond norm or just scale. → **B3**.

- **AYS-style non-uniform schedule for CD re-training**: requires
  re-training (1 day) to test. → **B4**.

---

## Cross-axis entanglement

Two places where axes genuinely entangle, not just mixed reporting:

1. **Teacher reference defines the on-manifold-ness metric for students**.
   H4 (Axis B) uses Axis A's teacher norm distribution as reference. Any
   change in Axis A (e.g., Trellis2 comparison changes our definition of
   "on-manifold") propagates to Axis B.

2. **tree_023 and other shared-failure categories**. Failures that show
   up for *both* teacher and students are not pure Axis B problems; they
   indicate Axis A limitations (capacity, training data coverage) that
   distillation cannot fix.

Everything else separates cleanly.

---

## Framing drift log

A record of how the central claim migrated during experiments. Kept so we
don't re-confuse future work.

1. **Original (Part 1, 04-08)**: "narrow decodable manifold via decoder
   refusal". Volume logit saturation at Phase I / SiD 0/105 → implied a
   decoder-centric manifold.

2. **After E1 (04-21)**: rejected. DMD1 has 0 Mode A; decoder never
   refuses valid student outputs. Unified-in-volume-logit hypothesis
   broken.

3. **Refined (after teacher_ref_n105)**: "narrow **teacher latent norm**
   distribution". Constraint is on the *teacher-reachable set*, not on
   decoder acceptance. Students fail by drifting below this distribution,
   not by crossing a decoder boundary. Measurable purely in latent
   space, no VAE decode needed.

4. **Current (this document)**: the refined claim holds with N=105
   evidence (std 0.115 stable, Spearman ρ = −0.53 for student norm
   prediction). Category-invariance is reframed as "narrow distribution
   spanning categories" rather than "3-digit-identical schedule". H1
   (Trellis2) will determine VecSet-specificity.
