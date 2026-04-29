---
marp: true
theme: default
paginate: true
style: |
  section {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 22px;
    padding: 40px 60px;
    color: #1a1a2e;
  }
  h1 {
    font-size: 38px;
    color: #1a1a2e;
    border-bottom: 3px solid #4a90d9;
    padding-bottom: 10px;
  }
  h2 { font-size: 28px; color: #2c3e70; margin-top: 0; }
  h3 { font-size: 22px; color: #4a90d9; }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }
  .columns3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.5rem; }
  .box { background: #f0f4ff; border-left: 4px solid #4a90d9; padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 8px 0; }
  .greenbox { background: #f0fff4; border-left: 4px solid #27ae60; padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 8px 0; }
  .orangebox { background: #fff8f0; border-left: 4px solid #e67e22; padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 8px 0; }
  .redbox { background: #fff0f0; border-left: 4px solid #e74c3c; padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 8px 0; }
  .tag { display: inline-block; background: #4a90d9; color: white; padding: 2px 10px; border-radius: 12px; font-size: 14px; margin-right: 6px; }
  .tag-green { display: inline-block; background: #27ae60; color: white; padding: 2px 10px; border-radius: 12px; font-size: 14px; margin-right: 6px; }
  .tag-orange { display: inline-block; background: #e67e22; color: white; padding: 2px 10px; border-radius: 12px; font-size: 14px; margin-right: 6px; }
  .tag-red { display: inline-block; background: #e74c3c; color: white; padding: 2px 10px; border-radius: 12px; font-size: 14px; margin-right: 6px; }
  section.title { background: linear-gradient(135deg, #1a1a2e 0%, #2c3e70 60%, #4a90d9 100%); color: white; display: flex; flex-direction: column; justify-content: center; }
  section.title h1 { color: white; border-bottom: 3px solid rgba(255,255,255,0.4); font-size: 42px; }
  section.title h2 { color: rgba(255,255,255,0.85); font-weight: 300; }
  section.title p { color: rgba(255,255,255,0.7); }
  code { background: #eef2ff; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; color: #2c3e70; }
  table { border-collapse: collapse; width: 100%; font-size: 19px; }
  th { background: #2c3e70; color: white; padding: 8px 12px; }
  td { padding: 7px 12px; border-bottom: 1px solid #dce4f0; }
  tr:nth-child(even) td { background: #f5f8ff; }
  .small { font-size: 18px; }
  .tiny { font-size: 16px; color: #555; }
---

<!-- _class: title -->

# Explaining Pi0.5 in the Lab
## A Mechanistic Look into Attention of VLAs — Final Report

Tony Wang · Vineet Pasumarti · Faraz Rahman
University of Pennsylvania · Spring 2026

---

# Agenda

1. **Motivation & Hypotheses** — two camps on VLA attention
2. **Toolchain** — attention capture, HDF5 pipeline, dashboard
3. **H1** — Layer-wise object localization (real DROID)
4. **H2** — Counterfactual prompt invariance
5. **H3** — Causal fidelity: static masking + closed-loop KV intervention (sim)
6. **Cross-model & cross-checkpoint** — π₀ vs π₀.₅ vs GR00T-N1.7
7. **Negative results & limitations**
8. **Future work** — 3D-aware attention, subtask head

---

# Motivation

<div class="columns">
<div>

### VLAs look smart, fail often
- π₀.₅ scores well on narrow benchmarks
- Real-world DROID rollouts are brittle — *spatial reasoning fails*
- Question: **why** and **where** in the network?

### Two competing hypotheses

<div class="greenbox">
<b>Optimistic:</b> Attention is structured and language-conditioned —
distinct layers handle objects, gripper, plan.
</div>

<div class="redbox">
<b>Pessimistic:</b> Attention is dominated by VLM visual saliency priors,
largely invariant to instructions.
</div>

</div>
<div>

### Three findings to validate at scale

1. Some PaliGemma layers **do** localize objects.
2. Attention is **highly correlated across counterfactual prompts**.
3. **Salience ≠ causality**: deep layers look prominent but barely affect actions; mid layers do the real work.

<div class="box">
This deck reports validation across <b>real DROID</b> and a new <b>closed-loop robocasa-365 sim</b> we built.
</div>

</div>
</div>

---

# Setup

<div class="columns">
<div>

### Models
- **π₀.₅-DROID** — flow-matching, state-as-text tokens
- **π₀-DROID** — flow-matching, no state tokens
- **π₀.₅-LIBERO**, **π₀.₅-robocasa365**
- **GR00T-N1.7-DROID** (NVIDIA, baseline)

### Token layout (PaliGemma backbone)
```
[0:256]   ext img patches
[256:512] wrist img patches
[512:768] zero pad
[768:N]   text + (π₀.₅) state-discretization
[N:N+8]   action tokens (suffix)
```
18 layers × 8 heads.

</div>
<div>

### Data sources
- **Real**: full DROID dataset (success / failure splits per episode)
- **Sim (new)**: robocasa-365 + robosuite tasks
  - Lift, Stack, Door, PickPlaceCan, NutAssemblySquare

### Object grounding
- DINO-X / Gemini segmentation masks
- Rule-based instruction-noun → object label matcher (`viz/config/object_matching.yaml`)

</div>
</div>

---

# Toolchain (Phase 1 — over-delivered)

<div class="columns">
<div>

### Attention capture
- **In-RAM buffer** in `gemma_pytorch.py` — no `.npy` files
- Captures both **prefix** (text→image) and **suffix** (action→image / action→text) in one `policy.infer()` call
- Per-frame HDF5: `/prefix/layer_i/text_to_img + full`, `/suffix/layer_i/action_to_img`, GT + predicted actions

### Multi-GPU pipeline
`viz/pipeline_mp.py` — episode-level parallelism, 4 workers / 2 GPUs, ~3.5× speedup

</div>
<div>

### Streamlit dashboard (`viz/dashboard/`)
12 tabs:
- grid heatmap, image overlay, full attention matrix
- **action view**, **trajectory**, denoising
- **counterfactual**, **ckpt compare**, **episode compare**
- CAG, image saliency, dataset browser

Three modes:
- Offline (HDF5)
- Results (Benchmark) — RESULTS_ROOT
- Online (live inference, GPU auto-select)

</div>
</div>

<div class="tiny">Refs: docs/0319_attn_pipeline_design.md, 0319_multiprocess_pipeline.md, 0428_viz_inventory.md</div>

---

# H1 · Do PaliGemma layers localize objects?

<div class="columns">
<div>

### Method (`viz/h1_wrist_object_corr.py`)
For each frame in DROID, for each of 18 layers:
1. Extract text→wrist attention (mean over real text tokens)
2. Threshold top-10% (90th pct)
3. Match to DINO-X mask of the instruction's target object
4. Compute **precision, recall, IoU, attn-on-object ratio, concentration**

Aggregate per-layer, split by **outcome** (success/failure) and **role** (grasp/place/other).

</div>
<div>

### Headline (real DROID)

<div class="box">
<b>Layers 5–10 are the object localizers.</b>
Top-10% precision peaks at layer [TODO: L?] = [TODO: %],
falls to ~baseline by layer 17.
</div>

<div class="orangebox">
<b>Success vs failure:</b> object-attention precision differs by [TODO: Δ%] —
small but present in mid layers, vanishes in deep layers.
</div>

`[FIGURE: per-layer top10 precision/recall, success vs failure — viz/h1_wrist_object_corr.py output]`

</div>
</div>

---

# H1 · Rotation equivariance probe

<div class="columns">
<div>

### Question
If a layer truly *localizes* the object, attention should **rotate with the image**, not stay pixel-anchored.

### Method (`h1_rotate_proof.py`, `0412_rotate_exp.md`)
Wrist image rotated 0° / 90° / 180° / 270°; re-run inference.

Metrics: `attn_on_obj_ratio`, top-10 IoU vs rotated mask, **centroid displacement**.

</div>
<div>

### Result

<div class="greenbox">
Mid-layers (5–10): centroid <b>tracks the object</b> across rotations →
genuine object-conditioned attention.
</div>

<div class="redbox">
Deep layer 17: centroid <b>stays pixel-anchored</b> (gripper region) →
position prior, not object grounding.
</div>

`[FIGURE: 4-angle panel × 3 layers]`
[TODO: insert numbers — IoU @ L7 vs L17]

</div>
</div>

---

# H2 · Are attention maps invariant to the prompt?

<div class="columns">
<div>

### Method (`viz/h2_cf_prompt.py`)
Same scene (pineapple, frame 50). 6 prompts:
- baseline: *"find the pineapple toy and pick it up"*
- counterfactuals: duck / banana / cat / bottle
- empty prompt

Layers [5, 10, 17]. Compute, per layer:
- mean / abs-mean diff
- L2 distance, **Pearson correlation** vs baseline

Three "uncertainty" scores (`0320_cf_uncertainty_score.md`):
**Memory Reliance, Instruction Sensitivity, Attention Shift.**

</div>
<div>

### Result

<div class="redbox">
Correlation(baseline, counterfactual) = [TODO: ~0.9+] across layers.
L2 distance is small relative to within-episode temporal variation.
</div>

<div class="orangebox">
Even an <b>empty prompt</b> produces near-identical attention →
strong evidence for the <b>visual-saliency-over-semantics</b> camp.
</div>

`[FIGURE: 6×3 grid — prompt × layer heatmaps]`
`[FIGURE: per-layer correlation bar chart]`

</div>
</div>

---

# H3 · Static input-masking (per-frame causal probe)

<div class="columns">
<div>

### Method (`viz/h3_casual_fidelity.py`)
For one frame:
- baseline action `A_orig = π(image)`
- mask **top-10%** attention pixels → `A_mask_high`
- mask **equal-area random low-attention** pixels → `A_mask_low`
- Fidelity = MSE(A_orig, A_mask_high) − MSE(A_orig, A_mask_low)

If attention is causal, masking high-attention regions should hurt more.

</div>
<div>

### Result (preview)

<div class="orangebox">
Fidelity score is <b>small or negative</b> for deep layers (10, 17):
masking <i>low</i>-attention background often perturbs actions
<b>at least as much</b> as masking the salient region.
</div>

[TODO: per-layer fidelity bar chart with CI]

> Limitation: input-space masking confounds the policy with OOD pixels.
> → motivates the closed-loop study on next slide.

</div>
</div>

---

# H3 · Closed-loop KV-cache intervention (NEW)

<div class="columns">
<div>

### Setup (`viz_sim/eval_perturb.py`, sim)
Inside `gemma_pytorch.py`, scale the **KV-cache value** at the
argmax/argmin attention position of layer 7, *every step* of the rollout.

7 conditions per task, paired seeds:
- baseline
- ext_zero_max / ext_zero_min / ext_strengthen_max
- wrist_zero_max / wrist_zero_min / wrist_strengthen_max

5 robosuite tasks × N episodes.

Metrics: success rate, max reward, **EEF L2 trajectory deviation**, **action L2**, paired Wilcoxon vs baseline.

</div>
<div>

### Headline (robocasa-365 sim)

<div class="redbox">
Zeroing the <b>min</b>-attention patch degrades success comparably to
zeroing the <b>max</b>-attention patch.
</div>

<div class="orangebox">
Strengthening max-attention <b>does not help</b> and sometimes hurts.
</div>

| Condition | Δ rmax | p (Wilcoxon) | Δ EEF L2 |
|---|---|---|---|
| ext_zero_max | [TODO] | [TODO] | [TODO] |
| ext_zero_min | [TODO] | [TODO] | [TODO] |
| ext_strengthen_max | [TODO] | [TODO] | [TODO] |
| wrist_zero_max | [TODO] | [TODO] | [TODO] |
| wrist_zero_min | [TODO] | [TODO] | [TODO] |

<div class="tiny">Aggregate from `build_perturb_report.py` → `results/report.md`.</div>

</div>
</div>

---

# H3 · What does this mean?

<div class="greenbox">
<b>Attention salience is not a faithful explanation of causal influence in π₀.₅.</b>
The "shortcut" view from the proposal is supported by closed-loop evidence —
not just static input masking.
</div>

<div class="columns">
<div>

### Where does causality live?
- Input-masking + KV-perturbation both implicate **mid layers (5–10)**, not deep layers (17).
- Consistent with the proposal's prediction: *mid layers carry causal load despite lower visual prominence.*

</div>
<div>

### Why deep layers look "smart" but aren't
- Deep-layer attention concentrates on gripper / wrist — a learned **pose prior**.
- Removing the highlighted region is fine; the policy's action heuristic doesn't actually use it.

</div>
</div>

---

# Cross-model: π₀ vs π₀.₅ vs GR00T-N1.7

<div class="columns">
<div>

### Setup
- Same robosuite tasks via `viz_sim/run_all_evals.sh`
- `build_combined_viz.py` → `results/combined.html`
- π₀ / π₀.₅: WebSocket policy server with attention export
- GR00T-N1.7-DROID: ZMQ baseline (no attention export)

### Per-task success rate

| Task | π₀-DROID | π₀.₅-DROID | π₀.₅-rc365 | GR00T-N1.7 |
|---|---|---|---|---|
| Lift | [TODO] | [TODO] | [TODO] | [TODO] |
| Stack | [TODO] | [TODO] | [TODO] | [TODO] |
| Door | [TODO] | [TODO] | [TODO] | [TODO] |
| PickPlaceCan | [TODO] | [TODO] | [TODO] | [TODO] |
| NutAssemblySquare | [TODO] | [TODO] | [TODO] | [TODO] |

</div>
<div>

### π₀ vs π₀.₅ attention (ckpt-compare view)
- π₀.₅ adds ~80 state-discretization tokens after the instruction; attention shifts toward those tokens in deep layers.
- π₀ (no state tokens) shows a cleaner instruction→image attention but **same low CF sensitivity** as π₀.₅.
- → The *visual-saliency shortcut* is **not** caused by state-token leakage.

`[FIGURE: side-by-side ckpt-compare dashboard screenshot]`

<div class="tiny">Caveat: π₀.₅-rc365 has a documented obs/action-space mismatch in our sim
(`docs/0428-robocasa-actionspace.md`); cross-model numbers are indicative.</div>

</div>
</div>

---

# Negative & null results (worth reporting)

<div class="columns">
<div>

### Action-self-attention does not separate success/failure
(`viz/action/`, `0409_finding_action.md`)

At layer 17, denoising step 9, action-self-attention variance is statistically indistinguishable between successful and failed episodes.

Discriminative signal lives **earlier** (steps 0–5) or in **mid layers**, not in the saturated final action representation.

</div>
<div>

### Action denoising trajectory
- Step 0: action→**text** dominates (≈1.0)
- Step 9: action→text drops to ~0.4; action→**self** grows to 0.6–0.8
- Wrist image dominates over exterior throughout

`[FIGURE: action→{text, self, ext, wrist} vs denoising step]`

→ *Language conditioning is consumed early, then discarded.* Consistent with H2.

</div>
</div>

---

# Limitations

<div class="orangebox">
<b>Sim-to-train gap on robocasa-365.</b>
π₀.₅-rc365 was trained on robocasa state (eef + base); our sim ships DROID joint obs.
The 12-dim robocasa action is sliced to 8. Numbers are <i>relative</i>, not absolute.
(See <code>docs/0428-robocasa-actionspace.md</code>.)
</div>

<div class="orangebox">
<b>Counterfactual study is single-frame</b> at full statistical scope.
Larger CF dataset is captured in HDF5 but the cross-episode aggregation
(<code>0320_cf_uncertainty_score.md</code>) is not yet rolled into the report numbers.
</div>

<div class="orangebox">
<b>KV-perturbation only at layer 7.</b>
A full layer-sweep of the closed-loop intervention would tighten the
"mid-layers carry causality" claim.
</div>

<div class="orangebox">
<b>Phase-3 "3D fix" not implemented.</b> Subtask-head plan
(<code>0428_subtask_module_plan.md</code>) drafted but not trained.
</div>

---

# What we shipped vs proposal

| Proposal | Status |
|---|---|
| Phase 1 — toolchain on π₀.₅ | <span class="tag-green">DONE+</span> 12-tab dashboard, multi-GPU, multi-ckpt |
| H1 object localization (DROID) | <span class="tag-green">DONE</span> 18-layer per-frame metrics + rotation probe |
| H2 counterfactual invariance | <span class="tag">PARTIAL</span> single-frame done; episode-scale aggregation pending |
| H3 causal fidelity | <span class="tag-green">DONE+</span> static masking **+ new closed-loop KV intervention** |
| Phase 2 large-scale (Libero / SimplerEnv) | <span class="tag-orange">PIVOTED</span> → robocasa-365 + robosuite |
| Phase 3 "3D fix" / depth tokens | <span class="tag-red">DROPPED</span> |
| Subtask language head | <span class="tag-orange">PLANNED</span> design doc only |
| GR00T-N1.7 baseline | <span class="tag-green">BONUS</span> not in proposal, added |

---

# Future work

<div class="columns">
<div>

### Short-term
- Full layer-sweep of KV-perturbation (not just L7)
- Episode-scale CF metrics: roll up Memory Reliance / Instruction Sensitivity / Attention Shift
- Pi0.5-rc365 action-space fix → re-run sim with native robocasa obs

### Medium-term
- **Subtask language head** — interpretable intermediate between prompt and action flow
- Activation patching across denoising steps (not just layers)

</div>
<div>

### Long-term (CoRL aim)
- **3D-aware attention** — VGGT-style depth tokens fused into PaliGemma backbone
- "Slow-thinking" affordance verification at test time
- Train-time loss to **decorrelate attention from visual saliency** and bind to language

<div class="greenbox">
The closed-loop KV-intervention infra we built is a ready evaluation harness for any of these.
</div>

</div>
</div>

---

<!-- _class: title -->

# Thanks

Tony Wang · Vineet Pasumarti · Faraz Rahman

Code: `github.com/.../openpi_vis` (branch `tony/visualize_attention`)
Dashboard: `bash viz/start_app.sh`
Sim eval: `bash viz_sim/perturb_orch.sh`

Questions?
