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
  h2 {
    font-size: 28px;
    color: #2c3e70;
    margin-top: 0;
  }
  h3 {
    font-size: 22px;
    color: #4a90d9;
  }
  .columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
  }
  .columns3 {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1.5rem;
  }
  .box {
    background: #f0f4ff;
    border-left: 4px solid #4a90d9;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0;
  }
  .greenbox {
    background: #f0fff4;
    border-left: 4px solid #27ae60;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0;
  }
  .orangebox {
    background: #fff8f0;
    border-left: 4px solid #e67e22;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0;
  }
  .redbox {
    background: #fff0f0;
    border-left: 4px solid #e74c3c;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0;
  }
  .tag {
    display: inline-block;
    background: #4a90d9;
    color: white;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 14px;
    margin-right: 6px;
  }
  .tag-green {
    display: inline-block;
    background: #27ae60;
    color: white;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 14px;
    margin-right: 6px;
  }
  .tag-orange {
    display: inline-block;
    background: #e67e22;
    color: white;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 14px;
    margin-right: 6px;
  }
  section.title {
    background: linear-gradient(135deg, #1a1a2e 0%, #2c3e70 60%, #4a90d9 100%);
    color: white;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  section.title h1 {
    color: white;
    border-bottom: 3px solid rgba(255,255,255,0.4);
    font-size: 42px;
  }
  section.title h2 {
    color: rgba(255,255,255,0.85);
    font-weight: 300;
  }
  section.title p {
    color: rgba(255,255,255,0.7);
  }
  code {
    background: #eef2ff;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.9em;
    color: #2c3e70;
  }
  table {
    border-collapse: collapse;
    width: 100%;
    font-size: 19px;
  }
  th {
    background: #2c3e70;
    color: white;
    padding: 8px 12px;
  }
  td {
    padding: 7px 12px;
    border-bottom: 1px solid #dce4f0;
  }
  tr:nth-child(even) td {
    background: #f5f8ff;
  }
---

<!-- _class: title -->

# Attention Visualization for VLA Models
## Progress Report — Pi0 / Pi0.5

Tony Wang · March 25, 2026
Mentor: Edward

---

# Agenda

1. **Motivation** — Why visualize attention in VLA?
2. **What I Built** — Infrastructure overview
3. **Research Hypotheses** — H1, H2, H3
4. **Preliminary Findings** — What we observe so far
5. **Open Questions** — Need your input
6. **Next Steps** — Proposed plan

---

# Motivation

## VLA models are black boxes

<div class="columns">
<div>

**The model knows what to do — but we don't know *why*.**

Given:
- Side camera (224×224)
- Wrist camera (224×224)
- Text instruction (e.g. `"pick up the cube"`)

Pi0.5 predicts 8 joint velocities.

**But which pixels drove that decision? Does the instruction matter?**

</div>
<div>

### Why it matters

<div class="box">
🔍 <strong>Interpretability</strong><br/>
Understand what the model attends to during manipulation
</div>

<div class="box">
🛡️ <strong>Safety</strong><br/>
Detect when the model focuses on wrong regions
</div>

<div class="box">
🎯 <strong>Fine-tuning</strong><br/>
Know which layers/heads encode task-relevant semantics
</div>

<div class="box">
🔬 <strong>Research</strong><br/>
Compare model families (π₀ vs π₀.₅) mechanistically
</div>

</div>
</div>

---

# Model Architecture — Token Layout

## Pi0.5 unified sequence (PaliGemma + Action Expert)

```
[0 : 256]         [256 : 512]       [512 : 768]      [768 : N]         [N : N+8]
ext_camera         wrist_camera      zero_padding      text_tokens       action_tokens
  256 patches        256 patches        (masked)         ~100 tokens        8 tokens
  SigLIP emb.        SigLIP emb.                        Gemma tok.         noisy Δ
```

**Self-attention over a unified multi-modal sequence — no explicit cross-attention.**
Causal mask makes action tokens attend to all prefix tokens.

<div class="columns3">

<div class="box">
<strong>Prefix pass</strong><br/>
Bidirectional attention<br/>
image ↔ text ↔ image<br/>
→ <em>text-to-image attention</em>
</div>

<div class="box">
<strong>Suffix pass</strong><br/>
Action tokens attend to prefix<br/>
action → image + text<br/>
→ <em>action-to-image attention</em>
</div>

<div class="box">
<strong>18 layers × 8 heads</strong><br/>
144 attention maps per forward pass<br/>
per frame, per camera
</div>

</div>

---

# What I Built — Toolchain Overview

<div class="columns">
<div>

### Capture
- Modified `gemma_pytorch.py` — in-RAM buffer (no npy files)
- Both **prefix** and **suffix** attention captured
- Supports π₀, π₀.₅, π₀-FAST checkpoints

### Storage
- HDF5 per frame: `{frame:05d}.h5`
- Stores: images / attention / gt_action / pred_action / meta
- Counterfactuals: `{frame:05d}_{cf_key}.h5`

### Pipeline
- `pipeline.py` — single GPU, batch offline
- `pipeline_mp.py` — multi-GPU, episode-level parallel
- Resume support via `pi05.md` marker files

</div>
<div>

### Dashboard (Streamlit)

| Tab | Purpose |
|---|---|
| Grid Heatmap | 16×16 attention per word token |
| Image Heatmap | Attention overlaid on RGB |
| Attn Matrix | Full sequence attention |
| Action View | Where actions attend; pred vs GT |
| Trajectory | Multi-frame attention + action |
| Counterfactual | Prompt variation comparison |
| Ckpt Compare | π₀ vs π₀.₅ side-by-side |

</div>
</div>

---

# Hypothesis Framework

## Three research questions

<div class="columns3">

<div class="box">
<span class="tag">H1</span> <strong>Causal Fidelity</strong><br/><br/>
Are high-attention regions causally important for action prediction?<br/><br/>
<em>Test: occlude top-10% attention → action change vs occlude random region</em>
</div>

<div class="box">
<span class="tag">H2</span> <strong>Instruction Grounding</strong><br/><br/>
Does changing the prompt shift attention to the named object?<br/><br/>
<em>Test: counterfactual prompts — object swap / negation / empty</em>
</div>

<div class="box">
<span class="tag">H3</span> <strong>Layer Specialization</strong><br/><br/>
What functional roles do different transformer layers play?<br/><br/>
<em>Test: entropy + attention focus across layers 0–17</em>
</div>

</div>

<br/>

### Counterfactual prompt variants (implemented)

| Method | Example |
|---|---|
| **object_swap** | `"pick up the cube"` → `"pick up the pen"` |
| **negation** | `"do not pick up anything"` |
| **empty** | `""` (no instruction) |
| **style** | `"pick it up"` / verbose rephrase |

---

# Preliminary Findings — H3: Layer Specialization

## Qualitative observation across layers 0–17

<div class="columns">
<div>

### Pattern observed

<div class="greenbox">
<strong>Layer 0–2 (Early layers)</strong><br/>
High entropy, diffuse attention<br/>
Sensitive to low-level pixel features<br/>
Occluding any region disrupts actions
</div>

<div class="greenbox">
<strong>Layer ~10 (Semantic layer)</strong><br/>
Localizes named objects accurately<br/>
Strong binocular fusion (ext + wrist)<br/>
Responds to instruction changes
</div>

<div class="greenbox">
<strong>Layer 17 (Pre-output layer)</strong><br/>
Strongly focused on <strong>wrist camera</strong><br/>
Action tokens concentrate here<br/>
"Hand-eye coordination" layer
</div>

</div>
<div>

### What this suggests

<div class="box">
Different layers encode different functional roles — consistent with findings in language-only transformers (Tenney et al.) but never shown for VLA.
</div>

<div class="orangebox">
⚠️ <strong>Status:</strong> qualitative only, 1 episode<br/>
Need: systematic measurement across layers + episodes
</div>

**Proposed metric**: attention entropy per layer
```
H(layer) = -Σ p·log(p)
```
Low entropy = focused/specialized
High entropy = distributed/generic

</div>
</div>

---

# Preliminary Findings — H2: Instruction Grounding

## Does the model "listen" to the instruction?

<div class="columns">
<div>

### Counterfactual attention shift

When prompt changes from `"pick up the cube"` → `"pick up the pen"`:

- Attention shifts away from cube region
- Wrist camera attention redistributes
- Effect strongest at layers 8–14

**Memory Reliance Score** (proposed):
```
MR(frame) = ‖action_baseline − action_empty‖_F
```
High score → model depends on instruction
Low score → vision-only behavior

</div>
<div>

### Instruction Sensitivity Score (proposed)

```
IS(frame) = mean( Var_{prompts}( pred_action ) )
```

Variance over prompt dimension, averaged over action dims.

<div class="greenbox">
✅ Data pipeline is complete — all CF variants already saved as HDF5
</div>

<div class="orangebox">
⚠️ Aggregation code not yet written<br/>
→ scores exist conceptually, not numerically
</div>

</div>
</div>

---

# Preliminary Findings — H1: Causal Fidelity

## Do high-attention regions actually matter?

<div class="columns">
<div>

### Occlusion experiment design

```
Original image
    ↓
Run inference → action_orig
    ↓
Occlude top-10% attention patches
    ↓
Run inference → action_high
    ↓
Occlude random 10% patches (baseline)
    ↓
Run inference → action_low
```

**Fidelity score**:
```
F = MSE(orig, high) − MSE(orig, low)
```
`F > 0` → attention is causally meaningful

</div>
<div>

### Early observations

<div class="greenbox">
✅ Occluding high-attention region → larger action magnitude change (scale disruption)
</div>

<div class="greenbox">
✅ Occluding background → affects action direction more than scale (context disruption)
</div>

<div class="orangebox">
⚠️ Tested on ~3 frames only (duck episode)<br/>
Not yet statistically significant
</div>

<div class="box">
Interesting: background occlusion affects action <em>direction</em> more than <em>magnitude</em> — background may encode spatial reference frame.
</div>

</div>
</div>

---

# Checkpoint Comparison — π₀ vs π₀.₅

## New: side-by-side inference on same input frame

<div class="columns">
<div>

### What's implemented
- Both models run forward pass on identical input
- Shared controls: token selector, layer, camera, aggregation
- Four views: Grid / Image Heatmap / Attn Matrix / Entropy Curves

### Key architectural difference
| | π₀ | π₀.₅ |
|---|---|---|
| Text format | instruction only | instruction + robot state |
| Text tokens | ~20–50 | ~100 |
| Token labels | instruction-only | state-augmented |

</div>
<div>

### Why this matters

<div class="box">
π₀.₅ tokenizes robot joint state as discrete text tokens → the model can "read" state from language.
</div>

<div class="box">
π₀ gets state as a continuous vector — outside the attention sequence entirely.
</div>

**Research question this enables:**
> Does π₀.₅'s state-in-language representation produce qualitatively different attention patterns than π₀'s continuous-state conditioning?

<div class="orangebox">
⚠️ Comparison not yet run on real data
</div>

</div>
</div>

---

# Open Questions — Need Your Input

<div class="columns">
<div>

### 1. Which RQ should we commit to?

<div class="box">
<strong>Option A — Interpretability</strong><br/>
"Does VLA attention reflect semantic grounding?"<br/>
→ H1 + H2 combined
</div>

<div class="box">
<strong>Option B — Robustness</strong><br/>
"When does the model rely on language vs vision?"<br/>
→ Memory Reliance Score at scale
</div>

<div class="box">
<strong>Option C — Model Comparison</strong><br/>
"How does π₀ vs π₀.₅ differ mechanistically?"<br/>
→ Ckpt Compare + layer analysis
</div>

</div>
<div>

### 2. Scale of experiments

- How many episodes are "enough" for a statistical claim?
- Should I use the full toy cube benchmark, or a curated subset?

### 3. Methodology gap

- Is occlusion-based fidelity (H1) the right test, or is there a better causal intervention?
- Should attention shift be measured with L2, KL divergence, or something else?

### 4. Related work I may be missing

- Are there existing interpretability papers on VLA / diffusion policy?
- Any known results on layer specialization in vision-language models?

</div>
</div>

---

# Next Steps — Proposed Plan

<div class="columns">
<div>

### Short-term (2 weeks)

<div class="greenbox">
<span class="tag-green">Week 1</span><br/>
Run H1 fidelity test at scale<br/>
≥20 episodes, all layers<br/>
→ first quantitative result
</div>

<div class="greenbox">
<span class="tag-green">Week 2</span><br/>
Implement Memory Reliance Score<br/>
Aggregate CF HDF5 data already saved<br/>
→ per-frame + per-episode scores
</div>

### Medium-term (1 month)

<div class="box">
Run π₀ vs π₀.₅ ckpt comparison on standardized episodes
</div>

<div class="box">
Layer entropy curves across full benchmark
</div>

</div>
<div>

### What I need to pause

<div class="redbox">
🛑 No new dashboard features until first quantitative result is written up
</div>

### Decision needed today

1. Confirm primary RQ (A / B / C or something else)
2. Confirm scale of experiments
3. Any suggested papers to read

### Longer-term question

> Is there a path toward a workshop paper (e.g. CoRL, RSS, ICRA)?
> What would the minimum viable contribution look like?

</div>
</div>

---

# Summary

<div class="columns">
<div>

### What's done ✅

- Complete attention capture pipeline (prefix + suffix)
- HDF5 batch storage with multi-GPU parallelism
- Interactive dashboard with 7 views
- Counterfactual prompt infrastructure
- Checkpoint comparison (π₀ vs π₀.₅)
- Preliminary qualitative findings on layer specialization

</div>
<div>

### What's needed next 🎯

- **Commit to one RQ** and stop expanding scope
- **First quantitative result**: H1 fidelity over ≥20 episodes
- **Written summary** of preliminary findings for future reference

### The core claim (draft)

<div class="box">
<em>"In Pi0.5, transformer layers specialize functionally: early layers encode low-level features, mid layers ground language to objects, and late layers concentrate on wrist-camera regions critical for action generation."</em>
</div>

</div>
</div>

---

<!-- _class: title -->

# Thank You

**Questions / Guidance Welcome**

Repo: `openpi` branch `tony/visualize_attention`
Dashboard: `uv run streamlit run viz/dashboard/app.py`

---

# Appendix A — HDF5 Schema

```
{frame:05d}.h5
├── /meta
│   ├── prefix_len       (int)         768
│   ├── seq_len          (int)         full sequence length
│   ├── instruction      (str)
│   ├── token_texts      (str[])       per-token labels
│   └── n_real_tokens    (int)
├── /images
│   ├── exterior         uint8(224,224,3)
│   └── wrist            uint8(224,224,3)
├── /prefix/layer_{i}
│   ├── text_to_img      float32(8, n_text, 512)   ← text→image attention
│   └── full             float32(8, seq, seq)
├── /suffix/layer_{i}
│   └── action_to_img    float32(n_heads, 8, 512)  ← action→image attention
├── /gt_action           float32(8, 8)
└── /pred_action         float32(8, 8)
```

---

# Appendix B — Score Definitions

| Score | Formula | Interpretation |
|---|---|---|
| **Memory Reliance** | `‖action_base − action_empty‖_F` | High = depends on instruction |
| **Instruction Sensitivity** | `mean(Var_prompts(pred_action))` | High = sensitive to wording |
| **Attention Shift** | `‖mean_heads(attn_base) − mean_heads(attn_cf)‖_2` | High = looks at different patches |
| **Fidelity** | `MSE(orig, high_mask) − MSE(orig, low_mask)` | Positive = attention is causal |
| **Entropy** | `−Σ p·log(p)` over image patches | Low = focused; High = diffuse |

---

# Appendix C — Checkpoints Supported

| Model | Config | Notes |
|---|---|---|
| π₀.₅ DROID | `pi05_droid` | Primary model; state-in-text |
| π₀ DROID | `pi0_droid` | Continuous state conditioning |
| π₀ ALOHA Towel | `pi0_aloha_towel` | Different embodiment |
| π₀-FAST DROID | `pi0_fast_droid` | Autoregressive + FAST tokenizer |

All converted from JAX → PyTorch via `convert_jax_model_to_pytorch.py`
