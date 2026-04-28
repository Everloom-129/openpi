# Findings of Action→Image Attention

Dataset: `faraz_action/left`, 46 episodes (1769 frames), episode-fair averaging.
Denoising steps compared: **0** (most noisy) / **5** (mid) / **9** (clean action).

![Line chart](docs/img/denoising_attn.png)
![Step × layer × group grid](docs/img/denoising_attn_grid.png)

---

## Key Observations

### 1. Text attention dominates at step 0, early layers
- At layer 0, step 0: text attention ≈ **1.0** — almost the entire attention budget goes to text tokens at the start of denoising.
- By step 5 it drops to ~0.5 at L0, and by step 9 to ~0.4. The biggest change happens in the **first half** of denoising (step 0 → 5), not the second half.
- Interpretation: the model begins denoising by heavily consulting the language instruction, then gradually shifts away.

### 2. Action self-attention grows monotonically across denoising steps
- Step 0 action self-attention is near **0** at all layers.
- By step 9, late layers (L14–L17) reach **0.6–0.8** — action tokens increasingly attend to each other as the action is refined toward the final output.
- Clear ordering across all layers: step 9 > step 5 > step 0.
- Interpretation: as denoising progresses the action representation becomes self-consistent; the tokens "converge on themselves."

### 3. Wrist camera consistently dominates over ext camera
- Wrist attention range: 0.1–0.4 vs. ext camera: 0.05–0.25 across all steps and layers.
- Both camera streams peak in early layers (L0–L3) and at step 0.
- Wrist attention falls off faster than ext camera as denoising progresses — the wrist view is most consulted when the action is most uncertain.

### 4. Early layers are the most step-sensitive
- The gap between step 0 and step 9 is largest at **L0–L2** for all four groups.
- Mid layers (L6–L12) converge: all three step curves are nearly indistinguishable.
- Late layers (L14–L17) diverge again, driven by the rising action self-attention at step 9.

### 5. Steps 5 and 9 are very similar; most dynamics happen in step 0 → 5
- The step-5 and step-9 curves track closely for ext camera, wrist camera, and text tokens.
- Only action self-attention continues to grow noticeably from step 5 → 9.
- Implication: the first half of the denoising trajectory (0 → 5) is where the policy resolves most of the uncertainty about which tokens to attend to.

### 6. Layer 17 (final layer) spike at step 0
- Ext camera shows an anomalous spike at L17 under step 0 that disappears by step 5.
- Suggests the last layer briefly over-relies on the exterior camera when the action is maximally noisy, but this resolves quickly.

---

## Experiment: Does Action Self-Attention Predict Episode Outcome?

**Hypothesis**: Failure episodes might show higher variance or lower magnitude in action self-attention at the clean denoising step (step 9, layer 17), reflecting an indecisive or inconsistent policy.

**Design**:
- Dataset: `faraz_action/left`, 46 episodes (success=11, failure=35), 1769 frames total.
- Metric: action self-attention mass at **layer 17, denoising step 9** — the final representation just before the output.
- Two statistics computed per episode:
  - **Within-episode std**: std of action_self across frames (intra-episode variability).
  - **Episode mean**: mean of action_self across frames.
- Statistical test: one-sided Mann-Whitney U (H₁: failure > success), since Mann-Whitney is non-parametric and robust to small/unequal group sizes.
- Episodes with only 1 frame excluded from the std analysis (std undefined).

**Results** (`viz/action/analyze_variance_by_outcome.py`):

| Metric | Success (n=11) | Failure (n=33–35) | MW p-value |
|---|---|---|---|
| Within-episode std | mean=0.0245, median=0.0241 | mean=0.0226, median=0.0222 | 0.965 |
| Episode mean | mean=0.9020, median=0.9020 | mean=0.9027, median=0.9012 | 0.408 |

![Outcome variance plot](docs/img/outcome_variance.png)

**Conclusion**: Neither the variance nor the mean of action self-attention at layer 17, step 9 differs significantly between success and failure episodes. The effect is in the *wrong direction* for variance (failure is slightly *less* variable). Action self-attention at late layers / clean steps saturates near 0.9 for all episodes — the model reliably forms a self-consistent action representation regardless of whether the episode succeeds.

**Implication**: If a discriminative signal exists, it is more likely in:
- Earlier denoising steps (step 0), where the model is most uncertain.
- Camera attention ratios (wrist vs ext), which may reflect whether the model is tracking the relevant object.
- Intermediate layers (L6–L12), which showed the least step-sensitivity but may carry outcome-predictive content.
