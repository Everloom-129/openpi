# Rotation Equivariance of Wrist-Camera Attention

**Date**: 2026-04-12
**Branch**: tony/visualize_attention

---

## Hypothesis

The attention pattern (text→image and action→image) on the wrist camera is spatially consistent with scene content, not with the absolute pixel position in the image. Specifically: if we rotate the wrist image by angle θ before feeding it to the model, the resulting attention heatmap should rotate by the same θ. If the model is truly attending to the task-relevant object (the grasp target), the attention concentration on that object's mask should remain high regardless of rotation.

A **failure mode** would be if attention is anchored to a fixed patch location (e.g., center or top-left), in which case attention on the object mask drops sharply under rotation.

> **Confound to note**: rotating only the wrist token changes the joint attention matrix in a non-trivial way because the ext image and text tokens are unchanged. This measures *attention sensitivity*, not pure equivariance. State this explicitly in the write-up.

---

## Stage 1: Minimal Example Data Proof

**Scope**: Single episode, 2–3 frames, 4 rotation angles (0°, 90°, 180°, 270°)

### Setup
- Select one well-characterized frame where the grasp object is clearly visible in the wrist cam and has high baseline attention (`attn_on_obj_ratio` > 0.4)
- Rotate **only** the wrist image (keep ext image unchanged); apply matching rotation to the perception mask
- Run inference live, or as a fast proxy: load from H5 and rotate the 16×16 attention grid + mask in 2D

### Key checks
- Does `attn_on_obj_ratio` stay stable across rotations?
- Does the attention peak follow the object's rotated position?

### Output figure
2×4 panel:
- Top row: rotated wrist image + attention overlay (one column per angle)
- Bottom row: rotated mask vs top-10% attention patches

### Success criterion
`attn_on_obj_ratio` degrades by < 20% relative, and the attention centroid moves in the expected direction.

---

## Stage 2: Larger-Scale Validation

**Scope**: All frames from N episodes (success + failure), 8 rotation angles (0°, 45°, ..., 315°)

### Setup
- Use `--from-h5` mode; for each stored H5, rotate the wrist-patch slice of the attention grid by θ (bilinear), apply the same rotation to the resized mask
- Stratify by role (grasp / place / other) and outcome (success / failure)

### Metrics per (frame, rotation, role)
| Metric | Description |
|--------|-------------|
| `attn_on_obj_ratio` | Fraction of total attention landing on the object mask |
| `top10_iou` | IoU between top-10% attention patches and object mask |
| `attn_concentration` | Mean attn inside object / mean attn outside object |
| `centroid_displacement` | Distance (in patches) between attention centroid and object mask centroid |

---

## Stage 3: Statistical Result Figures

Three panels:

**Panel 1 — Attention stability by role**
Line plot: mean `attn_on_obj_ratio` ± 95% CI vs rotation angle (0°–315°), one line per role (grasp / place / other).
- Flat line → equivariant, object-driven attention
- Sharp dip → position-anchored attention

**Panel 2 — Centroid displacement**
Boxplot: centroid displacement (patches) at each rotation angle, grasp objects only.
- Equivariant model → ~0 displacement at all angles

**Panel 3 — Layer sensitivity heatmap**
Heatmap: mean IoU drop (relative to 0°) at 180° rotation, across all 18 layers × roles.
- Identifies which layers lose spatial consistency first

---

## Stage 4: Analysis

| Result | Interpretation |
|--------|----------------|
| Flat across rotations | Attention is object-driven, not position-anchored → supports H1 (wrist attention is semantically meaningful) |
| Drops at non-zero rotations | Model has a positional prior baked in (expected for ViT patch embeddings); report angle sensitivity and which layers are worst |
| Grasp > place > other at 0° | Confirms existing H1 result; rotation is a valid perturbation probe |

---

## Implementation Plan

1. **`viz/h1_rotate_proof.py`** — Stage 1 minimal proof script (single frame, 4 angles, panel figure)
2. **`viz/h1_rotate_validation.py`** — Stage 2 batch script using `--from-h5`, outputs per-frame CSV
3. **`viz/h1_rotate_figures.py`** — Stage 3 aggregate figures from CSV

Batch entry point mirrors the existing `h1_wrist_object_corr.py` interface:
```bash
uv run python viz/h1_rotate_proof.py <DATA_ROOT> <RESULTS_ROOT> --episode <name>
uv run python viz/h1_rotate_validation.py <DATA_ROOT> <RESULTS_ROOT>
uv run python viz/h1_rotate_figures.py <RESULTS_ROOT>/aggregate/rotate/
```
