# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repo is a fork/extension of the [openpi](https://github.com/Physical-Intelligence/openpi) codebase — open-source Vision-Language-Action (VLA) models from Physical Intelligence — extended with attention visualization tools. The main additions live in `viz/`.

Models supported: **π₀** (flow-based), **π₀-FAST** (autoregressive + FAST tokenizer), **π₀.₅** (upgraded generalization).

## Package Manager

This project uses **uv** exclusively. Always prefix Python commands with `uv run`.

```bash
# Install dependencies
GIT_LFS_SKIP_SMUDGE=1 uv sync # usually only need to run this once

# Run any script
uv run python viz/convert_npy_to_h5.py ...
uv run streamlit run viz/dashboard/app.py
```

## Key Commands

```bash
# Run the attention visualization dashboard
uv run streamlit run viz/dashboard/app.py

# Convert raw .npy attention maps to HDF5 format
uv run python viz/convert_npy_to_h5.py --src attn/ --dst attn_h5/

# Run tests
uv run pytest src/

# Run a single test file
uv run pytest src/openpi/models/pi0_test.py

# Lint / format
uv run ruff check .
uv run ruff format .

# JAX training
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment

# PyTorch training
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name>

# Serve policy (for online inference mode)
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir=checkpoints/...
```

## Architecture

### Visualization Pipeline

The `viz/` directory is the primary area of active development. It has a three-layer architecture:

**1. Data Layer** — `attn_h5/`
Pre-computed attention maps stored in compressed HDF5 files, organized as:
`attn_h5/{checkpoint_id}/{episode_name}/{frame_idx:05d}.h5`

Each HDF5 file contains:
- `/meta` — prefix_len, seq_len, instruction, token_texts/ids
- `/images` — exterior + wrist RGB images (224×224, gzip compressed)
- `/prefix/layer_{i}/` — text→image attention slices for all layers; full matrix only for key layers `{1, 4, 5, 7, 10}`

**2. Loader Layer** — `viz/dashboard/loader.py`
Streamlit-cached functions for efficient data access. Key constants:
```python
NUM_IMAGE_TOKENS = 256      # 16×16 patches per camera
TOTAL_IMAGE_TOKENS = 512    # ext + wrist
TEXT_START_IDX = 768        # where text tokens begin in sequence
NUM_LAYERS = 18
```

**3. Visualization Layer** — `viz/dashboard/views/`
Modular tabs, each independently customizable:
- `grid_heatmap.py` — text→image attention as 16×16 grids per camera
- `image_heatmap.py` — attention overlaid on RGB images
- `attn_matrix.py` — full sequence-level attention matrix
- `action_view.py` — where action tokens attend
- `comparison.py` / `counterfactual.py` — multi-checkpoint or counterfactual prompt comparisons

### Token Layout (Pi0.5 / DROID)

```
[0:256]        [256:512]     [512:768]    [768:N]       [N:N+8]
ext_camera     wrist_camera  zero_padding text_tokens   action_tokens
   256              256          256         ~100            8
```

This layout is critical for all slicing/indexing in the visualization code.

### Dashboard Modes

- **Offline (HDF5)**: Browse pre-computed attention from `attn_h5/`
- **Online (Inference)**: Live inference using a running policy server, then visualize attention in real time

The dashboard entry point `viz/dashboard/app.py` wires the sidebar controls (checkpoint, episode, frame selectors) to the loader and then to each view.

### Source Model Code

`src/openpi/` contains the upstream model implementations:
- `models/` — JAX implementations (PaliGemma backbone + action expert)
- `models_pytorch/` — PyTorch equivalents
- `policies/` — Pi0, Pi0-FAST, Pi0.5 policy definitions
- `training/` — training configs and data loaders
- `serving/` — policy server for remote inference

### NPY → HDF5 Conversion

Raw attention captures land in `attn/{checkpoint_id}/layers_prefix/attn_map_layer_{i}.npy` (shape: `1, 8, seq, seq`). Run `viz/convert_npy_to_h5.py` to convert these to the dashboard-compatible HDF5 format. The converter also runs PaliGemmaTokenizer to embed token text labels.

## GPU Requirements

- Inference only: >8 GB (RTX 4090 sufficient)
- LoRA fine-tuning: >22.5 GB
- Full fine-tuning: >70 GB (A100/H100)

## Tests

Tests are co-located with source under `src/openpi/`. The `manual` pytest marker gates tests requiring manual setup/hardware. The test suite auto-falls back to CPU if no GPU is detected.
