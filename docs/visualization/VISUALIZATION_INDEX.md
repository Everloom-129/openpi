# FAST Tokenizer Visualization Index

## Overview

This document provides an index of all FAST tokenizer visualizations created for the OpenPI project. These visualizations explain how Physical Intelligence's FAST (Frequency-space Action Sequence Tokenization) works and why it's critical for enabling autoregressive VLAs to perform dexterous manipulation.

---

## 📊 Visualization Files

### 1. **fast_tokenizer_explanation.png** (653K)
**Purpose**: Comprehensive technical explanation of FAST  
**Contents**:
- Problem statement: Why naive binning fails
- DCT transform visualization
- FSQ quantization process
- Complete pipeline diagram
- Code reference from `tokenizer.py`

**Best for**: Understanding the technical details of FAST

---

### 2. **fast_in_action.png** (694K)
**Purpose**: Practical demonstration with real robot data  
**Contents**:
- Input: Robot action trajectory (15×8)
- DCT coefficients heatmap
- Compression visualization (energy per frequency)
- FSQ quantization simulation
- Token sequence in PaliGemma
- Reconstructed actions with MSE

**Best for**: Seeing FAST work on actual robot data

---

### 3. **fast_vs_naive.png** (388K)
**Purpose**: Direct comparison showing why FAST is better  
**Contents**:
- Naive binning visualization
- Token correlation analysis (shows 0.74 correlation!)
- FAST compressed representation
- Side-by-side statistics comparison

**Best for**: Understanding the key advantage of FAST

---

### 4. **tokenization_comparison.png** (462K)
**Purpose**: Comparison of all tokenization methods  
**Contents**:
- Naive Binning (RT-2, OpenVLA)
- FAST (Physical Intelligence)
- Diffusion (π₀ baseline)
- Performance comparison table
- When to use each method

**Best for**: Choosing the right tokenization approach

---

### 5. **paligemma_vs_fast.png** (638K)
**Purpose**: Side-by-side comparison of PaliGemma and FAST  
**Contents**:
- Input action sequence
- PaliGemma tokenization process
- FAST tokenization process
- Visual token sequence comparison
- Comparison table

**Best for**: Understanding the integration with PaliGemma

---

### 6. **fast_infographic.png** (497K)
**Purpose**: High-level explanation of why FAST matters  
**Contents**:
- The challenge (correlated tokens)
- The insight (JPEG-like compression)
- The solution (DCT + FSQ)
- Results and impact
- Analogy: "H.264 for robot actions"

**Best for**: Quick overview and motivation

---

### 7. **paligemma_special_tokens.png** (384K)
**Purpose**: Explanation of PaliGemma special tokens  
**Contents**:
- BOS (Begin of Sequence)
- SEP (Separator: `\n`)
- EOS (End of Sequence)
- PAD (Padding)
- Visual token flow diagram

**Best for**: Understanding PaliGemma tokenizer basics

---

### 8. **paligemma_token_ids.png** (259K)
**Purpose**: Actual token IDs from PaliGemma  
**Contents**:
- Sample prompt tokenization
- Token ID to piece mapping
- Visual representation

**Best for**: Debugging tokenization issues

---

## 🎯 Quick Navigation

### I want to understand...

**...what FAST is:**
→ Start with `fast_infographic.png`

**...how FAST works technically:**
→ Read `fast_tokenizer_explanation.png`

**...why FAST is better than naive binning:**
→ See `fast_vs_naive.png`

**...how FAST integrates with PaliGemma:**
→ Check `paligemma_vs_fast.png`

**...FAST with real robot data:**
→ View `fast_in_action.png`

**...all tokenization methods:**
→ Compare in `tokenization_comparison.png`

---

## 📝 Key Insights Summary

### The Problem
- Naive per-timestep binning creates highly correlated tokens (correlation ~0.74)
- Autoregressive models learn trivial "copy previous token" behavior
- Fails completely on high-frequency dexterous tasks

### The Solution
- **DCT Transform**: Converts time-series to frequency domain (like JPEG)
- **FSQ Quantization**: Discrete quantization of frequency coefficients
- **Result**: 15x compression (120 values → 8 tokens) with decorrelated tokens

### The Impact
- ✅ 83% success on table bussing (naive: ~0%)
- ✅ 60% success on t-shirt folding (naive: ~0%)
- ✅ 5x faster training than diffusion VLAs
- ✅ Matches diffusion performance on dexterous tasks

---

## 🔧 Generating Visualizations

All visualization scripts are in `viz/`:

```bash
# Conceptual explanation
uv run python viz/explain_fast_tokenizer.py

# Practical demonstration
uv run python viz/visualize_fast_in_action.py

# Tokenizer comparison
uv run python viz/compare_tokenizers.py

# PaliGemma special tokens
uv run python viz/explain_paligemma_tokens.py
```

---

## 📚 Related Files

### Code
- `src/openpi/models/tokenizer.py` - FASTTokenizer implementation
- `src/openpi/models/utils/fsq_tokenizer.py` - FSQ quantization
- `viz/attn_map.py` - Integrated tokenizer visualization

### Documentation
- `results/FAST_TOKENIZER_SUMMARY.md` - Comprehensive summary
- `results/VISUALIZATION_INDEX.md` - This file

---

## 🔗 References

1. **Paper**: [FAST: Efficient Action Tokenization for Vision-Language-Action Models](https://arxiv.org/pdf/2501.09747)
2. **Authors**: Physical Intelligence, UC Berkeley, Stanford
3. **Website**: [https://pi.website/research/fast](https://pi.website/research/fast)

---

## 💡 Analogy

**FAST is to robot actions what JPEG is to images:**

| Domain | Raw Data | Naive Approach | Compressed Approach |
|--------|----------|----------------|---------------------|
| Images | Pixels | Store each pixel | JPEG (DCT + quantization) |
| Robot Actions | Timesteps | Bin each timestep | FAST (DCT + FSQ) |
| Result | - | Correlated, inefficient | Decorrelated, efficient |

---

**Generated**: December 23, 2025  
**OpenPI Project**: FAST Tokenizer Visualization Suite

