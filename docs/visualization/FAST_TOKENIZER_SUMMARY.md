# FAST Tokenizer Visualization Summary

## Overview

This document summarizes the FAST (Frequency-space Action Sequence Tokenization) approach from the Physical Intelligence paper: [https://arxiv.org/pdf/2501.09747](https://arxiv.org/pdf/2501.09747)

## What is FAST?

FAST is a novel tokenization method for robot actions that enables autoregressive Vision-Language-Action (VLA) models to learn dexterous, high-frequency control tasks. It addresses the fundamental problem that naive per-timestep binning fails for high-frequency robot control.

### The Problem

**Naive Binning (used in RT-2, OpenVLA):**
- Discretizes each action dimension independently at each timestep
- For 15 timesteps × 8 dimensions = 120 tokens
- **Critical Issue:** Consecutive tokens are highly correlated (correlation ~0.74)
- Autoregressive models can achieve low loss by simply copying previous tokens
- **Result:** Fails completely on dexterous manipulation tasks

### The Solution

**FAST uses DCT compression (like JPEG for images):**

```
Actions [15×8] → DCT Transform → FSQ Quantization → 8 Tokens
```

**Key Steps:**

1. **DCT Transform**: Converts time-series actions to frequency domain
   - Removes temporal correlation
   - Compresses smooth motion into low-frequency components
   
2. **FSQ Quantization**: Finite Scalar Quantization
   - Projects DCT coefficients to lower dimension
   - Quantizes to discrete bins (e.g., 8×6×5 = 240 codes for 2^8 codebook)
   - Converts to single token ID
   
3. **Integration with PaliGemma**:
   - FAST tokens mapped to last 128 slots in PaliGemma vocab
   - Inserted as: `"Action: <token_1> <token_2> ... <token_N> |"`

## Results

### Compression
- **15x reduction**: 120 action values → 8 tokens
- Maintains reconstruction quality (MSE < 0.001)

### Performance
| Task | Naive Binning | FAST | Diffusion |
|------|--------------|------|-----------|
| Table Bussing | ~0% | 83% | 85% |
| T-Shirt Folding | ~0% | 60% | 60% |
| Laundry Folding | ~0% | 40% | 40% |
| DROID (16 tasks) | N/A | 61% | N/A |

### Training Efficiency
- **5x faster** than diffusion VLAs
- Matches diffusion performance on dexterous tasks
- Enables training on 10k hours of robot data

## Implementation in OpenPI

### Code Structure

**`src/openpi/models/tokenizer.py`:**
```python
class FASTTokenizer:
    def tokenize(self, prompt, state, actions):
        # 1. Encode text prefix
        prefix = f"Task: {prompt}, State: {state_str};\n"
        prefix_tokens = paligemma_tokenizer.encode(prefix, add_bos=True)
        
        # 2. Encode actions with FAST (DCT + FSQ)
        action_tokens = fast_tokenizer(actions)
        action_tokens_pg = self._map_to_paligemma_vocab(action_tokens)
        
        # 3. Construct full sequence
        postfix = encode("Action: ") + action_tokens_pg + encode("|", add_eos=True)
        return prefix_tokens + postfix_tokens
```

**`src/openpi/models/utils/fsq_tokenizer.py`:**
- `FsqCodebook`: Implements FSQ quantization
- Supports multiple codebook sizes (2^8, 2^10, 2^12, 2^14, 2^16)
- Bins per dimension: e.g., (8, 6, 5) for 2^8

### Token Sequence Structure

```
[BOS] Task: place duck ... State: 123 45 ... \n Action: <FAST_1> <FAST_2> ... <FAST_8> | [EOS]
 │                                               │                                        │
 └─ Prefix (bidirectional attention)            └─ Suffix (causal attention)            └─ End
```

## Visualizations Created

### 1. `results/fast_tokenizer_explanation.png`
- Complete pipeline overview
- DCT transform visualization
- FSQ quantization process
- Token sequence structure

### 2. `results/tokenization_comparison.png`
- Naive Binning vs FAST vs Diffusion
- Performance comparison table
- When to use each method

### 3. `results/fast_in_action.png`
- Real robot action trajectory (15×8)
- DCT coefficients heatmap
- Compression visualization
- Reconstruction quality

### 4. `results/fast_vs_naive.png`
- Token correlation analysis
- Shows why naive binning fails (correlation 0.74)
- Compression ratio comparison
- Training efficiency insights

## Key Insights

### 1. Decorrelation is Critical
- DCT transforms correlated time-series into uncorrelated frequency components
- Similar to JPEG compression for images
- Enables meaningful gradient signals for autoregressive training

### 2. Compression Enables Efficiency
- 15x reduction in token count
- Faster inference and training
- More context fits in limited sequence length

### 3. Universal Tokenizer (FAST+)
- Trained on 1M robot trajectories
- Works across different robots, action spaces, control frequencies
- Can be used as black-box tokenizer

### 4. Matches Diffusion Performance
- Achieves comparable results on dexterous tasks
- 5x faster training time
- Simpler inference (no iterative denoising)

## When to Use FAST

**Use FAST when:**
- ✅ High-frequency control (20-60Hz)
- ✅ Dexterous manipulation tasks
- ✅ Training efficiency matters
- ✅ Using autoregressive VLAs
- ✅ Need multi-robot generalization

**Use Naive Binning when:**
- ✅ Low-frequency tasks (<5Hz)
- ✅ Simple pick-and-place
- ✅ Quick prototyping

**Use Diffusion when:**
- ✅ 5x more compute budget available
- ✅ Inference latency not critical
- ✅ Highly multimodal action distributions

## Analogy

**Think of FAST like JPEG compression:**
- **Naive Binning** = Storing each pixel value separately (wasteful, correlated)
- **FAST** = Storing frequency coefficients (efficient, decorrelated)
- **Diffusion** = Storing raw image (accurate but slow to process)

## References

1. **Paper**: [FAST: Efficient Action Tokenization for Vision-Language-Action Models](https://arxiv.org/pdf/2501.09747)
2. **Authors**: Physical Intelligence, UC Berkeley, Stanford
3. **Website**: [https://pi.website/research/fast](https://pi.website/research/fast)

## Visualization Scripts

All visualization scripts are in `viz/`:
- `viz/explain_fast_tokenizer.py` - Conceptual explanation
- `viz/visualize_fast_in_action.py` - Practical demonstration
- `viz/attn_map.py` - Integrated tokenizer visualization

Run any script with:
```bash
uv run python viz/<script_name>.py
```

---

**Generated**: December 23, 2025  
**OpenPI Project**: FAST Tokenizer Analysis

