"""
Visualization of FAST (Frequency-space Action Sequence Tokenization)
Based on the paper: https://arxiv.org/pdf/2501.09747

FAST uses DCT (Discrete Cosine Transform) compression to tokenize robot actions,
similar to JPEG compression for images.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
from scipy.fftpack import dct, idct

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 14), facecolor="white")

# ============================================================================
# 1. Problem: Naive Binning vs FAST
# ============================================================================
ax1 = plt.subplot(4, 2, (1, 2))
ax1.set_title(
    "Problem: Naive Per-Timestep Binning Fails for High-Frequency Control", fontsize=16, weight="bold", pad=15
)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 3)
ax1.axis("off")

# Naive binning example
y_pos = 2.2
ax1.text(0.5, y_pos + 0.3, "Naive Binning (e.g., OpenVLA):", fontsize=12, weight="bold", color="darkred")

# Show correlated action sequence
actions_naive = [0.5, 0.52, 0.51, 0.53, 0.52, 0.54, 0.53, 0.55]
tokens_naive = ["bin_128", "bin_133", "bin_130", "bin_135", "bin_133", "bin_138", "bin_135", "bin_140"]

for i, (action, token) in enumerate(zip(actions_naive, tokens_naive)):
    x = i * 1.2
    # Action value
    ax1.text(
        x,
        y_pos,
        f"{action:.2f}",
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.7),
    )
    # Arrow
    ax1.annotate(
        "", xy=(x, y_pos - 0.2), xytext=(x, y_pos - 0.05), arrowprops=dict(arrowstyle="->", lw=1.5, color="black")
    )
    # Token
    ax1.text(x, y_pos - 0.4, token, ha="center", fontsize=8, bbox=dict(boxstyle="round", facecolor="pink", alpha=0.8))

ax1.text(
    5,
    y_pos - 0.7,
    "❌ Highly correlated tokens → Poor for autoregressive prediction",
    ha="center",
    fontsize=11,
    style="italic",
    color="darkred",
)

# FAST approach
y_pos = 0.8
ax1.text(0.5, y_pos + 0.3, "FAST (Frequency-space Tokenization):", fontsize=12, weight="bold", color="darkgreen")

# Show DCT compressed sequence
ax1.text(1, y_pos, "Actions", ha="center", fontsize=10, bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7))
ax1.annotate("", xy=(2, y_pos), xytext=(1.5, y_pos), arrowprops=dict(arrowstyle="->", lw=2, color="black"))
ax1.text(
    2.5, y_pos, "DCT\nCompress", ha="center", fontsize=10, bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7)
)
ax1.annotate("", xy=(3.5, y_pos), xytext=(3, y_pos), arrowprops=dict(arrowstyle="->", lw=2, color="black"))
ax1.text(
    4.5,
    y_pos,
    "Frequency\nCoefficients",
    ha="center",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
)
ax1.annotate("", xy=(5.5, y_pos), xytext=(5, y_pos), arrowprops=dict(arrowstyle="->", lw=2, color="black"))
ax1.text(
    6.5, y_pos, "FSQ\nQuantize", ha="center", fontsize=10, bbox=dict(boxstyle="round", facecolor="orange", alpha=0.7)
)
ax1.annotate("", xy=(7.5, y_pos), xytext=(7, y_pos), arrowprops=dict(arrowstyle="->", lw=2, color="black"))
ax1.text(8.5, y_pos, "Tokens", ha="center", fontsize=10, bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.7))

ax1.text(
    5,
    y_pos - 0.5,
    "✅ Decorrelated tokens → Better for autoregressive prediction",
    ha="center",
    fontsize=11,
    style="italic",
    color="darkgreen",
)

# ============================================================================
# 2. DCT Transform Visualization
# ============================================================================
ax2 = plt.subplot(4, 2, 3)
ax2.set_title("Step 1: DCT Transform (Time → Frequency)", fontsize=14, weight="bold")

# Generate example action trajectory
t = np.linspace(0, 2 * np.pi, 50)
action_trajectory = 0.5 * np.sin(t) + 0.3 * np.sin(3 * t) + 0.2 * np.sin(5 * t)

ax2.plot(t, action_trajectory, "b-", linewidth=2, label="Action trajectory")
ax2.set_xlabel("Time", fontsize=11)
ax2.set_ylabel("Action value", fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_facecolor("lightyellow")

ax3 = plt.subplot(4, 2, 4)
ax3.set_title("DCT Coefficients (Frequency Domain)", fontsize=14, weight="bold")

# Apply DCT
dct_coeffs = dct(action_trajectory, norm="ortho")
ax3.stem(dct_coeffs[:20], basefmt=" ", linefmt="g-", markerfmt="go")
ax3.set_xlabel("Frequency index", fontsize=11)
ax3.set_ylabel("Coefficient magnitude", fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
ax3.set_facecolor("lightgreen")
ax3.text(
    10,
    max(dct_coeffs[:20]) * 0.8,
    "Low frequencies\n(smooth motion)",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
)

# ============================================================================
# 3. FSQ Quantization
# ============================================================================
ax4 = plt.subplot(4, 2, 5)
ax4.set_title("Step 2: FSQ Quantization", fontsize=14, weight="bold")
ax4.axis("off")
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)

fsq_text = """
FSQ (Finite Scalar Quantization):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Project DCT coefficients to lower dimension
   z = tanh(Linear(dct_coeffs))
   
2. Quantize to discrete bins
   For codebook size 2^8:
   • bins_per_dim = (8, 6, 5)
   • Total codes = 8 × 6 × 5 = 240 ≈ 2^8
   
3. Convert to single token ID
   token_id = d₀ + d₁×8 + d₂×8×6
   
Example:
  DCT coeffs: [2.3, -1.5, 0.8, ...]
       ↓
  Quantized: [3, 2, 1]  (digits in each dimension)
       ↓
  Token ID: 3 + 2×8 + 1×48 = 67
"""

ax4.text(
    0.05,
    0.95,
    fsq_text,
    fontsize=10,
    va="top",
    ha="left",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.9),
)

# ============================================================================
# 4. Token Sequence Structure
# ============================================================================
ax5 = plt.subplot(4, 2, 6)
ax5.set_title("Step 3: Token Sequence in PaliGemma", fontsize=14, weight="bold")
ax5.set_xlim(0, 10)
ax5.set_ylim(0, 2)
ax5.axis("off")

tokens = [
    ("Image\nTokens", "lightblue", 1.2),
    ("BOS", "yellow", 0.4),
    ("Prefix\n(Task+State)", "lightgreen", 1.5),
    ("\\n", "orange", 0.3),
    ("Action:", "pink", 0.6),
    ("FAST\nTokens", "lightcoral", 1.2),
    ("|", "orange", 0.2),
    ("EOS", "yellow", 0.4),
]

y_pos = 1.0
x_offset = 0.2
for i, (text, color, width) in enumerate(tokens):
    x_start = x_offset

    box = FancyBboxPatch(
        (x_start, y_pos - 0.25),
        width,
        0.5,
        boxstyle="round,pad=0.03",
        facecolor=color,
        edgecolor="black",
        linewidth=1.5,
    )
    ax5.add_patch(box)

    ax5.text(x_start + width / 2, y_pos, text, ha="center", va="center", fontsize=9, weight="bold")

    x_offset += width + 0.1

    if i < len(tokens) - 1:
        ax5.annotate(
            "",
            xy=(x_offset - 0.05, y_pos),
            xytext=(x_offset - 0.1, y_pos),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="black"),
        )

ax5.text(
    5,
    0.3,
    "FAST tokens are mapped to last 128 tokens in PaliGemma vocab",
    ha="center",
    fontsize=10,
    style="italic",
    color="darkblue",
)

# ============================================================================
# 5. Complete Pipeline
# ============================================================================
ax6 = plt.subplot(4, 2, (7, 8))
ax6.set_title("Complete FAST Pipeline", fontsize=16, weight="bold", pad=15)
ax6.axis("off")
ax6.set_xlim(0, 1)
ax6.set_ylim(0, 1)

pipeline_text = """
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                            FAST Tokenization Pipeline                                     ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝

ENCODING (Actions → Tokens):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: Action trajectory [T × D]
  T = action horizon (e.g., 15 timesteps)
  D = action dimension (e.g., 8 for joint velocities + gripper)

Step 1: DCT Transform
  ├─ Apply 1D DCT along time dimension for each action dimension
  ├─ Compresses temporal correlations into frequency components
  └─ Output: DCT coefficients [T × D]

Step 2: FSQ Quantization
  ├─ Project to lower dimension: z = tanh(Linear(dct_coeffs))
  ├─ Quantize each dimension to discrete bins
  ├─ Convert multi-dimensional bins to single token ID
  └─ Output: Token sequence [N tokens]
     • N depends on compression ratio (e.g., 15×8 actions → 8 tokens)

Step 3: Map to PaliGemma Vocab
  ├─ FAST tokens use last 128 slots in PaliGemma vocab (257024-257151)
  ├─ token_pg = vocab_size - 1 - 128 - token_fast
  └─ Insert into sequence: "Action: <token_1> <token_2> ... <token_N> |"

DECODING (Tokens → Actions):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: Extract FAST tokens from generated sequence
  └─ Parse: "Action: <tokens> |" → extract token IDs

Step 2: Inverse FSQ Quantization
  ├─ Convert token ID back to multi-dimensional bins
  ├─ Dequantize: z = (bins / (bases-1)) × 2 - 1
  └─ Project up: dct_coeffs = Linear(z)

Step 3: Inverse DCT Transform
  ├─ Apply IDCT along time dimension
  └─ Output: Reconstructed actions [T × D]

KEY ADVANTAGES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Compression: 15×8 = 120 values → 8 tokens (15x reduction)
✓ Decorrelation: DCT removes temporal correlation between tokens
✓ Universal: FAST+ trained on 1M trajectories works across robots
✓ Efficiency: 5x faster training than diffusion VLAs
✓ Performance: Matches diffusion VLA performance on dexterous tasks

CODE REFERENCE (from src/openpi/models/tokenizer.py):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FASTTokenizer:
    def tokenize(self, prompt, state, actions):
        # Encode text prefix
        prefix = f"Task: {prompt}, State: {state_str};\\n"
        prefix_tokens = paligemma_tokenizer.encode(prefix, add_bos=True)
        
        # Encode actions with FAST
        action_tokens = fast_tokenizer(actions)  # DCT + FSQ
        action_tokens_pg = self._map_to_paligemma_vocab(action_tokens)
        
        # Construct postfix
        postfix = encode("Action: ") + action_tokens_pg + encode("|", add_eos=True)
        
        return prefix_tokens + postfix_tokens
"""

ax6.text(
    0.02,
    0.98,
    pipeline_text,
    fontsize=9,
    va="top",
    ha="left",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="lavender", alpha=0.95),
)

plt.tight_layout()
plt.savefig("results/fast_tokenizer_explanation.png", dpi=150, bbox_inches="tight")
print("✅ Saved FAST tokenizer explanation to: results/fast_tokenizer_explanation.png")
plt.close()

# ============================================================================
# Create second figure: Comparison with other tokenization methods
# ============================================================================
fig2 = plt.figure(figsize=(18, 10), facecolor="white")

ax = fig2.add_subplot(111)
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("Tokenization Methods Comparison", fontsize=18, weight="bold", pad=20)

comparison_text = """
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                     Robot Action Tokenization Methods Comparison                          ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝

1. NAIVE BINNING (RT-2, OpenVLA)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Method: Per-dimension, per-timestep binning
  • Discretize each action dimension into N bins (e.g., 256)
  • Each timestep gets D tokens (one per dimension)
  • For 15 timesteps × 8 dims = 120 tokens

Pros:
  ✓ Simple to implement
  ✓ No learned components

Cons:
  ✗ Highly correlated consecutive tokens
  ✗ Fails for high-frequency control (>10Hz)
  ✗ Large number of tokens for action chunks
  ✗ Poor for dexterous manipulation

Example: action = [0.52, -0.31, 0.15, ...]
         → tokens = [bin_133, bin_89, bin_165, ...]


2. FAST (Frequency-space Action Sequence Tokenization)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Method: DCT compression + FSQ quantization
  • Apply DCT to compress temporal dimension
  • Use FSQ to quantize frequency coefficients
  • 15×8 actions → 8 tokens (15x compression)

Pros:
  ✓ Decorrelates consecutive tokens
  ✓ Works for high-frequency control (20-60Hz)
  ✓ Efficient compression (15x reduction)
  ✓ Universal tokenizer (FAST+) works across robots
  ✓ 5x faster training than diffusion

Cons:
  ✗ Requires learned FSQ quantizer
  ✗ Slight reconstruction error

Example: actions [15×8] → DCT → FSQ → [8 tokens]


3. DIFFUSION (π₀ baseline)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Method: Continuous diffusion process
  • No tokenization - predict continuous actions directly
  • Use denoising diffusion to generate action sequences

Pros:
  ✓ No discretization error
  ✓ Works well for dexterous tasks
  ✓ Handles multimodal action distributions

Cons:
  ✗ 5x slower training than FAST
  ✗ Slower inference (requires multiple denoising steps)
  ✗ Cannot leverage pre-trained language models directly


PERFORMANCE COMPARISON (from paper):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Task                  │ Naive Binning │  FAST   │ Diffusion
──────────────────────┼───────────────┼─────────┼──────────
Table Bussing         │     ~0%       │  83%    │   85%
T-Shirt Folding       │     ~0%       │  60%    │   60%
Laundry Folding       │     ~0%       │  40%    │   40%
DROID (16 tasks)      │     N/A       │  61%    │   N/A

Training Time         │     1x        │   1x    │   5x
Inference Speed       │    Fast       │  Fast   │  Slow


WHEN TO USE EACH METHOD:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Naive Binning:
  • Low-frequency tasks (<5Hz)
  • Simple pick-and-place
  • When training speed is not critical

FAST:
  • High-frequency control (20-60Hz)
  • Dexterous manipulation
  • When training efficiency matters
  • When using autoregressive VLAs
  • Multi-robot generalization (use FAST+)

Diffusion:
  • When you have 5x more compute budget
  • When inference latency is not critical
  • Highly multimodal action distributions
"""

ax.text(
    0.02,
    0.98,
    comparison_text,
    fontsize=10,
    va="top",
    ha="left",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95),
)

plt.tight_layout()
plt.savefig("results/tokenization_comparison.png", dpi=150, bbox_inches="tight")
print("✅ Saved tokenization comparison to: results/tokenization_comparison.png")
plt.close()

print("\n📚 Summary:")
print("   • FAST uses DCT (like JPEG) to compress robot actions")
print("   • FSQ quantization converts frequency coefficients to discrete tokens")
print("   • 15x compression: 120 action values → 8 tokens")
print("   • Enables autoregressive VLAs for dexterous, high-frequency control")
print("   • 5x faster training than diffusion while matching performance")
print("\n📖 Paper: https://arxiv.org/pdf/2501.09747")
