"""
Side-by-side comparison of PaliGemma and FAST tokenizers
Shows how the same robot action sequence is tokenized differently
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Create comprehensive comparison figure
fig = plt.figure(figsize=(20, 12), facecolor="white")

# ============================================================================
# Title
# ============================================================================
fig.suptitle("PaliGemma Tokenizer vs FAST Tokenizer Comparison", fontsize=18, weight="bold", y=0.98)

# ============================================================================
# 1. Input: Robot Action Sequence
# ============================================================================
ax1 = plt.subplot(4, 1, 1)
ax1.set_title("Input: Robot Action Sequence (15 timesteps × 8 dimensions)", fontsize=14, weight="bold", pad=10)

# Simulate action trajectory
np.random.seed(42)
action_horizon = 15
action_dim = 8
t = np.linspace(0, 2 * np.pi, action_horizon)
actions = np.zeros((action_horizon, action_dim))

for d in range(action_dim):
    freq = np.random.uniform(0.5, 3.0)
    amp = np.random.uniform(0.3, 0.6)
    actions[:, d] = amp * np.sin(freq * t) + np.random.normal(0, 0.05, action_horizon)

actions = np.clip(actions, -1, 1)

# Plot all dimensions
for d in range(action_dim):
    label = f"Joint {d + 1}" if d < 7 else "Gripper"
    ax1.plot(range(action_horizon), actions[:, d], "o-", linewidth=2, markersize=5, label=label, alpha=0.7)

ax1.set_xlabel("Timestep", fontsize=11)
ax1.set_ylabel("Action Value", fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.legend(ncol=8, fontsize=9, loc="upper right")
ax1.set_ylim(-1.2, 1.2)
ax1.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)

total_values = action_horizon * action_dim
ax1.text(
    7.5,
    1.1,
    f"Total values: {total_values} (15 timesteps × 8 dims)",
    ha="center",
    fontsize=11,
    weight="bold",
    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
)

# ============================================================================
# 2. PaliGemma Tokenizer (Naive Binning)
# ============================================================================
ax2 = plt.subplot(4, 1, 2)
ax2.set_title("PaliGemma Tokenizer: Per-Dimension, Per-Timestep Binning", fontsize=14, weight="bold", pad=10)
ax2.axis("off")
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

paligemma_text = """
PALIGEMMA TOKENIZER (Standard Approach):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Method: Discretize each action dimension independently at each timestep

Step 1: Discretize state (256 bins)
  state = [0.52, -0.31, 0.15, ...]
  discretized = [133, 89, 165, ...]  (bin indices)
  state_str = "133 89 165 ..."

Step 2: Construct prefix
  prefix = "Task: place duck, State: 133 89 165 ...;\\n"
  prefix_tokens = encode(prefix, add_bos=True)
  → [BOS, Task, :, place, duck, ,, State, :, 133, 89, 165, ..., ;, \\n]

Step 3: No action tokenization yet (just text)
  In PaliGemma, actions would be predicted as text tokens
  Each action value → separate token
  
Result:
  • Total tokens: ~50-100 for prefix + 120 for actions = 170+ tokens
  • Problem: Consecutive action tokens highly correlated
  • Correlation: ~0.74 (very high!)
  • Performance: Fails on high-frequency dexterous tasks

Token Sequence:
  [BOS] Task: place duck , State: 133 89 165 ... ; \\n Action: <action_tokens> | [EOS]
   └────────────────────────────────────────────┘  └──────────────────────┘
              Prefix (bidirectional)                    Suffix (causal)
"""

ax2.text(
    0.02,
    0.98,
    paligemma_text,
    fontsize=10,
    va="top",
    ha="left",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.9),
)

# ============================================================================
# 3. FAST Tokenizer
# ============================================================================
ax3 = plt.subplot(4, 1, 3)
ax3.set_title("FAST Tokenizer: DCT Compression + FSQ Quantization", fontsize=14, weight="bold", pad=10)
ax3.axis("off")
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)

fast_text = """
FAST TOKENIZER (Physical Intelligence Approach):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Method: Compress actions using DCT, then quantize with FSQ

Step 1: Same prefix as PaliGemma
  prefix = "Task: place duck, State: 133 89 165 ...;\\n"
  prefix_tokens = encode(prefix, add_bos=True)

Step 2: DCT Transform on actions
  actions [15×8] → DCT along time dimension → dct_coeffs [15×8]
  Removes temporal correlation, compresses to frequency domain

Step 3: FSQ Quantization
  dct_coeffs → Project down → Quantize to bins → Token IDs
  15×8 = 120 values → 8 tokens (15x compression!)
  
  Example: [67, 142, 89, 201, 34, 156, 78, 193]

Step 4: Map to PaliGemma vocab
  FAST tokens use last 128 slots: vocab[257024:257151]
  token_pg = vocab_size - 1 - 128 - token_fast

Step 5: Construct postfix
  postfix = encode("Action: ") + [67, 142, 89, 201, 34, 156, 78, 193] + encode("|", add_eos=True)

Result:
  • Total tokens: ~50 for prefix + 8 for actions = 58 tokens (3x reduction!)
  • Tokens are decorrelated by DCT
  • Correlation: Low (each token carries unique information)
  • Performance: Works for high-frequency dexterous tasks!

Token Sequence:
  [BOS] Task: place duck , State: 133 89 ... ; \\n Action: 67 142 89 201 34 156 78 193 | [EOS]
   └──────────────────────────────────────────┘  └─────────────────────────────────┘
              Prefix (bidirectional)                  Suffix (causal) - FAST tokens
"""

ax3.text(
    0.02,
    0.98,
    fast_text,
    fontsize=10,
    va="top",
    ha="left",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.9),
)

# ============================================================================
# 4. Visual Comparison
# ============================================================================
ax4 = plt.subplot(4, 1, 4)
ax4.set_title("Visual Token Sequence Comparison", fontsize=14, weight="bold", pad=10)
ax4.set_xlim(0, 20)
ax4.set_ylim(0, 4)
ax4.axis("off")

# PaliGemma sequence (simplified)
y_pos = 3.0
ax4.text(0.2, y_pos + 0.4, "PaliGemma:", fontsize=12, weight="bold", color="darkred")

pali_tokens = [
    ("BOS", "yellow", 0.5),
    ("Task:", "lightgreen", 0.6),
    ("place", "lightgreen", 0.6),
    ("duck", "lightgreen", 0.6),
    ("...", "lightgreen", 0.4),
    ("State:", "lightblue", 0.6),
    ("133", "lightblue", 0.4),
    ("89", "lightblue", 0.3),
    ("...", "lightblue", 0.4),
    ("\\n", "orange", 0.3),
    ("Action:", "pink", 0.7),
    ("...", "lightcoral", 0.4),
    ("many", "lightcoral", 0.5),
    ("tokens", "lightcoral", 0.6),
    ("...", "lightcoral", 0.4),
]

x = 0.2
for text, color, width in pali_tokens:
    box = FancyBboxPatch(
        (x, y_pos - 0.2), width, 0.4, boxstyle="round,pad=0.02", facecolor=color, edgecolor="black", linewidth=1
    )
    ax4.add_patch(box)
    ax4.text(x + width / 2, y_pos, text, ha="center", va="center", fontsize=8)
    x += width + 0.05

ax4.text(10, y_pos - 0.6, "~170+ tokens", ha="center", fontsize=10, style="italic", color="darkred", weight="bold")

# FAST sequence
y_pos = 1.5
ax4.text(0.2, y_pos + 0.4, "FAST:", fontsize=12, weight="bold", color="darkgreen")

fast_tokens = [
    ("BOS", "yellow", 0.5),
    ("Task:", "lightgreen", 0.6),
    ("place", "lightgreen", 0.6),
    ("duck", "lightgreen", 0.6),
    ("...", "lightgreen", 0.4),
    ("State:", "lightblue", 0.6),
    ("133", "lightblue", 0.4),
    ("89", "lightblue", 0.3),
    ("...", "lightblue", 0.4),
    ("\\n", "orange", 0.3),
    ("Action:", "pink", 0.7),
    ("67", "lightcoral", 0.4),
    ("142", "lightcoral", 0.4),
    ("89", "lightcoral", 0.4),
    ("201", "lightcoral", 0.4),
    ("34", "lightcoral", 0.4),
    ("156", "lightcoral", 0.4),
    ("78", "lightcoral", 0.4),
    ("193", "lightcoral", 0.4),
    ("|", "orange", 0.2),
]

x = 0.2
for text, color, width in fast_tokens:
    box = FancyBboxPatch(
        (x, y_pos - 0.2), width, 0.4, boxstyle="round,pad=0.02", facecolor=color, edgecolor="black", linewidth=1.5
    )
    ax4.add_patch(box)
    ax4.text(x + width / 2, y_pos, text, ha="center", va="center", fontsize=8, weight="bold")
    x += width + 0.05

ax4.text(
    10,
    y_pos - 0.6,
    "~58 tokens (3x fewer!)",
    ha="center",
    fontsize=10,
    style="italic",
    color="darkgreen",
    weight="bold",
)

# Add comparison table
y_pos = 0.3
comparison = [
    ("Metric", "PaliGemma", "FAST"),
    ("Total Tokens", "~170", "~58"),
    ("Action Tokens", "120", "8"),
    ("Compression", "1x", "15x"),
    ("Correlation", "0.74", "Low"),
    ("Dexterous Tasks", "❌ Fails", "✅ Works"),
    ("Training Speed", "1x", "5x faster"),
]

# Draw table
table_x = 6
table_y = 0.5
col_widths = [2.5, 2, 2]
row_height = 0.25

for i, (metric, pali, fast) in enumerate(comparison):
    y = table_y - i * row_height

    # Header row
    if i == 0:
        bg_color = "lightgray"
        text_weight = "bold"
    else:
        bg_color = "white" if i % 2 == 0 else "whitesmoke"
        text_weight = "normal"

    # Draw cells
    for j, (text, width) in enumerate(zip([metric, pali, fast], col_widths)):
        x = table_x + sum(col_widths[:j])
        rect = plt.Rectangle(
            (x, y - row_height / 2), width, row_height, facecolor=bg_color, edgecolor="black", linewidth=1
        )
        ax4.add_patch(rect)
        ax4.text(x + width / 2, y, text, ha="center", va="center", fontsize=9, weight=text_weight)

plt.tight_layout()
plt.savefig("results/paligemma_vs_fast.png", dpi=150, bbox_inches="tight")
print("✅ Saved comparison to: results/paligemma_vs_fast.png")
plt.close()

# ============================================================================
# Create a simplified infographic
# ============================================================================
fig2 = plt.figure(figsize=(16, 10), facecolor="white")

ax = fig2.add_subplot(111)
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("FAST: The Key Innovation in π₀", fontsize=20, weight="bold", pad=20)

infographic = """
╔═══════════════════════════════════════════════════════════════════════════════════════════════╗
║                        Why FAST Matters for Robot Learning                                    ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════╝

THE CHALLENGE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Autoregressive VLAs (like GPT for robots) need to predict discrete tokens.
But robot actions are continuous, high-frequency time-series!

Problem with naive tokenization:
  • Action at t=0: 0.50 → token 128
  • Action at t=1: 0.52 → token 133
  • Action at t=2: 0.51 → token 130
  
  ❌ Tokens are highly correlated (0.74)!
  ❌ Model learns "just copy previous token"
  ❌ No meaningful learning happens
  ❌ Fails completely on dexterous tasks


THE INSIGHT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Think about JPEG compression:
  • Don't store each pixel → highly correlated!
  • Store frequency coefficients → decorrelated!
  • Much more efficient representation

FAST applies the same idea to robot actions:
  • Don't tokenize each timestep → highly correlated!
  • Tokenize frequency coefficients (DCT) → decorrelated!
  • Each token carries unique information


THE SOLUTION: FAST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: DCT Transform
  Actions [15 timesteps × 8 dims] → Frequency domain
  Smooth motion → Low frequencies (most energy)
  Fine details → High frequencies (less energy)

Step 2: FSQ Quantization
  Frequency coefficients → Discrete tokens
  120 values → 8 tokens (15x compression!)

Step 3: Integrate with VLA
  Tokens inserted into PaliGemma sequence
  Trained with standard next-token prediction


THE RESULTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Performance:
  ✅ 83% success on table bussing (naive: ~0%)
  ✅ 60% success on t-shirt folding (naive: ~0%)
  ✅ 40% success on laundry folding (naive: ~0%)
  ✅ 61% average on DROID 16-task benchmark

Efficiency:
  ✅ 5x faster training than diffusion VLAs
  ✅ Matches diffusion performance on dexterous tasks
  ✅ Scales to 10k hours of robot data

Generalization:
  ✅ FAST+ universal tokenizer works across robots
  ✅ Trained on 1M trajectories from diverse embodiments
  ✅ Zero-shot transfer to new environments


WHY THIS IS IMPORTANT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before FAST:
  • Autoregressive VLAs limited to simple, low-frequency tasks
  • Dexterous manipulation required diffusion models (slow)
  • No universal tokenizer across robots

After FAST:
  • Autoregressive VLAs work for dexterous, high-frequency control
  • 5x faster training enables scaling to massive datasets
  • Universal tokenizer (FAST+) enables multi-robot learning

This is π₀'s secret sauce! 🚀


ANALOGY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Imagine teaching a language model to predict a movie:

❌ Bad: Predict each pixel value at each frame
   → Millions of highly correlated tokens
   → Model learns "copy previous frame"
   → Doesn't understand motion

✅ Good: Predict compressed video codes (like H.264)
   → Thousands of decorrelated tokens
   → Each token represents meaningful motion
   → Model learns video structure

FAST is the "H.264 for robot actions"!


IMPLEMENTATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

See: src/openpi/models/tokenizer.py (FASTTokenizer class)
     src/openpi/models/utils/fsq_tokenizer.py (FSQ implementation)

Paper: https://arxiv.org/pdf/2501.09747
Website: https://pi.website/research/fast
"""

ax.text(
    0.02,
    0.98,
    infographic,
    fontsize=9.5,
    va="top",
    ha="left",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95),
)

plt.tight_layout()
plt.savefig("results/fast_infographic.png", dpi=150, bbox_inches="tight")
print("✅ Saved infographic to: results/fast_infographic.png")
plt.close()

print("\n" + "=" * 70)
print("📊 Tokenizer Comparison Complete!")
print("=" * 70)
print("\n📁 Generated files:")
print("   • results/paligemma_vs_fast.png - Side-by-side comparison")
print("   • results/fast_infographic.png - Key insights infographic")
print("\n💡 Key Takeaway:")
print("   FAST enables autoregressive VLAs for dexterous manipulation")
print("   by decorrelating action tokens through DCT compression!")
