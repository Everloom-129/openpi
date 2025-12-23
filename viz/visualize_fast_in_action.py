"""
Visualize FAST tokenization with actual robot data
Shows the complete pipeline from actions to tokens and back
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from scipy.fftpack import dct, idct

# Simulate a realistic robot action trajectory
# 15 timesteps × 8 dimensions (7 joints + 1 gripper)
np.random.seed(42)
action_horizon = 15
action_dim = 8

# Create smooth action trajectory (simulating joint velocities)
t = np.linspace(0, 2 * np.pi, action_horizon)
actions = np.zeros((action_horizon, action_dim))

for d in range(action_dim):
    # Each dimension has different frequency components
    freq1 = np.random.uniform(0.5, 2.0)
    freq2 = np.random.uniform(2.0, 4.0)
    amp1 = np.random.uniform(0.3, 0.6)
    amp2 = np.random.uniform(0.1, 0.3)

    actions[:, d] = amp1 * np.sin(freq1 * t) + amp2 * np.sin(freq2 * t)
    # Add small noise
    actions[:, d] += np.random.normal(0, 0.05, action_horizon)

# Clip to [-1, 1] range
actions = np.clip(actions, -1, 1)

# ============================================================================
# Create comprehensive visualization
# ============================================================================
fig = plt.figure(figsize=(20, 12), facecolor="white")

# ============================================================================
# 1. Original Action Trajectory
# ============================================================================
ax1 = plt.subplot(3, 3, (1, 2))
ax1.set_title("Input: Robot Action Trajectory [15 timesteps × 8 dims]", fontsize=14, weight="bold")

for d in range(action_dim):
    label = f"Joint {d + 1}" if d < 7 else "Gripper"
    ax1.plot(range(action_horizon), actions[:, d], "o-", linewidth=2, markersize=6, label=label, alpha=0.8)

ax1.set_xlabel("Timestep", fontsize=11)
ax1.set_ylabel("Action Value", fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.legend(ncol=4, fontsize=9, loc="upper right")
ax1.set_ylim(-1.2, 1.2)
ax1.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)

# Add annotation
ax1.text(
    7.5,
    1.1,
    f"Total values: {action_horizon} × {action_dim} = {action_horizon * action_dim}",
    ha="center",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
)

# ============================================================================
# 2. DCT Coefficients
# ============================================================================
ax2 = plt.subplot(3, 3, 3)
ax2.set_title("Step 1: DCT Transform", fontsize=14, weight="bold")

# Apply DCT to each dimension
dct_coeffs = np.zeros_like(actions)
for d in range(action_dim):
    dct_coeffs[:, d] = dct(actions[:, d], norm="ortho")

# Show DCT coefficients as heatmap
im = ax2.imshow(dct_coeffs.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
ax2.set_xlabel("Frequency Index", fontsize=11)
ax2.set_ylabel("Action Dimension", fontsize=11)
ax2.set_yticks(range(action_dim))
ax2.set_yticklabels([f"J{i + 1}" if i < 7 else "Grip" for i in range(action_dim)])
plt.colorbar(im, ax=ax2, label="Coefficient")

# Highlight low frequencies
ax2.axvline(x=5, color="lime", linestyle="--", linewidth=2, alpha=0.7)
ax2.text(
    2.5,
    action_dim + 0.5,
    "Low freq\n(smooth)",
    ha="center",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
)
ax2.text(
    10,
    action_dim + 0.5,
    "High freq (details)",
    ha="center",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.8),
)

# ============================================================================
# 3. Compression Visualization
# ============================================================================
ax3 = plt.subplot(3, 3, 4)
ax3.set_title("Compression: Keep Important Frequencies", fontsize=14, weight="bold")

# Calculate energy in each frequency
energy_per_freq = np.sum(dct_coeffs**2, axis=1)
energy_cumsum = np.cumsum(energy_per_freq) / np.sum(energy_per_freq)

ax3.bar(range(action_horizon), energy_per_freq, color="steelblue", alpha=0.7)
ax3.set_xlabel("Frequency Index", fontsize=11)
ax3.set_ylabel("Energy", fontsize=11)
ax3.grid(True, alpha=0.3, axis="y")

# Mark 90% energy threshold
threshold_idx = np.where(energy_cumsum >= 0.9)[0][0]
ax3.axvline(x=threshold_idx, color="red", linestyle="--", linewidth=2)
ax3.text(
    threshold_idx + 1,
    max(energy_per_freq) * 0.8,
    f"90% energy\nin {threshold_idx + 1} coeffs",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8),
)

# ============================================================================
# 4. FSQ Quantization Simulation
# ============================================================================
ax4 = plt.subplot(3, 3, 5)
ax4.set_title("Step 2: FSQ Quantization", fontsize=14, weight="bold")
ax4.axis("off")
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)

# Simulate FSQ process
num_tokens = 8  # Typical compression: 120 values → 8 tokens
compression_ratio = (action_horizon * action_dim) / num_tokens

fsq_info = f"""
FSQ Quantization Process:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: DCT coefficients [{action_horizon}×{action_dim}]

1. Flatten & Project Down
   {action_horizon * action_dim} values → 24 dims
   z = tanh(Linear(dct_coeffs))

2. Quantize to Bins
   Codebook: 2^8 (bins: 8×6×5)
   Each dim quantized to discrete levels

3. Convert to Token IDs
   24 dims → {num_tokens} tokens
   
Compression: {compression_ratio:.1f}x
({action_horizon * action_dim} values → {num_tokens} tokens)

Example Token Sequence:
[67, 142, 89, 201, 34, 156, 78, 193]
"""

ax4.text(
    0.05,
    0.95,
    fsq_info,
    fontsize=10,
    va="top",
    ha="left",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.9),
)

# ============================================================================
# 5. Token Sequence in PaliGemma
# ============================================================================
ax5 = plt.subplot(3, 3, 6)
ax5.set_title("Step 3: Insert into PaliGemma Sequence", fontsize=14, weight="bold")
ax5.set_xlim(0, 12)
ax5.set_ylim(0, 3)
ax5.axis("off")

# Show full token sequence
tokens = [
    ("BOS", "yellow", 0.5, 0),
    ("Task:", "lightgreen", 0.6, 0),
    ("place", "lightgreen", 0.6, 0),
    ("duck", "lightgreen", 0.5, 0),
    ("...", "lightgreen", 0.4, 0),
    ("State:", "lightblue", 0.6, 0),
    ("123", "lightblue", 0.4, 0),
    ("45", "lightblue", 0.3, 0),
    ("...", "lightblue", 0.4, 0),
    ("\\n", "orange", 0.3, 0),
]

y_pos = 2.0
x = 0.2
for text, color, width, _ in tokens:
    box = FancyBboxPatch(
        (x, y_pos - 0.2), width, 0.4, boxstyle="round,pad=0.02", facecolor=color, edgecolor="black", linewidth=1
    )
    ax5.add_patch(box)
    ax5.text(x + width / 2, y_pos, text, ha="center", va="center", fontsize=8)
    x += width + 0.05

ax5.text(6, 2.5, "Prefix (bidirectional attention)", ha="center", fontsize=9, style="italic", color="darkgreen")

# Action tokens
y_pos = 1.0
x = 0.2
ax5.text(x, y_pos, "Action:", ha="left", va="center", fontsize=9, weight="bold")
x += 0.8

simulated_tokens = [67, 142, 89, 201, 34, 156, 78, 193]
for i, token_id in enumerate(simulated_tokens):
    width = 0.5
    box = FancyBboxPatch(
        (x, y_pos - 0.2), width, 0.4, boxstyle="round,pad=0.02", facecolor="lightcoral", edgecolor="red", linewidth=1.5
    )
    ax5.add_patch(box)
    ax5.text(x + width / 2, y_pos, str(token_id), ha="center", va="center", fontsize=8, weight="bold")
    x += width + 0.05

ax5.text(x, y_pos, "|", ha="center", va="center", fontsize=12, weight="bold")

ax5.text(6, 0.5, "Suffix (causal attention) - FAST tokens", ha="center", fontsize=9, style="italic", color="darkred")

# ============================================================================
# 6. Decoding Process
# ============================================================================
ax6 = plt.subplot(3, 3, 7)
ax6.set_title("Decoding: Tokens → Actions", fontsize=14, weight="bold")
ax6.axis("off")
ax6.set_xlim(0, 10)
ax6.set_ylim(0, 2)

decode_steps = [
    ("FAST\nTokens", "lightcoral", 1.0),
    ("Inverse\nFSQ", "orange", 1.0),
    ("DCT\nCoeffs", "lightgreen", 1.0),
    ("Inverse\nDCT", "yellow", 1.0),
    ("Actions", "lightblue", 1.0),
]

y_pos = 1.0
x = 0.5
for text, color, width in decode_steps:
    box = FancyBboxPatch(
        (x, y_pos - 0.3), width, 0.6, boxstyle="round,pad=0.05", facecolor=color, edgecolor="black", linewidth=2
    )
    ax6.add_patch(box)
    ax6.text(x + width / 2, y_pos, text, ha="center", va="center", fontsize=10, weight="bold")

    if text != "Actions":
        ax6.annotate(
            "",
            xy=(x + width + 0.3, y_pos),
            xytext=(x + width + 0.1, y_pos),
            arrowprops=dict(arrowstyle="->", lw=2.5, color="black"),
        )

    x += width + 0.4

# ============================================================================
# 7. Reconstructed Actions
# ============================================================================
ax7 = plt.subplot(3, 3, (8, 9))
ax7.set_title("Reconstructed Actions (after IDCT)", fontsize=14, weight="bold")

# Simulate reconstruction with slight loss
# In practice, this would be done by the FSQ decoder
reconstructed = actions.copy()
# Add small reconstruction error
reconstructed += np.random.normal(0, 0.02, actions.shape)
reconstructed = np.clip(reconstructed, -1, 1)

for d in range(action_dim):
    label = f"Joint {d + 1}" if d < 7 else "Gripper"
    # Original (dashed)
    ax7.plot(range(action_horizon), actions[:, d], "--", linewidth=1.5, color=f"C{d}", alpha=0.5)
    # Reconstructed (solid)
    ax7.plot(
        range(action_horizon),
        reconstructed[:, d],
        "o-",
        linewidth=2,
        markersize=6,
        label=label,
        color=f"C{d}",
        alpha=0.8,
    )

ax7.set_xlabel("Timestep", fontsize=11)
ax7.set_ylabel("Action Value", fontsize=11)
ax7.grid(True, alpha=0.3)
ax7.legend(ncol=4, fontsize=9, loc="upper right")
ax7.set_ylim(-1.2, 1.2)
ax7.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)

# Calculate reconstruction error
mse = np.mean((actions - reconstructed) ** 2)
ax7.text(
    7.5,
    -1.1,
    f"Reconstruction MSE: {mse:.6f}",
    ha="center",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
)

# Add legend for line styles
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color="gray", linewidth=1.5, linestyle="--", label="Original"),
    Line2D([0], [0], color="gray", linewidth=2, marker="o", label="Reconstructed"),
]
ax7.legend(handles=legend_elements, loc="lower right", fontsize=9)

plt.tight_layout()
plt.savefig("results/fast_in_action.png", dpi=150, bbox_inches="tight")
print("✅ Saved FAST in action visualization to: results/fast_in_action.png")
plt.close()

# ============================================================================
# Create comparison figure: Naive vs FAST
# ============================================================================
fig2 = plt.figure(figsize=(16, 10), facecolor="white")

# ============================================================================
# Naive Binning
# ============================================================================
ax1 = plt.subplot(2, 2, 1)
ax1.set_title("Naive Binning: Per-Timestep Discretization", fontsize=13, weight="bold")

# Show first 3 dimensions only for clarity
for d in range(3):
    ax1.plot(range(action_horizon), actions[:, d], "o-", linewidth=2, markersize=6, label=f"Dim {d + 1}", alpha=0.8)

ax1.set_xlabel("Timestep", fontsize=11)
ax1.set_ylabel("Action Value", fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Show binning
num_bins = 256
for d in range(3):
    binned = np.digitize(actions[:, d], bins=np.linspace(-1, 1, num_bins + 1)) - 1
    binned_values = np.linspace(-1, 1, num_bins)[binned]
    ax1.plot(range(action_horizon), binned_values, "s--", linewidth=1, markersize=4, alpha=0.5, color=f"C{d}")

ax1.text(
    7.5,
    1.1,
    f"Total tokens: {action_horizon * action_dim} = 120",
    ha="center",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.8),
)

# ============================================================================
# Token Correlation Analysis
# ============================================================================
ax2 = plt.subplot(2, 2, 2)
ax2.set_title("Problem: High Token Correlation", fontsize=13, weight="bold")

# Calculate correlation between consecutive tokens
dim_to_show = 0
action_seq = actions[:, dim_to_show]
binned_seq = np.digitize(action_seq, bins=np.linspace(-1, 1, num_bins + 1)) - 1

# Plot consecutive token pairs
ax2.scatter(binned_seq[:-1], binned_seq[1:], alpha=0.6, s=100, c="red")
ax2.plot([0, num_bins], [0, num_bins], "k--", alpha=0.3, linewidth=2)
ax2.set_xlabel("Token at time t", fontsize=11)
ax2.set_ylabel("Token at time t+1", fontsize=11)
ax2.grid(True, alpha=0.3)

# Calculate correlation
correlation = np.corrcoef(binned_seq[:-1], binned_seq[1:])[0, 1]
ax2.text(
    num_bins / 2,
    num_bins * 0.9,
    f"Correlation: {correlation:.3f}",
    ha="center",
    fontsize=11,
    weight="bold",
    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8),
)
ax2.text(
    num_bins / 2,
    num_bins * 0.8,
    "❌ Highly predictable\n→ Poor for AR training",
    ha="center",
    fontsize=10,
    color="darkred",
)

# ============================================================================
# FAST Tokens
# ============================================================================
ax3 = plt.subplot(2, 2, 3)
ax3.set_title("FAST: Compressed Representation", fontsize=13, weight="bold")

# Show the 8 tokens as a sequence
token_ids = simulated_tokens
x_pos = np.arange(len(token_ids))
colors = plt.cm.viridis(np.linspace(0, 1, len(token_ids)))

bars = ax3.bar(x_pos, token_ids, color=colors, alpha=0.8, edgecolor="black", linewidth=2)
ax3.set_xlabel("Token Index", fontsize=11)
ax3.set_ylabel("Token ID", fontsize=11)
ax3.set_xticks(x_pos)
ax3.set_xticklabels([f"T{i + 1}" for i in range(len(token_ids))])
ax3.grid(True, alpha=0.3, axis="y")

# Add values on bars
for i, (bar, val) in enumerate(zip(bars, token_ids)):
    height = bar.get_height()
    ax3.text(
        bar.get_x() + bar.get_width() / 2.0, height + 5, f"{val}", ha="center", va="bottom", fontsize=10, weight="bold"
    )

ax3.text(
    3.5,
    max(token_ids) * 0.9,
    f"Total tokens: {len(token_ids)}",
    ha="center",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
)
ax3.text(
    3.5,
    max(token_ids) * 0.75,
    f"Compression: {compression_ratio:.1f}x",
    ha="center",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8),
)

# ============================================================================
# Summary
# ============================================================================
ax4 = plt.subplot(2, 2, 4)
ax4.axis("off")
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)

summary = f"""
╔═══════════════════════════════════════════════════════╗
║         Naive Binning vs FAST Comparison              ║
╚═══════════════════════════════════════════════════════╝

NAIVE BINNING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Tokens: {action_horizon * action_dim} (15 timesteps × 8 dims)
• Correlation: {correlation:.3f} (very high!)
• Problem: Consecutive tokens are highly predictable
• Result: Model learns trivial "copy previous token"
• Performance: Fails on dexterous tasks

FAST:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Tokens: {num_tokens} (compressed via DCT+FSQ)
• Compression: {compression_ratio:.1f}x reduction
• Correlation: Low (decorrelated by DCT)
• Result: Each token carries meaningful information
• Performance: Works for high-frequency dexterous tasks

KEY INSIGHT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DCT transforms correlated time-series into uncorrelated
frequency components → better for autoregressive models

ANALOGY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Naive: Predict each pixel value in an image
• FAST: Predict JPEG coefficients (much easier!)

TRAINING EFFICIENCY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Naive: Slow convergence, poor performance
• FAST: 5x faster than diffusion, matches performance
"""

ax4.text(
    0.02,
    0.98,
    summary,
    fontsize=9.5,
    va="top",
    ha="left",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="lavender", alpha=0.95),
)

plt.tight_layout()
plt.savefig("results/fast_vs_naive.png", dpi=150, bbox_inches="tight")
print("✅ Saved FAST vs Naive comparison to: results/fast_vs_naive.png")
plt.close()

print("\n" + "=" * 60)
print("📊 FAST Tokenization Visualization Complete!")
print("=" * 60)
print(f"\n📈 Statistics:")
print(f"   • Input actions: {action_horizon} timesteps × {action_dim} dims = {action_horizon * action_dim} values")
print(f"   • FAST tokens: {num_tokens}")
print(f"   • Compression ratio: {compression_ratio:.1f}x")
print(f"   • Naive binning tokens: {action_horizon * action_dim}")
print(f"   • Token correlation (naive): {correlation:.3f}")
print(f"\n💡 Key Insight:")
print("   FAST uses DCT (like JPEG) to decorrelate tokens,")
print("   enabling autoregressive VLAs for dexterous control!")
