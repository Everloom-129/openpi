"""
Matplotlib Coordinate Systems Explanation
Demonstrates different coordinate systems in matplotlib
"""

import matplotlib.pyplot as plt
import numpy as np

# Create a figure with black background
fig = plt.figure(figsize=(16, 10), facecolor="black")

# ============================================================================
# 1. Data Coordinates - Default coordinate system
# ============================================================================
ax1 = plt.subplot(2, 2, 1, facecolor="black")
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.set_title("1. Data Coordinates", color="white", fontsize=14)
ax1.grid(True, alpha=0.3, color="white")

# Draw in data coordinates
ax1.plot([2, 8], [2, 8], "ro-", markersize=10, linewidth=2, label="Line in data coords")
ax1.text(
    5,
    5,
    "Point (5, 5)\nin data coords",
    color="yellow",
    fontsize=12,
    ha="center",
    bbox=dict(boxstyle="round", facecolor="blue", alpha=0.5),
)

# Label axes
ax1.text(10.5, 0, "X →", color="cyan", fontsize=12, va="center")
ax1.text(0, 10.5, "Y ↑", color="cyan", fontsize=12, ha="center")
ax1.legend(loc="upper left", facecolor="gray")

# ============================================================================
# 2. Axes Coordinates - Normalized coordinates from 0 to 1
# ============================================================================
ax2 = plt.subplot(2, 2, 2, facecolor="black")
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 100)
ax2.set_title("2. Axes Coordinates (0-1)", color="white", fontsize=14)

# Use transform=ax2.transAxes to use axes coordinates
# (0, 0) = bottom-left, (1, 1) = top-right
ax2.text(
    0.5,
    0.5,
    "Center (0.5, 0.5)\nin axes coords",
    transform=ax2.transAxes,
    color="yellow",
    fontsize=12,
    ha="center",
    va="center",
    bbox=dict(boxstyle="round", facecolor="blue", alpha=0.5),
)

ax2.text(0.1, 0.9, "(0.1, 0.9)", transform=ax2.transAxes, color="lime", fontsize=10)
ax2.text(0.9, 0.1, "(0.9, 0.1)", transform=ax2.transAxes, color="lime", fontsize=10)

# Draw boundary box
ax2.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], "c--", linewidth=2, transform=ax2.transAxes, label="Axes boundary")

ax2.text(0, -0.1, "(0, 0) bottom-left", transform=ax2.transAxes, color="cyan", fontsize=10, ha="left")
ax2.text(1, 1.05, "(1, 1) top-right", transform=ax2.transAxes, color="cyan", fontsize=10, ha="right")
ax2.legend(loc="upper left", facecolor="gray")

# ============================================================================
# 3. Application in your tokenizer visualization
# ============================================================================
ax3 = plt.subplot(2, 2, 3, facecolor="black")
ax3.axis("off")
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.set_title("3. Your Token Visualization (Axes Coordinates)", color="white", fontsize=14)

# Simulate the coordinate system in your code
x_start, y_start = 0.01, 0.95
x_limit = 0.98
line_h_frac = 0.08

# Draw usable area
ax3.axhline(y=y_start, color="cyan", linestyle="--", alpha=0.5, label=f"y_start={y_start}")
ax3.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="y=0 (bottom limit)")
ax3.axvline(x=x_start, color="cyan", linestyle="--", alpha=0.5, label=f"x_start={x_start}")
ax3.axvline(x=x_limit, color="orange", linestyle="--", alpha=0.5, label=f"x_limit={x_limit}")

# Simulate text placement
y = y_start
for i in range(5):
    ax3.text(
        x_start,
        y,
        f"Line {i + 1} (y={y:.2f})",
        color="white",
        fontsize=10,
        va="top",
        bbox=dict(facecolor="green", alpha=0.3, edgecolor="white"),
    )
    y -= line_h_frac
    if y < 0:
        ax3.text(x_start, y + line_h_frac, "...", color="red", fontsize=12, va="top")
        break

ax3.legend(loc="lower right", facecolor="gray", fontsize=9)

# Add annotation
ax3.annotate("", xy=(x_start, 0), xytext=(x_start, y_start), arrowprops=dict(arrowstyle="<->", color="yellow", lw=2))
ax3.text(
    x_start - 0.05,
    y_start / 2,
    f"Available\nheight\n≈{y_start:.2f}",
    color="yellow",
    fontsize=10,
    ha="right",
    va="center",
)

# ============================================================================
# 4. Coordinate Systems Summary
# ============================================================================
ax4 = plt.subplot(2, 2, 4, facecolor="black")
ax4.axis("off")
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.set_title("4. Coordinate Systems Summary", color="white", fontsize=14)

summary_text = """
Coordinate Systems Comparison:

1. Data Coordinates
   • Default coordinate system
   • Range determined by set_xlim/set_ylim
   • Example: ax.plot([0, 10], [0, 10])
   
2. Axes Coordinates
   • Normalized coordinates: (0, 0) to (1, 1)
   • (0, 0) = bottom-left, (1, 1) = top-right
   • Usage: transform=ax.transAxes
   • This is what your code uses!
   
3. Figure Coordinates
   • Normalized coordinates for entire figure
   • Usage: transform=fig.transFigure
   
4. Display Coordinates
   • Pixel coordinates
   • Rarely used directly

In your code:
• ax.set_xlim(0, 1), ax.set_ylim(0, 1)
  sets data coordinate range to 0-1
• So data coordinates = axes coordinates
• x=0.5 means horizontal center
• y=0.95 means near top (95%)
• y=0 means bottom
• y<0 means outside visible area (clipped)
"""

ax4.text(
    0.05,
    0.95,
    summary_text,
    color="white",
    fontsize=10,
    va="top",
    ha="left",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="darkblue", alpha=0.3),
)

plt.tight_layout()
plt.savefig("results/matplotlib_coordinates_explanation.png", dpi=150, facecolor="black")
print("Saved explanation to: results/matplotlib_coordinates_explanation.png")
plt.close()

# ============================================================================
# Create second figure: Detailed demonstration of your token layout logic
# ============================================================================
fig2 = plt.figure(figsize=(16, 10), facecolor="black")

ax = fig2.add_subplot(111, facecolor="black")
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("Token Layout Logic in Your Code", color="white", fontsize=16, pad=20)

# Parameters
char_w_frac = 0.011
line_h_frac = 0.08
x_start, y_start = 0.01, 0.95
x_limit = 0.98

# Draw boundaries
ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], "c-", linewidth=2, alpha=0.3)
ax.fill_between([x_start, x_limit], 0, y_start, alpha=0.1, color="green", label="Usable area")

# Simulate token placement
tokens = ["Task:", "place", "the", "duck", "toy", "into", "\\n", "State:", "123", "45", "67", "89", "\\n", "Action:"]
colors_demo = plt.cm.Set3(np.linspace(0, 1, len(tokens)))

x, y = x_start, y_start
for i, token in enumerate(tokens):
    display_str = token
    w = len(display_str) * char_w_frac

    # Check wrap
    is_newline = token == "\\n"
    if (x + w > x_limit) or is_newline:
        # Draw line wrap arrow
        ax.annotate(
            "",
            xy=(x_start, y - line_h_frac),
            xytext=(x, y),
            arrowprops=dict(arrowstyle="->", color="yellow", lw=1, alpha=0.5),
        )
        x = x_start
        y -= line_h_frac

    if y < 0:
        ax.text(x, y + line_h_frac, "...", color="red", fontsize=14, va="top", weight="bold")
        ax.text(0.5, -0.1, "⚠️ Content truncated! y < 0", color="red", fontsize=14, ha="center", weight="bold")
        break

    # Draw token
    ax.text(
        x,
        y,
        display_str,
        fontsize=12,
        fontname="monospace",
        backgroundcolor=colors_demo[i],
        va="top",
        color="black",
        bbox=dict(facecolor=colors_demo[i], edgecolor="white", pad=2),
    )

    # Annotate coordinates
    if i < 3 or token == "Action:":
        ax.text(x, y - 0.03, f"({x:.2f},{y:.2f})", fontsize=8, color="cyan", va="top")

    x += w + 0.001

# Add explanations
annotations = [
    (x_start, y_start, "Start position\n(x_start, y_start)", "lime"),
    (x_limit, y_start, "Right limit\n(x_limit)", "orange"),
    (x_start, 0, "Bottom limit\n(y=0)", "red"),
]

for x_pos, y_pos, text, color in annotations:
    ax.plot(x_pos, y_pos, "o", color=color, markersize=10)
    ax.text(x_pos, y_pos - 0.05, text, color=color, fontsize=10, ha="center", va="top")

# Add formula explanation
formula_text = """
Key Variables:
• char_w_frac = 0.011  (width per character)
• line_h_frac = 0.08   (line height)
• x_start = 0.01       (left margin)
• y_start = 0.95       (starting height)
• x_limit = 0.98       (right boundary)

Logic:
1. Each token width = len(token) × char_w_frac
2. If x + width > x_limit → wrap to next line
3. Line wrap: x = x_start, y -= line_h_frac
4. If y < 0 → stop (content truncated)
"""

ax.text(
    0.02,
    0.35,
    formula_text,
    fontsize=10,
    color="white",
    family="monospace",
    va="top",
    bbox=dict(boxstyle="round", facecolor="darkblue", alpha=0.5),
)

plt.tight_layout()
plt.savefig("results/token_layout_logic.png", dpi=150, facecolor="black")
print("Saved token layout logic to: results/token_layout_logic.png")
plt.close()

print("\n✅ Generated two explanation images:")
print("   1. results/matplotlib_coordinates_explanation.png")
print("   2. results/token_layout_logic.png")
