"""
Explanation of BOS and SEP tokens in PaliGemma
Explains BOS and SEP special tokens in PaliGemma
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig = plt.figure(figsize=(18, 12), facecolor="white")

# ============================================================================
# 1. PaliGemma Token Sequence Structure
# ============================================================================
ax1 = plt.subplot(3, 1, 1)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 2)
ax1.axis("off")
ax1.set_title("PaliGemma Input Sequence Structure", fontsize=18, weight="bold", pad=20)

# Token sequence
tokens = [
    ("Image\nTokens", "lightblue", 0.5, 1.5),
    ("BOS", "yellow", 2.0, 0.5),
    ("Prefix\n(Task + State)", "lightgreen", 3.5, 1.5),
    ("SEP\n(\\n)", "orange", 5.5, 0.5),
    ("Suffix\n(Action)", "pink", 7.0, 1.5),
]

y_pos = 1.0
for i, (text, color, width, x_offset) in enumerate(tokens):
    x_start = sum(t[2] for t in tokens[:i]) + i * 0.2

    box = FancyBboxPatch(
        (x_start, y_pos - 0.3), width, 0.6, boxstyle="round,pad=0.05", facecolor=color, edgecolor="black", linewidth=2
    )
    ax1.add_patch(box)

    ax1.text(x_start + width / 2, y_pos, text, ha="center", va="center", fontsize=12, weight="bold")

    # Add arrows between tokens
    if i < len(tokens) - 1:
        arrow_x = x_start + width + 0.05
        ax1.annotate(
            "",
            xy=(arrow_x + 0.1, y_pos),
            xytext=(arrow_x, y_pos),
            arrowprops=dict(arrowstyle="->", lw=2, color="black"),
        )

# Add labels
ax1.text(2.0, 0.2, "Start of text", ha="center", fontsize=10, style="italic", color="darkred")
ax1.text(5.5, 0.2, "Separator\n(newline)", ha="center", fontsize=10, style="italic", color="darkred")

# ============================================================================
# 2. Detailed Explanation
# ============================================================================
ax2 = plt.subplot(3, 1, 2)
ax2.axis("off")
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_title("Special Tokens Explanation", fontsize=16, weight="bold", pad=15)

explanation_text = """
🔹 BOS (Begin of Sequence) Token
   • Token ID: 2 (in PaliGemma/Gemma tokenizer)
   • Symbol: <bos> or <s>
   • Purpose: Marks the beginning of text sequence
   • Added by: tokenizer.encode(text, add_bos=True)
   • Position: After image tokens, before text tokens
   
🔹 SEP (Separator) Token  
   • In PaliGemma: Uses "\\n" (newline) as separator
   • Token ID: 108 (for "\\n" in PaliGemma tokenizer)
   • Purpose: Separates Prefix (input) and Suffix (output)
   • Position: After task description and state, before action prediction
   • Attention: Prefix uses bidirectional attention, Suffix uses causal attention
   
🔹 EOS (End of Sequence) Token
   • Token ID: 1
   • Symbol: </s>
   • Purpose: Marks sequence end (used during generation)
   
🔹 PAD (Padding) Token
   • Token ID: 0
   • Purpose: Pads sequence to fixed length
   • Attention mask: False (does not participate in attention computation)
"""

ax2.text(
    0.05,
    0.95,
    explanation_text,
    fontsize=11,
    va="top",
    ha="left",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
)

# ============================================================================
# 3. Real Example from Your Code
# ============================================================================
ax3 = plt.subplot(3, 1, 3)
ax3.axis("off")
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.set_title("Example: Pi05 Tokenization", fontsize=16, weight="bold", pad=15)

example_text = """
Input Prompt: "place the duck toy into the pink bowl"
State: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

Step 1: Construct Full Prompt
---------------------------------------
full_prompt = f"Task: {instruction}, State: {state_str};\\nAction: "
            = "Task: place the duck toy into the pink bowl, State: 123 45 67 89 ...;\\nAction: "

Step 2: Tokenize with BOS
---------------------------------------
tokens = tokenizer.encode(full_prompt, add_bos=True)

Result Token Sequence:
---------------------------------------
[2]          ← BOS token (added by add_bos=True)
[123, 456]   ← "Task:"
[789, ...]   ← "place the duck toy into the pink bowl"
[234, ...]   ← ", State:"
[567, ...]   ← "123 45 67 89 ..." (discretized state)
[890]        ← ";"
[108]        ← "\\n" (SEP token, separator)
[345, 678]   ← "Action:"

Attention Mechanism:
---------------------------------------
• Prefix (before \\n): Bidirectional attention (ar_mask=0)
  → All prefix tokens can attend to each other
  
• Suffix (after \\n): Causal attention (ar_mask=1)  
  → Each token only attends to previous tokens
  
• BOS token: Part of prefix, uses bidirectional attention
"""

ax3.text(
    0.05,
    0.95,
    example_text,
    fontsize=10,
    va="top",
    ha="left",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.8),
)

plt.tight_layout()
plt.savefig("results/paligemma_special_tokens.png", dpi=150, bbox_inches="tight")
print("✅ Saved explanation to: results/paligemma_special_tokens.png")
plt.close()

# ============================================================================
# Create a second figure showing the actual token IDs
# ============================================================================
fig2 = plt.figure(figsize=(16, 10), facecolor="white")

ax = fig2.add_subplot(111)
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("PaliGemma Special Token IDs", fontsize=18, weight="bold", pad=20)

token_info = """
╔══════════════════════════════════════════════════════════════════╗
║                 PaliGemma/Gemma Tokenizer Special Tokens         ║
╚══════════════════════════════════════════════════════════════════╝

Token ID │ Symbol    │ Name                │ Usage
─────────┼───────────┼─────────────────────┼──────────────────────────
    0    │ <pad>     │ PAD                 │ Padding token
    1    │ </s>      │ EOS                 │ End of sequence
    2    │ <bos>     │ BOS                 │ Begin of sequence ⭐
  108    │ \\n        │ Newline (SEP)       │ Separator in PaliGemma ⭐
  256    │ <image>   │ Image token         │ Placeholder for images
257152   │ -         │ Vocab size          │ Total vocabulary

Special Token Behavior in Your Code:
═══════════════════════════════════════════════════════════════════

1. tokenizer.encode(text, add_bos=True)
   ↓
   Automatically prepends BOS token (ID=2) to the sequence
   
2. full_prompt = "Task: ..., State: ...;\\nAction: "
                                        ↑
                                   SEP token (ID=108)
   
3. Attention Masks:
   • ar_mask = 0 for prefix (before \\n)  → Bidirectional
   • ar_mask = 1 for suffix (after \\n)   → Causal
   
4. Token Type IDs (for training):
   • token_type_id = 0 for prefix (no loss)
   • token_type_id = 1 for suffix (compute loss)

Code Reference:
═══════════════════════════════════════════════════════════════════

# From src/openpi/models/tokenizer.py

class PaligemmaTokenizer:
    def tokenize(self, prompt: str, state: np.ndarray | None = None):
        if state is not None:
            # Pi05 format
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
            #                                              ↑
            #                                         Adds BOS token
        else:
            # Pi0 format
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + \\
                     self._tokenizer.encode("\\n")
            #                                ↑
            #                           SEP token added separately
"""

ax.text(
    0.05,
    0.95,
    token_info,
    fontsize=10,
    va="top",
    ha="left",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="lavender", alpha=0.9),
)

plt.tight_layout()
plt.savefig("results/paligemma_token_ids.png", dpi=150, bbox_inches="tight")
print("✅ Saved token IDs reference to: results/paligemma_token_ids.png")
plt.close()

print("\n📚 Summary:")
print("   • BOS (Begin of Sequence): Token ID 2, marks start of text")
print("   • SEP (Separator): '\\n' (Token ID 108), separates prefix and suffix")
print("   • EOS (End of Sequence): Token ID 1, marks end (used in generation)")
print("   • PAD (Padding): Token ID 0, fills sequence to fixed length")
