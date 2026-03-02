"""H7.1 — Word-Specific Attention Pipeline.

Batch-processes episodes in the DROID dataset to compute per-word
attention statistics across layers and frames.

For each episode / keyframe:
  1. Runs policy inference (generates prefix attention maps).
  2. Tokenizes the prompt and finds subword indices for each target word.
  3. Extracts the attention rows for those tokens attending to image patches.
  4. Computes per-word statistics (entropy, concentration, peak).
  5. Saves heatmap overlays (optional) and JSON stats.

After processing all episodes an aggregation step:
  - Plots entropy vs layer per word (success vs failure).
  - Computes Gini-style concentration mean / std.
  - Saves a markdown report.

Usage:
    python viz/h7_word_attention_pipeline.py
"""
from __future__ import annotations

import functools
import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

import attn_map as _attn_map
from attn_map import get_keyframes, select_best_gpu
from h_word_attention import (
    TEXT_START_IDX,
    TOTAL_IMAGE_TOKENS,
    NUM_IMAGE_TOKENS,
    find_word_token_indices,
    load_prefix_attn,
    overlay_heatmap,
    tokenize_prompt,
)
from attn_pipeline import copy_instruction, get_video_length, load_toy_example


# ── Configuration ─────────────────────────────────────────────────────────────

CHECKPOINT_DIR = "./checkpoints/viz/pi05_droid_pytorch"
OPEN_LOOP_HORIZON = 8
LAYERS = [1, 4, 5, 7, 10]
FPS_VIDEO = 5
CAMERA = "right"

# Words to track.  Drawn from each episode's instruction at runtime + fixed set.
FIXED_TARGET_WORDS: list[str] = []   # e.g. ["pick", "place", "find"]
# If True, automatically add nouns from the instruction to the target list.
AUTO_EXTRACT_WORDS = True

HEAD_AGG = "max"   # "max" | "mean" — how to aggregate attention heads

SAVE_HEATMAPS = True   # Save per-word heatmap PNGs (uses disk space)
SAVE_ENTROPY_PLOT = True  # Save per-episode entropy-vs-layer plot

# ── Statistics helpers ─────────────────────────────────────────────────────────


def _concentration(attn_1d: np.ndarray, top_frac: float = 0.25) -> float:
    """Fraction of total attention mass in the top `top_frac` of patches.

    Values near 1.0 mean highly focused; near `top_frac` means uniform.
    """
    k = max(1, int(len(attn_1d) * top_frac))
    top_k_sum = float(np.sort(attn_1d)[-k:].sum())
    total = float(attn_1d.sum()) + 1e-10
    return top_k_sum / total


def extract_word_stats(
    attn: np.ndarray,
    token_texts: list[str],
    target_word: str,
    head_agg: str = "max",
) -> dict | None:
    """Return per-word attention statistics for one layer.

    Args:
        attn:        Prefix attention [heads, seq_len, seq_len].
        token_texts: Token string list for the full prompt (including BOS).
        target_word: Word to look up.
        head_agg:    How to aggregate heads ("max" | "mean").

    Returns:
        Dict with keys: entropy, concentration_ext, concentration_wrist,
        peak_ext, peak_wrist, token_indices, token_texts_found.
        Returns None when the word is not found or is out of range.
    """
    _, seq_len, _ = attn.shape

    local_indices = find_word_token_indices(token_texts, target_word)
    if not local_indices:
        return None

    global_indices = [TEXT_START_IDX + li for li in local_indices if (TEXT_START_IDX + li) < seq_len]
    if not global_indices:
        return None

    # [heads, n_word_tokens, TOTAL_IMAGE_TOKENS]
    word_attn = attn[:, global_indices, :TOTAL_IMAGE_TOKENS]   # [H, k, 512]
    word_attn = word_attn.mean(axis=1)                          # [H, 512]  avg over subwords

    if head_agg == "max":
        agg = word_attn.max(axis=0)   # [512]
    else:
        agg = word_attn.mean(axis=0)  # [512]

    # Entropy (over image patch distribution, per head, then averaged)
    per_head_ent = -np.sum(word_attn * np.log(word_attn + 1e-10), axis=-1)  # [H]
    entropy = float(per_head_ent.mean())

    ext_agg = agg[:NUM_IMAGE_TOKENS]       # [256]
    wrist_agg = agg[NUM_IMAGE_TOKENS:]     # [256]

    return {
        "entropy": entropy,
        "concentration_ext": _concentration(ext_agg),
        "concentration_wrist": _concentration(wrist_agg),
        "peak_ext": float(ext_agg.max()),
        "peak_wrist": float(wrist_agg.max()),
        "token_indices": global_indices,
        "token_texts_found": [token_texts[li] for li in local_indices if li < len(token_texts)],
    }


# ── Instruction word extraction ────────────────────────────────────────────────

_STOP_WORDS = {
    "the", "a", "an", "and", "or", "to", "in", "on", "of", "it",
    "is", "that", "this", "for", "with", "into", "from", "its",
    "find", "pick", "up", "place", "put", "move", "bring", "take",
    "task", "state", "action",
}


def extract_content_words(prompt: str) -> list[str]:
    """Extract non-stop content words from prompt for automatic targeting."""
    words = prompt.lower().replace(",", " ").replace(".", " ").split()
    seen: set[str] = set()
    result: list[str] = []
    for w in words:
        w = w.strip()
        if w and w not in _STOP_WORDS and w not in seen and len(w) > 2:
            seen.add(w)
            result.append(w)
    return result


# ── Per-frame processing ───────────────────────────────────────────────────────

def process_frame(
    policy,
    example: dict,
    frame_idx: int,
    episode_dir: Path,
    device_id: str,
    layers: list[int],
    target_words: list[str],
    head_agg: str = "max",
    save_heatmaps: bool = True,
) -> dict:
    """Run inference + word attention analysis for one frame.

    Returns:
        Nested dict  {layer_idx: {word: stats_dict}}.
    """
    # ── Run inference (writes attn maps to attn/{device_id}/layers_prefix/) ──
    _ = policy.infer(example)

    prefix_dir = f"attn/{device_id}/layers_prefix"

    # ── Tokenize once ────────────────────────────────────────────────────────
    try:
        _, token_texts = tokenize_prompt(example)
    except Exception as exc:
        print(f"    [h7_pipeline] Tokenizer failed: {exc}")
        return {}

    frame_results: dict = {}

    # Load images for optional heatmap overlays
    if save_heatmaps:
        ext_img = example["observation/exterior_image_1_left"]
        wrist_img = example["observation/wrist_image_left"]
        heatmap_dir = episode_dir / "word_attn" / f"{frame_idx:05d}"
        heatmap_dir.mkdir(parents=True, exist_ok=True)

    for layer_idx in layers:
        attn = load_prefix_attn(layer_idx, input_dir=prefix_dir)
        if attn is None:
            continue

        layer_stats: dict[str, dict] = {}

        for word in target_words:
            stats = extract_word_stats(attn, token_texts, word, head_agg=head_agg)
            if stats is None:
                continue
            layer_stats[word] = stats

            # ── Optional heatmap save ─────────────────────────────────────
            if save_heatmaps:
                agg_raw = attn[:, stats["token_indices"], :TOTAL_IMAGE_TOKENS].mean(axis=1)
                if head_agg == "max":
                    agg = agg_raw.max(axis=0)
                else:
                    agg = agg_raw.mean(axis=0)
                ext_hmap = agg[:NUM_IMAGE_TOKENS].reshape(16, 16)
                wrist_hmap = agg[NUM_IMAGE_TOKENS:].reshape(16, 16)

                _, ext_ov = overlay_heatmap(ext_img, ext_hmap)
                _, wrist_ov = overlay_heatmap(wrist_img, wrist_hmap)

                out = np.hstack([ext_ov, wrist_ov])   # 224×448 combined
                cv2.imwrite(
                    str(heatmap_dir / f"L{layer_idx:02d}_{word}.jpg"),
                    cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                )

        if layer_stats:
            frame_results[layer_idx] = layer_stats

    return frame_results


# ── Per-episode processing ─────────────────────────────────────────────────────

def process_episode(
    policy,
    data_dir: Path,
    episode_dir: Path,
    device_id: str,
    layers: list[int],
    camera: str = "right",
    fixed_words: list[str] | None = None,
    auto_extract: bool = True,
    head_agg: str = "max",
    save_heatmaps: bool = True,
    save_entropy_plot: bool = True,
) -> dict | None:
    """Process one episode: iterate over keyframes, collect stats, save outputs.

    Returns:
        Per-episode summary dict or None on failure.
    """
    total_frames = get_video_length(data_dir)
    if total_frames == 0:
        print(f"  [h7_pipeline] No frames found in {data_dir}")
        return None

    keyframes = get_keyframes(total_frames, OPEN_LOOP_HORIZON)
    print(f"  Keyframes: {keyframes}")

    # Storage: {frame_idx: {layer_idx: {word: stats}}}
    all_frame_results: dict = {}

    # Determine target words from the first loadable frame's instruction
    target_words = list(fixed_words or [])
    instruction_for_words_extracted = False

    for frame_idx in keyframes:
        try:
            example = load_toy_example(data_dir, frame_idx, camera=camera)
        except FileNotFoundError as exc:
            print(f"  Frame {frame_idx}: {exc}")
            continue

        # Auto-extract content words from instruction (once per episode)
        if auto_extract and not instruction_for_words_extracted:
            content = extract_content_words(example.get("prompt", ""))
            for w in content:
                if w not in target_words:
                    target_words.append(w)
            instruction_for_words_extracted = True
            print(f"  Target words: {target_words}")

        if not target_words:
            print(f"  [h7_pipeline] No target words; skipping.")
            return None

        print(f"  Frame {frame_idx}...")
        frame_stats = process_frame(
            policy,
            example,
            frame_idx,
            episode_dir,
            device_id=device_id,
            layers=layers,
            target_words=target_words,
            head_agg=head_agg,
            save_heatmaps=save_heatmaps,
        )
        if frame_stats:
            all_frame_results[frame_idx] = frame_stats

    if not all_frame_results:
        return None

    # ── Per-episode entropy-vs-layer plot ─────────────────────────────────────
    if save_entropy_plot and target_words:
        _save_entropy_vs_layer_plot(
            all_frame_results, target_words, layers, episode_dir / "word_attn_entropy.jpg"
        )

    # ── Compute episode-level layer statistics ────────────────────────────────
    layer_statistics = _compute_layer_statistics(all_frame_results, target_words, layers)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_path = episode_dir / "h7_word_attn_results.json"
    result_payload = {
        "target_words": target_words,
        "layers": layers,
        "frame_results": _serializable(all_frame_results),
        "layer_statistics": _serializable(layer_statistics),
    }
    with open(json_path, "w") as f:
        json.dump(result_payload, f, indent=2)
    print(f"  Saved → {json_path.name}")

    return result_payload


# ── Statistics helpers ─────────────────────────────────────────────────────────

def _compute_layer_statistics(
    all_frame_results: dict,
    target_words: list[str],
    layers: list[int],
) -> dict:
    """Aggregate frame-level stats into per-layer mean/std per word."""
    layer_stats: dict = {}
    for layer_idx in layers:
        layer_stats[layer_idx] = {}
        for word in target_words:
            entropies = []
            conc_ext = []
            conc_wrist = []
            for frame_data in all_frame_results.values():
                if layer_idx not in frame_data:
                    continue
                ws = frame_data[layer_idx].get(word)
                if ws is None:
                    continue
                entropies.append(ws["entropy"])
                conc_ext.append(ws["concentration_ext"])
                conc_wrist.append(ws["concentration_wrist"])
            if not entropies:
                continue
            layer_stats[layer_idx][word] = {
                "entropy_mean": float(np.mean(entropies)),
                "entropy_std": float(np.std(entropies)),
                "concentration_ext_mean": float(np.mean(conc_ext)),
                "concentration_wrist_mean": float(np.mean(conc_wrist)),
                "n_frames": len(entropies),
            }
    return layer_stats


def _save_entropy_vs_layer_plot(
    all_frame_results: dict,
    target_words: list[str],
    layers: list[int],
    out_path: Path,
) -> None:
    """Plot mean entropy ± std vs layer for each word across frames."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for word in target_words:
        means, stds = [], []
        valid_layers = []
        for layer_idx in layers:
            entropies = []
            for frame_data in all_frame_results.values():
                if layer_idx not in frame_data:
                    continue
                ws = frame_data[layer_idx].get(word)
                if ws is not None:
                    entropies.append(ws["entropy"])
            if entropies:
                means.append(np.mean(entropies))
                stds.append(np.std(entropies))
                valid_layers.append(layer_idx)
        if valid_layers:
            means_arr = np.array(means)
            stds_arr = np.array(stds)
            line, = ax.plot(valid_layers, means_arr, marker="o", label=f"'{word}'")
            ax.fill_between(valid_layers, means_arr - stds_arr, means_arr + stds_arr,
                            alpha=0.15, color=line.get_color())

    ax.set_xlabel("Layer")
    ax.set_ylabel("Attention Entropy (nats)")
    ax.set_title("H7.1 Word Attention Entropy vs Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()


def _serializable(obj):
    """Recursively convert numpy types to Python natives for JSON."""
    if isinstance(obj, dict):
        return {str(k): _serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serializable(i) for i in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ── Aggregation across episodes ────────────────────────────────────────────────

def aggregate_results(
    results_root: Path,
    layers: list[int],
    output_dir: Path | None = None,
) -> dict | None:
    """Aggregate per-episode JSON files into multi-episode summary.

    Directory structure expected:
        results_root/
          success/<date>/<episode>/h7_word_attn_results.json
          failure/<date>/<episode>/h7_word_attn_results.json

    Returns:
        Aggregated statistics dict.
    """
    if output_dir is None:
        output_dir = results_root
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("H7.1 AGGREGATING WORD ATTENTION RESULTS")
    print(f"{'=' * 60}\n")

    # ── Collect all episode JSON files ────────────────────────────────────────
    all_data: list[dict] = []
    episode_count: dict[str, int] = {"success": 0, "failure": 0}

    for outcome in ["success", "failure"]:
        outcome_dir = results_root / outcome
        if not outcome_dir.exists():
            continue
        json_files = list(outcome_dir.rglob("h7_word_attn_results.json"))
        print(f"  {outcome}: {len(json_files)} episodes")
        for jf in json_files:
            try:
                data = json.loads(jf.read_text())
                data["outcome"] = outcome
                data["episode_id"] = jf.parent.name
                all_data.append(data)
                episode_count[outcome] += 1
            except Exception as exc:
                print(f"  Error loading {jf}: {exc}")

    if not all_data:
        print("No episode data found.")
        return None

    total = sum(episode_count.values())
    print(f"\nLoaded {total} episodes: {episode_count}")

    # ── Collect all unique words and layers ───────────────────────────────────
    all_words: set[str] = set()
    for d in all_data:
        all_words.update(d.get("target_words", []))
    words = sorted(all_words)
    print(f"Words across all episodes: {words}")

    # ── Aggregate entropy per (outcome, layer, word) ──────────────────────────
    # Structure: agg[outcome][layer][word] = [entropy values]
    agg: dict[str, dict[int, dict[str, list[float]]]] = {
        "success": {l: {w: [] for w in words} for l in layers},
        "failure": {l: {w: [] for w in words} for l in layers},
    }
    conc_agg: dict[str, dict[int, dict[str, list[float]]]] = {
        "success": {l: {w: [] for w in words} for l in layers},
        "failure": {l: {w: [] for w in words} for l in layers},
    }

    for d in all_data:
        outcome = d["outcome"]
        layer_stats = d.get("layer_statistics", {})
        for layer_idx in layers:
            ls = layer_stats.get(str(layer_idx), {})
            for word in words:
                ws = ls.get(word)
                if ws is None:
                    continue
                agg[outcome][layer_idx][word].append(ws["entropy_mean"])
                c = (ws.get("concentration_ext_mean", 0) + ws.get("concentration_wrist_mean", 0)) / 2
                conc_agg[outcome][layer_idx][word].append(c)

    # ── Plot 1: Entropy vs Layer per word (success vs failure) ────────────────
    _plot_entropy_comparison(agg, words, layers, output_dir / "h7_multi_entropy_vs_layer.jpg", total)

    # ── Plot 2: Concentration heatmap (word × layer, two panels) ─────────────
    _plot_concentration_heatmap(conc_agg, words, layers, output_dir / "h7_concentration_heatmap.jpg")

    # ── Report ────────────────────────────────────────────────────────────────
    report_path = output_dir / "h7_word_attn_report.md"
    _write_report(agg, conc_agg, words, layers, episode_count, report_path)
    print(f"  Report → {report_path.name}")

    # ── Save aggregate JSON ───────────────────────────────────────────────────
    agg_json = {
        "total_episodes": total,
        "episode_count": episode_count,
        "words": words,
        "layers": layers,
        "entropy": {
            outcome: {
                str(l): {
                    w: {
                        "mean": float(np.mean(agg[outcome][l][w])) if agg[outcome][l][w] else None,
                        "std": float(np.std(agg[outcome][l][w])) if agg[outcome][l][w] else None,
                        "n": len(agg[outcome][l][w]),
                    }
                    for w in words
                }
                for l in layers
            }
            for outcome in ["success", "failure"]
        },
    }
    agg_json_path = output_dir / "h7_word_attn_aggregate.json"
    agg_json_path.write_text(json.dumps(agg_json, indent=2))
    print(f"  Aggregate JSON → {agg_json_path.name}")

    print(f"\n{'=' * 60}")
    print("AGGREGATION COMPLETE")
    print(f"{'=' * 60}\n")
    return agg_json


def _plot_entropy_comparison(
    agg: dict,
    words: list[str],
    layers: list[int],
    out_path: Path,
    total_episodes: int,
) -> None:
    """One subplot per word: entropy vs layer, success (blue) vs failure (red)."""
    n = len(words)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    colors = {"success": "steelblue", "failure": "coral"}

    for ax, word in zip(axes, words):
        for outcome in ["success", "failure"]:
            means, stds, valid_ls = [], [], []
            for l in layers:
                vals = agg[outcome][l][word]
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
                    valid_ls.append(l)
            if not valid_ls:
                continue
            m = np.array(means)
            s = np.array(stds)
            line, = ax.plot(valid_ls, m, marker="o", color=colors[outcome],
                            label=outcome.capitalize())
            ax.fill_between(valid_ls, m - s, m + s, alpha=0.15, color=colors[outcome])

        ax.set_title(f"'{word}'", fontsize=12)
        ax.set_xlabel("Layer")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Attention Entropy (nats)")
    plt.suptitle(
        f"H7.1 Word Attention Entropy: Success vs Failure\n({total_episodes} episodes)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Entropy plot → {out_path.name}")


def _plot_concentration_heatmap(
    conc_agg: dict,
    words: list[str],
    layers: list[int],
    out_path: Path,
) -> None:
    """Heatmap: rows=words, cols=layers, two panels (success / failure)."""
    if not words or not layers:
        return
    fig, axes = plt.subplots(1, 2, figsize=(max(10, 2 * len(layers)), max(4, len(words))))

    for ax, outcome in zip(axes, ["success", "failure"]):
        matrix = np.zeros((len(words), len(layers)))
        for wi, w in enumerate(words):
            for li, l in enumerate(layers):
                vals = conc_agg[outcome][l][w]
                matrix[wi, li] = np.mean(vals) if vals else 0.0
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([f"L{l}" for l in layers])
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.set_title(f"{outcome.capitalize()}", fontsize=11)
        ax.set_xlabel("Layer")
        for wi in range(len(words)):
            for li in range(len(layers)):
                ax.text(li, wi, f"{matrix[wi, li]:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if matrix[wi, li] > 0.6 else "black")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle("H7.1 Attention Concentration (top-25% patches)", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Concentration heatmap → {out_path.name}")


def _write_report(
    agg: dict,
    conc_agg: dict,
    words: list[str],
    layers: list[int],
    episode_count: dict,
    report_path: Path,
) -> None:
    lines = [
        "# H7.1 Word-Specific Attention — Multi-Episode Report\n\n",
        f"**Episodes**: {sum(episode_count.values())} "
        f"(success={episode_count['success']}, failure={episode_count['failure']})\n",
        f"**Layers analysed**: {layers}\n",
        f"**Words tracked**: {words}\n\n",
        "## Entropy per Word per Layer\n\n",
    ]
    for outcome in ["success", "failure"]:
        lines.append(f"### {outcome.capitalize()}\n\n")
        header = "| Word | " + " | ".join(f"L{l}" for l in layers) + " |\n"
        sep = "|------|" + "------|" * len(layers) + "\n"
        lines += [header, sep]
        for w in words:
            row = f"| {w} |"
            for l in layers:
                vals = agg[outcome][l][w]
                if vals:
                    row += f" {np.mean(vals):.2f}±{np.std(vals):.2f} |"
                else:
                    row += " — |"
            lines.append(row + "\n")
        lines.append("\n")

    lines.append("## Key Findings\n\n")
    # Lowest entropy word per layer (most focused)
    lines.append("### Most Focused Word per Layer (lowest entropy, success episodes)\n\n")
    for l in layers:
        best_word, best_ent = None, float("inf")
        for w in words:
            vals = agg["success"][l][w]
            if vals and np.mean(vals) < best_ent:
                best_ent = np.mean(vals)
                best_word = w
        if best_word:
            lines.append(f"- Layer {l}: **'{best_word}'** (entropy={best_ent:.2f})\n")

    report_path.write_text("".join(lines))


# ── Timer decorator ────────────────────────────────────────────────────────────

def _timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        val = func(*args, **kwargs)
        print(f"[timer] {func.__name__} completed in {time.perf_counter() - t0:.1f}s")
        return val
    return wrapper


# ── Main ───────────────────────────────────────────────────────────────────────

@_timer
def main():
    # ── Dataset paths ─────────────────────────────────────────────────────────
    TEST_CASE = "/data3/tonyw/aawr_offline/bookshelf_d/"
    DATA_ROOT = Path(TEST_CASE)
    RESULTS_ROOT = Path("/data3/tonyw/toy_cube_benchmark/pi05/") / DATA_ROOT.name / CAMERA / "word_attn"

    print(f"Data root:    {DATA_ROOT}")
    print(f"Results root: {RESULTS_ROOT}")
    print(f"Camera:       {CAMERA}")
    print(f"Layers:       {LAYERS}")
    print()

    # ── Load policy ───────────────────────────────────────────────────────────
    device_id = select_best_gpu()
    device = f"cuda:{device_id}"
    print(f"Loading policy from {CHECKPOINT_DIR} on {device}...")
    policy = _attn_map.get_policy(CHECKPOINT_DIR, device=device)
    print("Policy loaded.\n")

    # ── Counters ──────────────────────────────────────────────────────────────
    total_ep = processed_ep = skipped_ep = error_ep = 0

    for outcome in ["success", "failure"]:
        outcome_dir = DATA_ROOT / outcome
        if not outcome_dir.exists():
            print(f"Skipping {outcome} (directory not found)")
            continue

        print(f"\n{'=' * 60}")
        print(f"Outcome: {outcome.upper()}")
        print(f"{'=' * 60}\n")

        for date_dir in sorted(outcome_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            for h5_path in sorted(date_dir.rglob("trajectory.h5")):
                data_dir = h5_path.parent
                episode_id = data_dir.name
                total_ep += 1

                print(f"\n[{total_ep}] {outcome}/{date_dir.name}/{episode_id}")

                rel_path = data_dir.relative_to(DATA_ROOT)
                episode_dir = RESULTS_ROOT / rel_path
                episode_dir.mkdir(parents=True, exist_ok=True)

                # Skip if already done
                marker = episode_dir / "h7_word_attn_results.json"
                if marker.exists():
                    print("  Already processed, skipping.")
                    skipped_ep += 1
                    continue

                try:
                    result = process_episode(
                        policy,
                        data_dir=data_dir,
                        episode_dir=episode_dir,
                        device_id=str(device_id),
                        layers=LAYERS,
                        camera=CAMERA,
                        fixed_words=FIXED_TARGET_WORDS,
                        auto_extract=AUTO_EXTRACT_WORDS,
                        head_agg=HEAD_AGG,
                        save_heatmaps=SAVE_HEATMAPS,
                        save_entropy_plot=SAVE_ENTROPY_PLOT,
                    )
                    if result:
                        copy_instruction(data_dir, episode_dir)
                        processed_ep += 1
                        print(f"  Episode complete.")
                    else:
                        error_ep += 1
                        print(f"  No results generated.")
                except Exception as exc:
                    import traceback
                    print(f"  Error: {exc}")
                    traceback.print_exc()
                    error_ep += 1

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total:     {total_ep}")
    print(f"Processed: {processed_ep}")
    print(f"Skipped:   {skipped_ep}")
    print(f"Errors:    {error_ep}")
    print(f"Results:   {RESULTS_ROOT}\n")

    # ── Aggregation ───────────────────────────────────────────────────────────
    if processed_ep + skipped_ep > 0:
        aggregate_results(RESULTS_ROOT, LAYERS, output_dir=RESULTS_ROOT)


if __name__ == "__main__":
    main()
