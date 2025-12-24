"""
Object Detection Pipeline for Batch Processing

Process multiple episodes to evaluate attention-object correlation.
Compatible with DROID format datasets with object detection masks.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import attn_map
from h1_1_object_detection import (
    create_all_layer_videos,
    generate_summary_plot,
    run_object_detection,
)
from pipeline import copy_instruction, get_video_length, load_toy_example, timer


OPEN_LOOP_HORIZON = 8
LAYERS = range(18)
FPS_VIDEO = 5
CAMERA = "right"

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def has_object_masks(data_dir: Path, frame_idx: int) -> bool:
    """Check if object detection masks exist for a given frame"""
    dinox_info_path = data_dir / "dinox_info.json"
    if not dinox_info_path.exists():
        return False

    try:
        with open(dinox_info_path) as f:
            dinox_info = json.load(f)

        frame_key = str(frame_idx)
        if frame_key not in dinox_info:
            return False

        detections = dinox_info[frame_key]
        # Check if any detection has valid mask
        for det in detections:
            if det.get("mask_file") and det.get("score", 0) > 0:
                mask_path = data_dir / "masks" / det["mask_file"]
                if mask_path.exists():
                    return True
        return False
    except Exception as e:
        print(f"Error checking masks: {e}")
        return False


def get_white_list_frames(data_dir: Path) -> list[int]:
    """Load white list frames if available, otherwise return empty list"""
    white_list_path = data_dir / "white_list.json"
    if not white_list_path.exists():
        return []

    try:
        with open(white_list_path) as f:
            white_list = json.load(f)
        return sorted(white_list)
    except Exception as e:
        print(f"Error loading white list: {e}")
        return []


def aggregate_episodes_analysis(results_root: Path, layers: list[int], output_dir: Path | None = None):
    """
    Aggregate results from multiple episodes and generate comprehensive analysis.

    Args:
        results_root: Root directory containing all processed episodes
        layers: List of layer indices to analyze
        output_dir: Output directory for aggregated results (defaults to results_root)

    Returns:
        Dictionary with aggregated statistics
    """
    if output_dir is None:
        output_dir = results_root

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("AGGREGATING MULTI-EPISODE RESULTS")
    print(f"{'=' * 60}\n")
    print(f"Results root: {results_root}")

    # Collect all episode results
    all_episode_data = []
    episode_count = {"success": 0, "failure": 0}

    for outcome in ["success", "failure"]:
        outcome_dir = results_root / outcome
        if not outcome_dir.exists():
            continue

        # Find all JSON result files
        json_files = list(outcome_dir.rglob("h1_1_obj_attn_results.json"))
        print(f"Found {len(json_files)} episodes in {outcome}")

        for json_path in json_files:
            try:
                with open(json_path) as f:
                    data = json.load(f)

                episode_path = json_path.parent
                episode_id = episode_path.name

                # Extract layer statistics
                if "layer_statistics" in data:
                    all_episode_data.append(
                        {
                            "outcome": outcome,
                            "episode_id": episode_id,
                            "episode_path": str(episode_path),
                            "layer_stats": data["layer_statistics"],
                            "frame_count": len(data.get("frame_results", {})),
                        }
                    )
                    episode_count[outcome] += 1

            except Exception as e:
                print(f"  Error loading {json_path}: {e}")

    if not all_episode_data:
        print("No episode data found!")
        return None

    total_episodes = sum(episode_count.values())
    print(f"\nLoaded {total_episodes} episodes:")
    print(f"  Success: {episode_count['success']}")
    print(f"  Failure: {episode_count['failure']}")

    # Aggregate statistics by layer
    layer_aggregates = {}
    for layer in layers:
        layer_key = str(layer)
        success_metrics = {"overlap": [], "concentration": [], "iou": []}
        failure_metrics = {"overlap": [], "concentration": [], "iou": []}

        for episode in all_episode_data:
            layer_stats = episode["layer_stats"]
            if layer_key not in layer_stats:
                continue

            stats = layer_stats[layer_key]
            outcome = episode["outcome"]
            target = success_metrics if outcome == "success" else failure_metrics

            target["overlap"].append(stats.get("overlap_ratio_mean", 0))
            target["concentration"].append(stats.get("concentration_mean", 0))
            target["iou"].append(stats.get("iou_mean", 0))

        layer_aggregates[layer] = {
            "success": success_metrics,
            "failure": failure_metrics,
        }

    # Generate comprehensive visualizations
    print("\nGenerating visualizations...")

    # 1. Multi-panel comparison plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    metrics = ["overlap", "concentration", "iou"]
    metric_names = ["Overlap Ratio", "Attention Concentration", "IoU"]
    colors = {"success": "steelblue", "failure": "coral"}

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]

        # Prepare data for plotting
        x_pos = np.arange(len(layers))
        width = 0.35

        success_means = []
        success_stds = []
        failure_means = []
        failure_stds = []

        for layer in layers:
            agg = layer_aggregates[layer]

            success_vals = agg["success"][metric]
            failure_vals = agg["failure"][metric]

            success_means.append(np.mean(success_vals) if success_vals else 0)
            success_stds.append(np.std(success_vals) if success_vals else 0)
            failure_means.append(np.mean(failure_vals) if failure_vals else 0)
            failure_stds.append(np.std(failure_vals) if failure_vals else 0)

        # Plot bars
        ax.bar(
            x_pos - width / 2,
            success_means,
            width,
            yerr=success_stds,
            label="Success",
            color=colors["success"],
            alpha=0.8,
            capsize=5,
        )
        ax.bar(
            x_pos + width / 2,
            failure_means,
            width,
            yerr=failure_stds,
            label="Failure",
            color=colors["failure"],
            alpha=0.8,
            capsize=5,
        )

        ax.set_ylabel(metric_name, fontsize=12, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"L{l}" for l in layers])
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)

        # Add episode count annotation
        if idx == 0:
            ax.text(
                0.02,
                0.98,
                f"Success: {episode_count['success']} episodes | Failure: {episode_count['failure']} episodes",
                transform=ax.transAxes,
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

    axes[-1].set_xlabel("Layer", fontsize=12, fontweight="bold")
    plt.suptitle(
        "Multi-Episode Attention-Object Correlation Analysis\n(Success vs Failure)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    comparison_path = output_dir / "multi_episode_comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Comparison plot: {comparison_path}")

    # 2. Distribution plots (violin plots)
    fig, axes = plt.subplots(len(metrics), len(layers), figsize=(18, 10))

    for metric_idx, metric in enumerate(metrics):
        for layer_idx, layer in enumerate(layers):
            ax = axes[metric_idx, layer_idx]
            agg = layer_aggregates[layer]

            data_to_plot = []
            labels = []

            for outcome in ["success", "failure"]:
                vals = agg[outcome][metric]
                if vals:
                    data_to_plot.append(vals)
                    labels.append(outcome.capitalize())

            if data_to_plot:
                parts = ax.violinplot(
                    data_to_plot, positions=range(len(data_to_plot)), showmeans=True, showmedians=True
                )

                # Color the violins
                for pc, outcome in zip(parts["bodies"], ["success", "failure"][: len(data_to_plot)]):
                    pc.set_facecolor(colors[outcome])
                    pc.set_alpha(0.7)

                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, fontsize=8)

            ax.set_title(f"L{layer}", fontsize=10)
            ax.grid(axis="y", alpha=0.3)

            if layer_idx == 0:
                ax.set_ylabel(metric_names[metric_idx], fontsize=10)

    plt.suptitle("Distribution of Metrics Across Episodes", fontsize=14, fontweight="bold")
    plt.tight_layout()

    distribution_path = output_dir / "multi_episode_distributions.png"
    plt.savefig(distribution_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Distribution plot: {distribution_path}")

    # 3. Heatmap: Layer x Outcome
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]

        # Create matrix: rows=outcomes, cols=layers
        matrix = np.zeros((2, len(layers)))

        for layer_idx, layer in enumerate(layers):
            agg = layer_aggregates[layer]
            success_vals = agg["success"][metric]
            failure_vals = agg["failure"][metric]

            matrix[0, layer_idx] = np.mean(success_vals) if success_vals else 0
            matrix[1, layer_idx] = np.mean(failure_vals) if failure_vals else 0

        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([f"L{l}" for l in layers])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Success", "Failure"])
        ax.set_title(metric_name, fontsize=12, fontweight="bold")

        # Add text annotations
        for i in range(2):
            for j in range(len(layers)):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="black", fontsize=9)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Heatmap: Attention-Object Metrics by Outcome", fontsize=14, fontweight="bold")
    plt.tight_layout()

    heatmap_path = output_dir / "multi_episode_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Heatmap: {heatmap_path}")

    # 4. Generate summary report
    report_lines = [
        "# Multi-Episode Attention-Object Correlation Report\n",
        f"## Dataset: {results_root.name}\n",
        f"**Total Episodes**: {total_episodes}\n",
        f"- Success: {episode_count['success']}\n",
        f"- Failure: {episode_count['failure']}\n",
        f"\n**Layers Analyzed**: {layers}\n",
        "\n## Summary Statistics\n",
    ]

    for layer in layers:
        report_lines.append(f"\n### Layer {layer}\n")
        agg = layer_aggregates[layer]

        for outcome in ["success", "failure"]:
            report_lines.append(f"\n**{outcome.capitalize()}**:\n")
            for metric, metric_name in zip(metrics, metric_names):
                vals = agg[outcome][metric]
                if vals:
                    mean = np.mean(vals)
                    std = np.std(vals)
                    median = np.median(vals)
                    report_lines.append(
                        f"- {metric_name}: {mean:.4f} ± {std:.4f} (median: {median:.4f}, n={len(vals)})\n"
                    )
                else:
                    report_lines.append(f"- {metric_name}: No data\n")

    # Key findings
    report_lines.append("\n## Key Findings\n")

    # Find best layers
    best_layers = {"overlap": None, "concentration": None, "iou": None}
    for metric in metrics:
        max_val = 0
        best_layer = None
        for layer in layers:
            success_vals = layer_aggregates[layer]["success"][metric]
            if success_vals:
                mean_val = np.mean(success_vals)
                if mean_val > max_val:
                    max_val = mean_val
                    best_layer = layer
        best_layers[metric] = (best_layer, max_val)

    report_lines.append(f"\n**Best Performing Layers (Success Episodes)**:\n")
    for metric, metric_name in zip(metrics, metric_names):
        layer, val = best_layers[metric]
        report_lines.append(f"- {metric_name}: Layer {layer} ({val:.4f})\n")

    # Success vs Failure comparison
    report_lines.append(f"\n**Success vs Failure Comparison**:\n")
    for layer in layers:
        report_lines.append(f"\nLayer {layer}:\n")
        for metric, metric_name in zip(metrics, metric_names):
            success_vals = layer_aggregates[layer]["success"][metric]
            failure_vals = layer_aggregates[layer]["failure"][metric]

            if success_vals and failure_vals:
                success_mean = np.mean(success_vals)
                failure_mean = np.mean(failure_vals)
                diff = success_mean - failure_mean
                diff_pct = (diff / success_mean * 100) if success_mean != 0 else 0

                report_lines.append(f"  - {metric_name}: Success={success_mean:.4f}, Failure={failure_mean:.4f}, ")
                report_lines.append(f"Δ={diff:+.4f} ({diff_pct:+.1f}%)\n")

    report_path = output_dir / "multi_episode_report.md"
    with open(report_path, "w") as f:
        f.writelines(report_lines)
    print(f"  ✓ Report: {report_path}")

    # Save aggregated data as JSON
    aggregated_json = {
        "total_episodes": total_episodes,
        "episode_count": episode_count,
        "layers": list(layers),  # Convert range to list for JSON serialization
        "layer_aggregates": {
            str(layer): {
                outcome: {
                    metric: {
                        "values": layer_aggregates[layer][outcome][metric],
                        "mean": float(np.mean(layer_aggregates[layer][outcome][metric]))
                        if layer_aggregates[layer][outcome][metric]
                        else 0,
                        "std": float(np.std(layer_aggregates[layer][outcome][metric]))
                        if layer_aggregates[layer][outcome][metric]
                        else 0,
                        "count": len(layer_aggregates[layer][outcome][metric]),
                    }
                    for metric in metrics
                }
                for outcome in ["success", "failure"]
            }
            for layer in layers
        },
    }

    json_path = output_dir / "multi_episode_aggregate.json"
    with open(json_path, "w") as f:
        json.dump(aggregated_json, f, indent=2)
    print(f"  ✓ JSON data: {json_path}")

    print(f"\n{'=' * 60}")
    print("AGGREGATION COMPLETE")
    print(f"{'=' * 60}\n")

    return aggregated_json


@timer
def main():
    # if len(sys.argv) < 2:
    #     print("Usage: python viz/object_pipeline.py <DATA_ROOT>")
    #     print("Example: python viz/object_pipeline.py /data3/tonyw/aawr_offline/dual/")
    #     sys.exit(1)
    # TEST_CASE = "/data3/tonyw/aawr_offline/gold/"
    TEST_CASE = "/data3/tonyw/aawr_offline/dual/"
    DATA_ROOT = Path(TEST_CASE)
    RESULTS_ROOT = Path("/data3/tonyw/aawr_offline/pi05/") / DATA_ROOT.name / CAMERA

    print(f"Data root: {DATA_ROOT}")
    print(f"Results root: {RESULTS_ROOT}")
    print(f"Camera: {CAMERA}")
    print(f"Layers: {LAYERS}")
    print()

    # Load Policy
    checkpoint_dir = "./checkpoints/viz/pi05_droid_pytorch"
    print(f"Loading policy from {checkpoint_dir}...")
    policy = attn_map.get_policy(checkpoint_dir)
    print("Policy loaded.\n")

    # Statistics
    total_episodes = 0
    processed_episodes = 0
    skipped_episodes = 0
    error_episodes = 0

    for outcome in ["success", "failure"]:
        outcome_dir = DATA_ROOT / outcome
        if not outcome_dir.exists():
            print(f"Skipping outcome: {outcome} (directory not found)")
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing outcome: {outcome.upper()}")
        print(f"{'=' * 60}\n")

        for date_dir in sorted(outcome_dir.iterdir()):
            if not date_dir.is_dir():
                continue

            for h5_path in sorted(date_dir.rglob("trajectory.h5")):
                data_dir = h5_path.parent
                episode_id = data_dir.name
                total_episodes += 1

                print(f"\n[{total_episodes}] Episode: {outcome}/{date_dir.name}/{episode_id}")

                # Setup Output Dir: results_object/{dataset_name}/{camera}/{outcome}/{date}/{episode}
                rel_path = data_dir.relative_to(DATA_ROOT)
                episode_dir = RESULTS_ROOT / rel_path
                episode_dir.mkdir(parents=True, exist_ok=True)

                # Skip if already processed
                marker_file = episode_dir / "object_detection.md"
                if marker_file.exists():
                    print(f"  ✓ Already processed, skipping")
                    skipped_episodes += 1
                    continue

                # Check if object masks exist
                white_list = get_white_list_frames(data_dir)
                if not white_list:
                    print(f"  ✗ No white list found, skipping")
                    skipped_episodes += 1
                    continue

                print(f"  White list: {len(white_list)} frames with object detections")

                # Determine total frames
                total_frames = get_video_length(data_dir)
                if total_frames == 0:
                    print(f"  ✗ No video frames found, skipping")
                    skipped_episodes += 1
                    continue

                # Use white list frames (they already have object detections)
                keyframes = white_list
                print(f"  Processing {len(keyframes)} keyframes: {keyframes[:5]}{'...' if len(keyframes) > 5 else ''}")

                all_results = {}
                successful_frames = 0

                # Process Keyframes
                for frame_idx in keyframes:
                    try:
                        print(f"\n  Frame {frame_idx}:")

                        # Load example
                        example = load_toy_example(data_dir, frame_idx, camera=CAMERA)

                        # Run inference to generate attention maps
                        print(f"    Running inference...")
                        _ = policy.infer(example)

                        # Run object detection analysis
                        print(f"    Analyzing attention-object correlation...")
                        frame_results = run_object_detection(
                            policy,
                            example,
                            frame_idx,
                            str(data_dir),
                            episode_dir,
                            layers=LAYERS,
                            camera=CAMERA,
                        )

                        if frame_results:
                            all_results[frame_idx] = frame_results
                            successful_frames += 1

                            # Print brief summary
                            for layer_key, layer_data in frame_results.items():
                                if "aggregate_metrics" in layer_data:
                                    layer_idx = layer_key.split("_")[1]
                                    metrics_list = list(layer_data["aggregate_metrics"].values())
                                    if metrics_list:
                                        import numpy as np

                                        avg_overlap = np.mean([m["overlap_ratio"] for m in metrics_list])
                                        print(f"      Layer {layer_idx}: Overlap={avg_overlap:.3f}")

                    except FileNotFoundError as e:
                        print(f"    ✗ File not found: {e}")
                    except Exception as e:
                        print(f"    ✗ Error: {e}")
                        import traceback

                        traceback.print_exc()

                # Generate summary if we have results
                if all_results:
                    print(f"\n  Generating summary for {len(all_results)} frames...")

                    # Generate summary plot
                    summary_plot_path = episode_dir / "h1_1_obj_attn_summary.png"
                    layer_stats = generate_summary_plot(all_results, str(summary_plot_path), LAYERS)

                    # Save JSON results
                    results_json_path = episode_dir / "h1_1_obj_attn_results.json"

                    def convert_to_native(obj):
                        import numpy as np

                        if isinstance(obj, np.integer):
                            return int(obj)
                        if isinstance(obj, np.floating):
                            return float(obj)
                        if isinstance(obj, np.ndarray):
                            return obj.tolist()
                        if isinstance(obj, dict):
                            return {key: convert_to_native(value) for key, value in obj.items()}
                        if isinstance(obj, list):
                            return [convert_to_native(item) for item in obj]
                        return obj

                    with open(results_json_path, "w") as f:
                        json.dump(
                            {
                                "frame_results": convert_to_native(all_results),
                                "layer_statistics": convert_to_native(layer_stats),
                            },
                            f,
                            indent=2,
                        )

                    print(f"    ✓ Summary plot: {summary_plot_path}")
                    print(f"    ✓ JSON results: {results_json_path}")

                    # Generate videos
                    print(f"  Generating videos...")
                    video_paths = create_all_layer_videos(episode_dir, layers=LAYERS, fps=FPS_VIDEO, add_text=True)
                    if video_paths:
                        print(f"    ✓ Created {len(video_paths)} videos in {episode_dir / 'videos'}")

                    # Copy instruction
                    copy_instruction(data_dir, episode_dir)

                    # Create marker file
                    marker_file.write_text(
                        f"# Object Detection Analysis Complete\n\n"
                        f"Episode: {episode_id}\n"
                        f"Outcome: {outcome}\n"
                        f"Date: {date_dir.name}\n"
                        f"Total Frames: {total_frames}\n"
                        f"Keyframes Processed: {successful_frames}/{len(keyframes)}\n"
                        f"Keyframes: {keyframes}\n"
                        f"Layers: {LAYERS}\n"
                        f"Camera: {CAMERA}\n"
                    )

                    processed_episodes += 1
                    print(f"  ✓ Episode processed successfully ({successful_frames} frames)")
                else:
                    print(f"  ✗ No results generated")
                    error_episodes += 1

    # Final summary
    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total episodes found: {total_episodes}")
    print(f"Successfully processed: {processed_episodes}")
    print(f"Skipped (already done): {skipped_episodes}")
    print(f"Errors: {error_episodes}")
    print(f"\nResults saved to: {RESULTS_ROOT}")

    # Generate multi-episode aggregated analysis
    if processed_episodes > 0 or skipped_episodes > 0:
        print("\n" + "=" * 60)
        print("Generating multi-episode aggregated analysis...")
        print("=" * 60)
        aggregate_episodes_analysis(RESULTS_ROOT, LAYERS, output_dir=RESULTS_ROOT)


if __name__ == "__main__":
    main()
