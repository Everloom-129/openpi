"""
Standalone script to batch generate counterfactual videos for processed episodes.

This script can be run independently to generate videos for episodes that
already have counterfactual analysis results but are missing videos.

Usage:
    python viz/generate_cf_videos.py
"""

from pathlib import Path

from object_pipeline import (
    ANALYSIS_CAMERA,
    CAMERA,
    COUNTERFACTUAL_LAYERS,
    FPS_VIDEO,
    batch_generate_counterfactual_videos,
)


def main():
    """Generate counterfactual videos for all processed episodes."""
    # Configuration (should match object_pipeline.py)
    TEST_CASE = "/data3/tonyw/aawr_offline/dual/"
    DATA_ROOT = Path(TEST_CASE)
    RESULTS_ROOT = Path("/data3/tonyw/aawr_offline/pi05/") / DATA_ROOT.name / CAMERA

    print("=" * 80)
    print("Counterfactual Video Generation Script")
    print("=" * 80)
    print(f"Results root: {RESULTS_ROOT}")
    print(f"Layers: {list(COUNTERFACTUAL_LAYERS)}")
    print(f"FPS: {FPS_VIDEO}")
    print(f"Camera: {ANALYSIS_CAMERA}")
    print()

    # Run batch video generation
    stats = batch_generate_counterfactual_videos(RESULTS_ROOT, COUNTERFACTUAL_LAYERS, fps=FPS_VIDEO)

    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  Total episodes: {stats['total']}")
    print(f"  Processed: {stats['processed']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Errors: {stats['errors']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
