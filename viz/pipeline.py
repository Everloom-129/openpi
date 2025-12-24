from __future__ import annotations

import functools
import glob
import logging
from pathlib import Path
import shutil
import sys
import time
import numpy as np

from attn_map import get_keyframes
from PIL import Image

import attn_map
import combine_video
from h1_mask_effect import run_fidelity_test


OPEN_LOOP_HORIZON = 8
LAYERS = [1, 4, 5, 7, 10]
FPS = 2
CREATE_SUMMARY_VIDEO = False
CREATE_HEAD_VIDEOS = False


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def timer(func):
    """A decorator that prints the execution time of the function it decorates."""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        logger.info(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value

    return wrapper_timer


def load_toy_example(data_dir: Path, index: int, camera: str = "right"):
    """
    Loads a toy example (images + instruction) from the given directory.
    Compatible with load_duck_example structure.
    """
    data_dir = Path(data_dir)

    if camera == "left":
        side_camera = "varied_camera_1"
    elif camera == "right":
        side_camera = "varied_camera_2"
    else:
        raise ValueError("camera must be 'left' or 'right'")

    # Use pathlib for path construction
    frames_dir = data_dir / "recordings" / "frames"
    ext_path = frames_dir / side_camera / f"{index:05d}.jpg"
    hand_path = frames_dir / "hand_camera" / f"{index:05d}.jpg"

    if not ext_path.exists():
        raise FileNotFoundError(f"Exterior image not found: {ext_path}")
    if not hand_path.exists():
        raise FileNotFoundError(f"Hand image not found: {hand_path}")

    ext_img = np.array(Image.open(ext_path))
    hand_img = np.array(Image.open(hand_path))

    instruction_path = data_dir / "instruction.txt"
    if instruction_path.exists():
        instruction = instruction_path.read_text().strip()
    else:
        print(f"Warning: Instruction file not found at {instruction_path}, using default.")
        instruction = "find the pineapple toy and pick it up"

    print(f"Instruction: {instruction}")

    return {
        "observation/exterior_image_1_left": ext_img,
        "observation/wrist_image_left": hand_img,
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": instruction,
    }


def get_video_length(data_dir: Path):
    hand_camera_dir = data_dir / "recordings" / "frames" / "hand_camera"
    if not hand_camera_dir.exists():
        return 0
    jpg_files = list(hand_camera_dir.glob("*.jpg"))
    return len(jpg_files)


def copy_instruction(data_dir: Path, output_dir: Path):
    instruction_path = data_dir / "instruction.txt"
    if not instruction_path.exists():
        return
    shutil.copy(instruction_path, output_dir / "instruction.txt")


@timer
def main():
    if len(sys.argv) < 2:
        print("Usage: python viz/pipeline.py <DATA_ROOT>")
        sys.exit(1)

    DATA_ROOT = Path(sys.argv[1])
    RESULTS_ROOT = Path("results_toy_right")

    # Load Policy
    checkpoint_dir = "./checkpoints/viz/pi05_droid_pytorch"
    print(f"Loading policy from {checkpoint_dir}...")
    policy = attn_map.get_policy(checkpoint_dir)
    print("Policy loaded.")

    for outcome in ["success", "failure"]:
        outcome_dir = DATA_ROOT / outcome
        if not outcome_dir.exists():
            continue

        for date_dir in outcome_dir.iterdir():
            if not date_dir.is_dir():
                continue

            for h5_path in date_dir.rglob("trajectory.h5"):
                data_dir = h5_path.parent
                episode_id = data_dir.name
                print(f"Processing episode: {outcome}/{date_dir.name}/{episode_id}")

                # Setup Output Dir
                # Structure: results_toy/success/date/episode
                rel_path = data_dir.relative_to(DATA_ROOT)
                output_base_dir = RESULTS_ROOT / rel_path
                output_base_dir.mkdir(parents=True, exist_ok=True)

                # Skip if already processed
                marker_file = output_base_dir / "pi05.md"
                if marker_file.exists():
                    print(f"Skipping episode {episode_id} (already processed)")
                    continue

                # Determine Frames
                total_frames = get_video_length(data_dir)
                if total_frames == 0:
                    print("No video frames found, skipping.")
                    continue

                # Action Horizon 8
                keyframes = get_keyframes(total_frames, OPEN_LOOP_HORIZON)

                # Process Keyframes
                for index in keyframes:
                    # Skip if already processed? (Optional optimization)
                    # if (output_base_dir / f"{index:05d}").exists(): continue

                    try:
                        example = load_toy_example(data_dir, index, camera="right")
                        # Create subfolder for timestep: results_toy/.../episode/{index}
                        # This generates the attention maps in results/layers_prefix
                        result = attn_map.process_episode(
                            policy, example, str(output_base_dir), f"{index:05d}", layers=LAYERS
                        )

                        # Run Fidelity Test immediately after inference (while attn maps exist)
                        print(f"  Running Fidelity Test for frame {index}...")
                        fidelity_results = run_fidelity_test(
                            policy,
                            example,
                            result["actions"],  # action_orig
                            output_dir=output_base_dir / f"{index:05d}" / "fidelity",
                            layers=LAYERS,
                            attn_root_dir="results",
                            save_vis=True,
                        )
                        if fidelity_results:
                            best = max(fidelity_results, key=lambda x: x["fidelity"])
                            print(f"  Fidelity Test Best: Layer {best['layer']} Score {best['fidelity']:.4f}")

                    except Exception as e:
                        print(f"Error processing frame {index}: {e}")

                # Create a marker file to indicate processing is complete
                # marker_file = output_base_dir / "pi05.md"
                marker_file.write_text(
                    f"# Processing Complete\n\nEpisode: {episode_id}\nOutcome: {outcome}\nDate: {date_dir.name}\nTotal Frames: {total_frames}\nKeyframes: {keyframes}\n"
                )

                for layer in LAYERS:
                    # Find images: output_base_dir/*/prefix_L{layer}_attn_vis_max.jpg
                    pattern = str(output_base_dir / "*" / f"prefix_L{layer}_attn_vis_max.jpg")
                    image_paths = sorted(glob.glob(pattern))

                    if not image_paths:
                        print(f"No images found for layer {layer}, skipping.")
                        continue
                    # Video output path: results_toy_video/{success/failure}/{date}/{episode}/L{layer}_prefix_max.mp4
                    video_output_dir = Path("results_toy_video_right") / rel_path
                    video_output_dir.mkdir(parents=True, exist_ok=True)

                    if CREATE_SUMMARY_VIDEO:
                        combine_video.create_summary_video(
                            layer,
                            fps=FPS,
                            mode="max",
                            output_dir=video_output_dir,
                            timesteps=keyframes,
                            input_dir=output_base_dir,
                        )
                    if CREATE_HEAD_VIDEOS:
                        for head in range(8):
                            combine_video.create_video_for_head(
                                layer,
                                head,
                                fps=FPS,
                                timesteps=keyframes,
                                output_dir=video_output_dir,
                                input_dir=output_base_dir,
                            )

                    copy_instruction(data_dir, video_output_dir)


if __name__ == "__main__":
    main()
