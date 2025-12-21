from __future__ import annotations
import json, logging, os
from pathlib import Path
import cv2
import numpy as np
import sys
import time
import functools
import shutil
import glob

from PIL import Image

import attn_map
import combine_video

from attn_map import visualize_attention, visualize_heads, get_keyframes, process_episode

OPEN_LOOP_HORIZON = 8


# # Ensure src is in pythonpath
# sys.path.append(os.path.join(os.getcwd(), "src"))
# # Ensure current dir (viz) is in pythonpath if running from root
# if os.getcwd() not in sys.path:
#     sys.path.append(os.getcwd())


# except ImportError:
#     try:
#         from viz import attn_map
#         from viz import combine_video
#     except ImportError:
#         # Fallback if running from inside viz directory
#         sys.path.append(str(Path(__file__).parent))
#         import attn_map
#         import combine_video

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


def load_toy_example(data_dir: Path, index: int, camera: str = "left"):
    if camera == "left":
        side_camera = "varied_camera_1"
    elif camera == "right":
        side_camera = "varied_camera_2"
    else:
        raise ValueError("camera must be 'left' or 'right'")

    ext_path = os.path.join(data_dir, "recordings", "frames", side_camera, f"{index:05d}.jpg")
    hand_path = os.path.join(data_dir, "recordings", "frames", "hand_camera", f"{index:05d}.jpg")

    ext_img = np.array(Image.open(ext_path))
    hand_img = np.array(Image.open(hand_path))

    instruction_path = os.path.join(data_dir, "instruction.txt")
    with open(instruction_path, "r") as f:
        instruction = f.read().strip()
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


@timer
def main():
    if len(sys.argv) < 2:
        print("Usage: python viz/pipeline.py <DATA_ROOT>")
        sys.exit(1)

    DATA_ROOT = Path(sys.argv[1])
    RESULTS_ROOT = Path("results_toy")

    # Load Policy
    checkpoint_dir = "./checkpoints/viz/pi05_droid_pytorch"
    print(f"Loading policy from {checkpoint_dir}...")
    policy = attn_map.get_policy(checkpoint_dir)
    print("Policy loaded.")

    for outcome in ["success", "failure"]:
        outcome_dir = DATA_ROOT / outcome

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
                        example = load_toy_example(data_dir, index, camera="left")
                        # Create subfolder for timestep: results_toy/.../episode/{index}
                        attn_map.process_episode(policy, example, str(output_base_dir), f"{index:05d}")
                    except Exception as e:
                        print(f"Error processing frame {index}: {e}")
                
                # Create a marker file to indicate processing is complete
                marker_file = output_base_dir / "pi05.md"
                marker_file.write_text(f"# Processing Complete\n\nEpisode: {episode_id}\nOutcome: {outcome}\nDate: {date_dir.name}\nTotal Frames: {total_frames}\nKeyframes: {keyframes}\n")

if __name__ == "__main__":
    main()
