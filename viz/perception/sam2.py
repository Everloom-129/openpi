"""SAM2 segmentation — local predictor only (no tiptop dependency)."""

import base64
import io
import logging
import os
from functools import cache
from pathlib import Path

import numpy as np
import requests
import torch.cuda
from PIL import Image
from jaxtyping import Float
from tqdm import tqdm

_log = logging.getLogger(__name__)

_SAM2_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"
_DEFAULT_CHECKPOINT = Path("checkpoints/viz/sam2.1_hiera_large.pt")
_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def download_sam2_checkpoint(
    model_name: str = "sam2.1_hiera_large.pt",
    dest_dir: Path = _DEFAULT_CHECKPOINT.parent,
) -> Path:
    """Download SAM2 checkpoint if it doesn't already exist."""
    model_url = os.path.join(_SAM2_BASE_URL, model_name)
    dest_path = dest_dir / model_name

    if dest_path.exists():
        _log.debug(f"SAM2 checkpoint {model_name} already exists at {dest_path}.")
        return dest_path

    dest_dir.mkdir(parents=True, exist_ok=True)
    _log.info(f"Downloading SAM2 checkpoint from {model_url} to {dest_path}.")
    response = requests.get(model_url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 KB

    with (
        open(dest_path, "wb") as file,
        tqdm(total=total_size, unit="iB", unit_scale=True, desc=model_name) as progress_bar,
    ):
        for data in response.iter_content(block_size):
            file.write(data)
            progress_bar.update(len(data))

    _log.info(f"SAM2 checkpoint {model_name} downloaded successfully.")
    return dest_path


@cache
def _sam2_predictor(checkpoint: str, device: str):
    """Load and cache the SAM2 image predictor."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    config = os.environ.get("SAM2_CONFIG", _SAM2_CONFIG)
    _log.info(f"Loading SAM2 with checkpoint={checkpoint}, config={config}, device={device}")
    predictor = SAM2ImagePredictor(build_sam2(config, checkpoint, device=device))
    _log.info("Successfully loaded SAM2")
    return predictor


def sam2_client(
    checkpoint: str | Path = _DEFAULT_CHECKPOINT,
    device: str | None = None,
) -> None:
    """Warm up SAM2: pre-load the local predictor."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _sam2_predictor(str(checkpoint), device)


def sam2_segment_objects(
    rgb_pil: Image.Image,
    detection_results: list[dict],
    checkpoint: str | Path = _DEFAULT_CHECKPOINT,
    device: str | None = None,
) -> Float[np.ndarray, "n 1 h w"]:
    """Segment detection results from Gemini with SAM2.

    Args:
        rgb_pil: PIL Image to segment.
        detection_results: List of detection dicts from Gemini, each with a 'box_2d' key
                           in [ymin, xmin, ymax, xmax] format normalized to 0-1000.
        checkpoint: Path to SAM2 checkpoint file.
        device: Torch device string. Defaults to 'cuda' if available, else 'cpu'.

    Returns:
        Segmentation masks of shape (N, 1, H, W).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert Gemini bbox format [ymin, xmin, ymax, xmax] (0-1000) to SAM2 [x0, y0, x1, y1] (pixels)
    img_height, img_width = rgb_pil.height, rgb_pil.width
    boxes = np.array([
        [
            (xmin / 1000.0) * img_width,
            (ymin / 1000.0) * img_height,
            (xmax / 1000.0) * img_width,
            (ymax / 1000.0) * img_height,
        ]
        for detection in detection_results
        if len(box_2d := detection.get("box_2d", [])) == 4
        for ymin, xmin, ymax, xmax in [box_2d]
    ])

    if len(boxes) == 0:
        h, w = rgb_pil.height, rgb_pil.width
        return np.zeros((0, 1, h, w), dtype=bool)

    predictor = _sam2_predictor(str(checkpoint), device)

    import torch
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        predictor.set_image(rgb_pil)
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )

    _log.info(f"Generated {len(masks)} segmentation masks, shape: {masks.shape}")

    if masks.ndim == 3:
        masks = masks[:, None]  # (N, H, W) → (N, 1, H, W)
    return masks
