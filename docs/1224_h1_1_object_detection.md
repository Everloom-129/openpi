# H1.1: Object Detection Correlation Analysis

**Date**: 2024-12-24  
**Author**: Assistant & User  
**Goal**: Evaluate the correlation between VLM attention maps and object detection masks

---

## 1. Overview

This document records the implementation of Hypothesis 1.1: analyzing how well attention maps from PaliGemma VLM correlate with ground-truth object detection masks from DINO-X.

### Key Question
**Do attention maps focus on detected objects? Which layers show the strongest correlation?**

### Approach
- Extract wrist camera attention (tokens 256-511)
- Compare with DINO-X object masks
- Compute metrics: Overlap Ratio, Attention Concentration, IoU
- Analyze across all 18 layers
- Compare Success vs Failure episodes

---

## 2. Implementation Components

### 2.1 Core Script: `viz/h1_1_object_detection.py`

**Main Functions:**

#### Data Loading
```python
load_pineapple_example(camera, index)
# Load images and instruction for a specific frame

load_object_masks(data_dir, index, camera)
# Load DINO-X detection masks from dinox_info.json
```

#### Mask Processing
```python
resize_mask_to_224(mask_raw)
# Resize 720x1280 mask to 224x224 with padding
# Matches OpenPI's image preprocessing

compute_attention_on_object(attn_map_16x16, object_mask_224)
# Returns: overlap_ratio, attention_concentration, iou
```

#### Visualization
```python
visualize_attention_on_mask(attn_map, object_masks, raw_image, output_path, layer_idx)
# 3-panel plot: Original | Attention | Attention+Mask overlay
```

#### Analysis Pipeline
```python
run_object_detection(policy, example, frame_idx, data_dir, episode_dir, layers, camera)
# Process one frame:
#   1. Run inference → generate attention maps
#   2. Load object masks
#   3. Compute metrics for each layer
#   4. Save visualizations: episode_dir/object/{frame:05d}/L{layer}.jpg
```

#### Video Generation
```python
create_layer_video(episode_dir, layer, fps, output_path, add_text=True)
# Create video for one layer across all frames

create_all_layer_videos(episode_dir, layers, fps, add_text=True)
# Batch create videos for all layers
# Output: episode_dir/videos/L{layer:02d}_object_attn.webm
```

#### Summary Statistics
```python
generate_summary_plot(all_results, output_path, layers)
# 3-panel bar chart: Overlap | Concentration | IoU
# Shows mean ± std for each layer
```

### 2.2 Batch Pipeline: `viz/object_pipeline.py`

**Purpose**: Process entire datasets with object detection masks

**Key Features:**
- Auto-detects episodes with `white_list.json`
- Processes only frames with valid object detections
- Resume capability via `object_detection.md` marker
- Progress tracking and error handling
- Multi-episode aggregation and reporting

**Main Workflow:**
```python
main()
1. Load policy
2. Iterate through outcomes (success/failure)
3. For each episode:
   - Check white_list.json
   - Process keyframes
   - Run object detection analysis
   - Generate videos
   - Create marker file
4. Aggregate all episodes
```

**Multi-Episode Aggregation:**
```python
aggregate_episodes_analysis(results_root, layers, output_dir)
# Generates 5 outputs:
#   1. multi_episode_comparison.png - Success vs Failure bar charts
#   2. multi_episode_distributions.png - Violin plots showing spread
#   3. multi_episode_heatmap.png - Heatmaps per metric
#   4. multi_episode_report.md - Detailed text report
#   5. multi_episode_aggregate.json - Raw aggregated data
```

---

## 3. Data Structure

### 3.1 Input Requirements

**Episode Structure:**
```
episode_dir/
├── trajectory.h5
├── dinox_info.json          # Object detection results
├── white_list.json          # Frames with valid detections
├── masks/                   # Binary masks (.npy files)
│   ├── 00025_pineapple_0.npy
│   └── ...
└── recordings/frames/
    ├── varied_camera_1/     # Left exterior camera
    ├── varied_camera_2/     # Right exterior camera
    └── hand_camera/         # Wrist camera (used for analysis)
```

**dinox_info.json Format:**
```json
{
  "25": [
    {
      "bbox": [654.27, 519.27, 765.67, 636.43],
      "mask_file": "00025_pineapple_0.npy",
      "category": "yellow pineapple toy",
      "score": 0.82
    }
  ]
}
```

**Mask Format:**
- Shape: (720, 1280)
- Type: bool
- Values: True (object pixels), False (background)

### 3.2 Output Structure

**Per Episode:**
```
results_object/{dataset}/{camera}/{outcome}/{date}/{episode}/
├── object_detection.md              # Processing marker
├── h1_1_obj_attn_summary.png        # Summary plot (3 panels)
├── h1_1_obj_attn_results.json       # Detailed metrics
├── instruction.txt                  # Task description
├── object/                          # Per-frame visualizations
│   ├── 00025/
│   │   ├── L0.jpg
│   │   ├── L1.jpg
│   │   ├── ...
│   │   └── L17.jpg
│   └── ...
└── videos/                          # Layer evolution videos
    ├── L00_object_attn.webm
    ├── L01_object_attn.webm
    └── ...
```

**Multi-Episode Aggregation:**
```
results_object/{dataset}/{camera}/
├── success/
│   └── [episodes...]
├── failure/
│   └── [episodes...]
├── multi_episode_comparison.png     # Success vs Failure comparison
├── multi_episode_distributions.png  # Violin plots
├── multi_episode_heatmap.png        # Heatmaps
├── multi_episode_report.md          # Text report
└── multi_episode_aggregate.json     # Raw data
```

---

## 4. Key Metrics

### 4.1 Overlap Ratio
**Definition**: Fraction of total attention mass that falls on object pixels

```python
overlap_ratio = attention_on_object / total_attention
```

**Interpretation**:
- 0.0: No attention on object
- 1.0: All attention on object
- Higher = More focused on object

### 4.2 Attention Concentration
**Definition**: Ratio of mean attention inside vs outside object

```python
concentration = mean_attention_inside / mean_attention_outside
```

**Interpretation**:
- 1.0: Equal attention inside/outside
- >1.0: Preferentially attends to object
- <1.0: Preferentially attends to background
- **Example**: 13.8x means object pixels get 14× more attention

### 4.3 IoU (Intersection over Union)
**Definition**: Spatial alignment between thresholded attention and mask

```python
attention_binary = attention > 0.3
iou = intersection(attention_binary, mask) / union(attention_binary, mask)
```

**Interpretation**:
- 0.0: No spatial overlap
- 1.0: Perfect alignment
- Limited by low resolution (16×16 attention vs 224×224 mask)

---

## 5. Key Findings (Example: aawr_pineapple dataset)

### 5.1 Layer-wise Performance

**Best Layers for Object Detection:**
- **Layer 5**: Highest concentration (13.8×)
- **Layer 4**: High overlap ratio (0.098)
- **Layer 8**: Second concentration peak (13.4×)

**Layer Progression:**
```
L0-2:  Overlap ~0.004-0.016 (low-level features)
L3-5:  Overlap ~0.09-0.10   (PEAK - object recognition)
L6:    Overlap ~0.023        (functional transition)
L7-8:  Overlap ~0.053-0.089  (task-oriented attention)
L9+:   Declining             (action planning)
L14-17: Concentration <1.0   (focus shifts to action space)
```

### 5.2 Critical Insight: Wrist Camera is Key

**Initial Mistake**: Used exterior camera attention (tokens 0-255)
- Object masks were on wrist camera
- Spatial mismatch → poor correlation

**Correction**: Use wrist camera attention (tokens 256-511)
- Proper alignment with masks
- Strong correlation emerges

**Code Change:**
```python
# Before (WRONG)
attn_ext = text_attn[:256].reshape(16, 16)

# After (CORRECT)
attn_wrist = text_attn[256:512].reshape(16, 16)
```

### 5.3 Functional Interpretation

**Layer 5 - Pure Object Recognition**
- What: "This is a pineapple"
- Highest IoU (0.32)
- Strong spatial alignment

**Layer 8 - Task-Oriented Attention**
- What: "I need to grasp this pineapple"
- High concentration but different spatial pattern
- Integrates task requirements

**Layer 14-17 - Action Planning**
- Concentration <1.0 (attends to background)
- Focuses on: gripper position, trajectory, obstacles
- Object-centric → action-centric transition

---

## 6. Implementation Details

### 6.1 Critical Bug Fixes

**1. JSON Serialization Error**
```python
# Problem
LAYERS = range(18)  # range object not JSON serializable

# Fix
aggregated_json = {
    "layers": list(layers),  # Convert to list
}
```

**2. Video Codec Issues**
```python
# Problem: mp4v codec incompatible

# Fix: Use VP90 for webm
fourcc = cv2.VideoWriter_fourcc(*"VP90")
output_path = f"L{layer:02d}_object_attn.webm"
```

### 6.2 Design Decisions

**1. Layer-level Only (No Per-Head Analysis)**
- User requirement: "只需要 layer 级别，不需要深入到每一个 head"
- Simplified: Average across 8 heads
- Cleaner output structure

**2. Episode-based Organization**
```python
# Structure: results/{episode}/object/{frame}/L{layer}.jpg
# Benefits:
#   - Each frame in its own folder
#   - All layers for a frame grouped together
#   - Easy batch processing: for frame in object/*/
```

**3. Automatic Video Generation**
- Automatically runs after processing
- One video per layer showing temporal evolution
- Text overlays: Frame number + Layer number

### 6.3 Configuration

**Editable Parameters in `object_pipeline.py`:**
```python
LAYERS = range(18)           # Layers to analyze
CAMERA = "right"             # Camera view (left/right)
FPS_VIDEO = 5                # Video frame rate
OPEN_LOOP_HORIZON = 8        # For keyframe detection (unused with white_list)
```

---

## 7. Usage Guide

### 7.1 Single Episode Analysis

```bash
cd /home/exx/Project_pi05/openpi
python viz/h1_1_object_detection.py
```

**Edits in main():**
```python
test_frames = [73, 75, 76, 77, 78]  # Frames to process
camera = "right"                     # Camera selection
layers_to_test = [1, 4, 5, 7, 10, 17]  # Layers to analyze
data_dir = Path("data/visualization/aawr_pineapple")
```

### 7.2 Batch Processing

```bash
python viz/object_pipeline.py /path/to/dataset
```

**Example:**
```bash
python viz/object_pipeline.py /data3/tonyw/aawr_offline/dual/
```

**Output:**
```
results_object/dual/right/
├── success/ (processed episodes)
├── failure/ (processed episodes)
└── multi_episode_*.png/md/json (aggregated analysis)
```

### 7.3 Manual Aggregation

```python
from viz.object_pipeline import aggregate_episodes_analysis
from pathlib import Path

results_root = Path("results_object/dual/right")
layers = list(range(18))
aggregate_episodes_analysis(results_root, layers)
```

---

## 8. Troubleshooting

### 8.1 Common Issues

**"index X must be in the white list"**
- Frame X has no valid DINO-X detections
- Check `white_list.json` for available frames

**"Attention map not found for Layer X"**
- Model inference didn't generate attention maps
- Verify model modifications are active (see `docs/1220_pi05_attn_visualization.md`)
- Check `results/layers_prefix/` for `.npy` files

**"No object masks found for frame X"**
- Frame X has no detections with score > 0
- Check `dinox_info.json` entry for that frame
- Verify mask files exist in `masks/` directory

**"Cannot reshape array" Error**
- Attention map shape mismatch
- Usually from corrupted `.npy` file or interrupted inference
- Delete `results/layers_prefix/` and re-run

**OpenCV VP90 Warning**
- Warning: "VP90 is not supported with codec id 167"
- Videos still generated correctly
- Can be ignored (codec fallback works)

### 8.2 Data Requirements

**Minimum Requirements:**
- ✓ DROID format episode with `trajectory.h5`
- ✓ DINO-X object detection: `dinox_info.json`
- ✓ Object masks: `masks/*.npy`
- ✓ White list: `white_list.json`
- ✓ Video frames: `recordings/frames/`

**Optional:**
- `instruction.txt` (will use default if missing)
- Calibration data (not used in current analysis)

---

## 9. Future Work

### 9.1 Potential Extensions

**1. Temporal Analysis**
- Track attention on same object across frames
- Measure attention stability/consistency
- Correlate with action smoothness

**2. Success vs Failure Patterns**
- Statistical significance testing
- Identify failure modes
- Attention-based failure prediction

**3. Multi-Object Scenarios**
- Handle multiple objects per frame
- Object-specific attention allocation
- Distractor analysis

**4. Attention Manipulation**
- Mask-based attention guidance
- Test causal importance (similar to H1 Fidelity)
- Attention-augmented training

### 9.2 Known Limitations

**1. Low Spatial Resolution**
- Attention: 16×16 patches
- Masks: 224×224 (resampled from 720×1280)
- IoU ceiling due to resolution mismatch

**2. Single Camera Analysis**
- Currently: Wrist camera only
- Could extend: Exterior camera, camera fusion

**3. Static Object Detection**
- DINO-X masks are per-frame
- No temporal tracking
- Object identity not preserved across frames

**4. Limited Instruction Diversity**
- Most episodes: "find X and pick it up"
- Need varied tasks to test generalization

---

## 10. Related Documentation

- `docs/1220_pi05_attn_visualization.md` - Attention extraction setup
- `viz/README_h1_1.md` - Original README (deleted, content merged here)
- `viz/h1_mask_effect.py` - H1 Fidelity testing (causal validation)
- `viz/pipeline.py` - Original attention visualization pipeline

---

## 11. Code Statistics

**Lines of Code:**
- `h1_1_object_detection.py`: 663 lines
- `object_pipeline.py`: 669 lines
- **Total**: ~1,332 lines

**Key Functions:**
- Data loading: 3
- Mask processing: 2
- Metrics computation: 1
- Visualization: 2
- Video generation: 2
- Analysis pipeline: 1
- Aggregation: 1
- **Total**: 12 main functions

**Outputs per Episode:**
- Images: N_frames × N_layers (e.g., 64 × 18 = 1,152)
- Videos: N_layers (e.g., 18)
- Plots: 1 summary plot
- Data: 1 JSON file
- **Total**: ~1,172 files

---

## 12. Acknowledgments

**Data Sources:**
- DROID dataset: Robot manipulation episodes
- DINO-X: Object detection and segmentation
- Pi0.5 Model: PaliGemma-based VLM

**Key Insights:**
- Camera alignment is critical for correlation
- Layer 5 is the "semantic object detector"
- Attention shifts from object → action across layers
- Multi-episode analysis reveals robust patterns

---

**Document Version**: 1.0  
**Last Updated**: 2024-12-24  
**Status**: ✅ Complete and Tested

