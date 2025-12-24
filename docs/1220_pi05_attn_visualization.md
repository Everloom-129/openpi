# Pi0.5 PaliGemma Attention Visualization 实践记录
2024-12-20

## 1. 目标
对 Pi0.5 (Droid) 模型中的 VLM Backbone (PaliGemma) 进行注意力机制可视化，以理解：
1. **Prefix Phase**: 模型在处理图像和文本 Prompt 时，文本 Token 关注图像的哪些区域。
2. **Suffix Phase**: 模型在去噪生成 Action 时，Action Token 关注图像的哪些区域。

## 2. 核心挑战与修改

为了提取 Attention Map，我们需要深入模型内部并绕过 JIT 编译的限制。

### 2.1 修改模型源码以提取 Attention
**文件**: `src/openpi/models_pytorch/gemma_pytorch.py`

Hugging Face 的 Transformer 默认不返回 Attention Weights。我们在 `forward` 函数中强制开启 `output_attentions=True`，并使用 Hack 方式将 Tensor 转存为 `.npy` 文件。

### 2.2 禁用 Torch Compile
**文件**: `src/openpi/models_pytorch/pi0_pytorch.py`

由于我们插入了 I/O 操作 (`numpy.save`) 且需要使用 `pdb` 调试，必须禁用 `torch.compile`。JIT 编译会隐藏中间变量并导致图捕获错误。

## 3. 工具脚本说明

我们开发了一套工具链用于分析 Attention Map。1

### 3.1 可视化生成 (`viz/attn_map.py`)
- **功能**: 加载模型推理，提取 Attention，生成可视化图片。
- **Keyframe Inference**: 针对 Open-Loop Control，只在关键帧 (t=0, 8, 16...) 进行推理和可视化。
- **两种模式**:
    - **Summary Overlay**: 叠加热力图到原图，快速预览。
    - **Detailed Heads**: 分离展示 Layer 的 8 个 Head，用于精细分析。


### 3.2 视频合成 (`viz/combine_video.py`)
- **功能**: 将多个关键帧的可视化图片合成为 `.webm` 或 `.avi` 视频。
- **用途**: 观察 Attention 随时间步的动态变化。
- **自动筛选**: 包含简单的方差筛选逻辑，自动推荐“最活跃”的 Head。

### 3.3 批量流水线处理 (`viz/pipeline.py`)
- **功能**: 针对大规模数据集（如 Toy Cube Benchmark）进行批量处理。
- **流程**:
    1. 遍历 `"/data3/tonyw/toy_cube_benchmark"` (DROID格式) 下的所有 Episode。
    2. 加载 `recordings` 中的视频帧（自动识别 Side/Wrist Camera）。
    3. 调用 `attn_map.py` 的核心逻辑生成 Attention Map 和可视化图片。
    4. 调用 `combine_video.py` 生成 Suffix/Prefix Attention 视频。
    5. **断点续传**: 通过检测 `pi05.md` 标记文件跳过已处理的 Episode。
- **输出结构**: 
    - 图片: `results_toy/{outcome}/{date}/{episode}/{frame_idx}/...`
    - 视频: `results_toy_video/{outcome}/{date}/{episode}/L{layer}_prefix_max.mp4`

## 4. Hypothesis and Validations


### 4.1 因果性验证 (`viz/h1_mask_effect.py`)
- **功能**: 通过遮挡实验验证 Attention Map 的真实性（Fidelity）。
    - **High Mask**: 遮挡 Attention 高亮区域。
    - **Low Mask**: 随机遮挡等面积的背景区域。
    - **Metric**: 计算 Action 的 Scale-Invariant MSE 变化。
- **多视角测试**: 支持分别测试 Both, Side Only, Wrist Only 视角的遮挡效果。
- **输出**: 自动生成 Markdown 报告，对比各层的因果性得分。

-  初步观察 (Observation)

基于少量样本的定性分析：

1.  **分层功能差异**:
    - **浅层 (Layer 0-2)**: 关注底层特征，遮挡后对输出影响极大（破坏性最强）。
    - **语义层 (Layer 10)**: 能够准确定位语义物体（如 "duck", "bowl"），表现出较强的双目融合特性。
    - **动作层 (Layer 17)**: 在输出前的最后一层，高度聚焦于 **Wrist Camera**，显示出手眼协调在动作生成阶段的主导地位。

2.  **视角依赖**:
    - 不同层级对 Side vs Wrist 视角的依赖程度不同。中间层偏向全局 Side View，深层偏向局部 Wrist View。

3.  **Scale vs Direction**:
    - 遮挡高关注区往往导致 Action 幅度（Scale）剧烈变化。
    - 遮挡背景区有时会显著干扰 Action 的轨迹形状（Direction），暗示背景包含重要的定位信息。


## 5. Results 数据结构

### 5.1 Input data 

在SSD硬盘上，我们以`/data3/tonyw/toy_cube_benchmark/` 为例

该数据集是 DROID 格式的机器人操作数据集，用于 Toy Cube 任务的 Benchmark。

#### 整体统计
- **总大小**: ~13 GB
- **Episode 总数**: 82 个
  - **Success**: 包含成功完成任务的 Episode
  - **Failure**: 包含失败的 Episode
- **时间跨度**: 2025-12-10 至 2025-12-13

#### 目录层级结构

```
toy_cube_benchmark/
├── success/                    # 成功的 Episode
│   ├── 2025-12-10/            # 按日期组织
│   ├── 2025-12-11/
│   ├── 2025-12-12/
│   └── 2025-12-13/
└── failure/                    # 失败的 Episode
    ├── 2025-12-10/
    ├── 2025-12-11/
    └── 2025-12-13/
```

#### Episode 级别结构

每个 Episode 以时间戳命名（格式：`YYYY-MM-DD_HH-MM-SS`），包含以下内容：

```
{outcome}/{date}/{episode_timestamp}/
├── instruction.txt                                    
├── calibration.json                                   
├── trajectory.h5                                      # HDF5 format, mainly used
├── trajectory.npz                                     # NumPy copy, do not use it
├── success_pi05_fm_{task_name}_{timestamp}.mp4        # a combined video of whole episode
└── recordings/                                        # raw recording data, do not use svo2 files here
    ├── 14436910.svo2                                 
    ├── 25455306.svo2                                 
    ├── 26368109.svo2                                
    ├── hand_camera.mp4                               # wrist camera video
    ├── varied_camera_1.mp4                           # left camera video
    ├── varied_camera_2.mp4                           # right camera video
    └── frames/                                        # extracted video frames, use it for attn vis
```

#### 文件说明

1. **`instruction.txt`**: 
   - natural language task instruction
   - example: `"place cube with G into the purple bowl"`

2. **`calibration.json`**: 
   - contains all camera intrinsic and extrinsic calibration data
   - camera IDs: `14436910`, `25455306`, `26368109`
   - each camera has `left` and `right` two views (stereo)
   - contains `intrinsics` (3x3 intrinsic matrix) and `extrinsics` (6D pose)

3. **`trajectory.h5`**: 
   - HDF5 format stored complete trajectory data
   - **Action data** (181 timesteps):
     - `cartesian_position`: (181, 6) - end-effector pose
     - `cartesian_velocity`: (181, 6) - end-effector velocity
     - `gripper_position`: (181,) - gripper position
     - `gripper_delta`: gripper delta control
     - `joint_position`, `joint_velocity`: joint space data
     - `robot_state`, `controller_info`, `timestamp`
   - **Observation data**:
     - `camera_extrinsics`, `camera_intrinsics`: camera parameters
     - `camera_type`: contains 3 camera IDs (`14436910`, `25455306`, `26368109`)
     - `robot_state`, `timestamp` - robot state and timestamp

4. **`recordings/frames/`**:
   - extracted JPEG image sequence from video
   - naming format: `00000.jpg` - `00180.jpg` (5 zeros padding)
   - 3 views: `hand_camera` (Wrist), `varied_camera_1/2` (Side)
   - each Episode has approximately 181 frames (corresponding to trajectory length)

#### camera configuration

- **Hand Camera**: wrist camera, fixed on the end-effector, provides a local fine-grained view
- **Varied Camera 1/2**: side camera, provides a global scene view
- **stereo system**: each ZED camera has Left/Right two views
- **resolution**: approximately 1280x720 (inferred from intrinsic matrix)


### 5.2 Attention Map Visualization data 

在当前目录下，我们以`results_toy_right/` 为例

该目录存储了从模型推理中提取的 Attention Map 可视化图片（静态帧）。

#### 整体统计
- **总大小**: ~973 MB
- **Episode 总数**: 32 个
- **图片文件总数**: 20,975 张 `.jpg` 文件
- **处理策略**: 仅处理关键帧 (Keyframe Inference, 间隔 8 帧)

#### 目录层级结构

```
results_toy_right/
├── success/                    # 成功的 Episode
│   ├── 2025-12-10/            # 按日期组织
│   ├── 2025-12-11/
│   └── 2025-12-13/
└── failure/                    # 失败的 Episode
    ├── 2025-12-10/
    └── 2025-12-13/
```

#### Episode 级别结构

每个 Episode 包含多个关键帧目录和一个处理标记文件：

```
{outcome}/{date}/{episode_timestamp}/
├── pi05.md                                    # 处理完成标记文件
├── 00000/                                     # 第 0 帧的 Attention 可视化
│   ├── prefix_L1_attn_vis_max.jpg            # Layer 1 最活跃 Head 的热力图叠加
│   ├── prefix_L4_attn_vis_max.jpg            # Layer 4 最活跃 Head 的热力图叠加
│   ├── prefix_L5_attn_vis_max.jpg            # Layer 5 最活跃 Head 的热力图叠加
│   ├── prefix_L7_attn_vis_max.jpg            # Layer 7 最活跃 Head 的热力图叠加
│   ├── prefix_L10_attn_vis_max.jpg           # Layer 10 最活跃 Head 的热力图叠加
│   ├── L1_prefix_heads/                      # Layer 1 的所有 8 个 Head
│   │   ├── head_00.jpg
│   │   ├── head_01.jpg
│   │   ├── ...
│   │   └── head_07.jpg
│   ├── L4_prefix_heads/                      # Layer 4 的所有 8 个 Head
│   │   └── head_00.jpg - head_07.jpg
│   ├── L5_prefix_heads/
│   ├── L7_prefix_heads/
│   └── L10_prefix_heads/
├── 00008/                                     # 第 8 帧的 Attention 可视化
│   └── (同上结构)
├── 00016/                                     # 第 16 帧的 Attention 可视化
├── 00024/
├── ...
└── 00176/                                     # 最后一个关键帧
```

#### 文件说明

1. **`pi05.md`**: 
   - 处理完成标记文件，用于断点续传
   - 包含 Episode 元信息：
     - Episode ID 和 Outcome
     - 总帧数 (如 181)
     - 关键帧列表 (如 `[0, 8, 16, 24, ..., 176]`)

2. **关键帧目录** (`00000/`, `00008/`, ...):
   - 命名格式: 5位零填充的帧索引
   - 间隔: 每 8 帧处理一次 (Open-Loop Control 的重新推理点)
   - 一个 181 帧的 Episode 生成约 23 个关键帧目录

3. **`prefix_L{layer}_attn_vis_max.jpg`**:
   - 每个 Layer 中方差最大（最活跃）的 Head 的可视化
   - 热力图叠加在原始图像上
   - 用于快速浏览和对比不同层的关注模式
   - 图片尺寸: 取决于输入图像（通常 ~640x480 或类似）

4. **`L{layer}_prefix_heads/head_{id}.jpg`**:
   - 每个 Layer 的 8 个 Attention Head 的独立可视化
   - Head ID: `00` - `07`
   - 用于精细分析每个 Head 的功能分化
   - 包含 5 个关键层: L1, L4, L5, L7, L10

#### 可视化内容

- **Prefix Phase**: 文本 Token 对图像 Patch 的注意力权重
- **热力图**: 使用颜色编码表示注意力强度（通常红色=高，蓝色=低）
- **叠加模式**: 热力图半透明叠加在原始图像上，便于理解空间对应关系

#### 数据用途

1. **逐帧分析**: 观察单个时间步的注意力分布
2. **层级对比**: 对比不同层 (L1 vs L10) 的功能差异
3. **Head 分化**: 分析同一层内不同 Head 的专业化
4. **成功/失败对比**: 对比 `success/` 和 `failure/` 的注意力模式差异
5. **原始素材**: 为视频合成 (`results_toy_video_right/`) 提供帧素材

### 5.3 Video Output data 

在当前目录下，我们以`results_toy_video_right/` 为例

> **Note**: 这里与 `results_toy_video` 相似，但使用了 **Right Camera** 作为输入。ZED 双目相机的 Right 视角相比 Left 视角可能在某些场景下提供更好的视野或减少遮挡。

该目录存储了针对 Toy Cube Benchmark 的 Attention 可视化视频结果，通过将关键帧图片合成为视频，便于观察 Attention 的时序演化。

#### 整体统计
- **总大小**: ~1.2 GB
- **Episode 总数**: 48 个
  - **Success**: 21 个 Episode
  - **Failure**: 27 个 Episode
- **视频文件总数**: 1,460 个 `.webm` 文件

#### 目录层级结构

```
results_toy_video_right/
├── success/                    # 成功的 Episode
│   ├── 2025-12-10/            # 按日期组织
│   ├── 2025-12-11/
│   └── 2025-12-13/
└── failure/                    # 失败的 Episode
    ├── 2025-12-10/
    └── 2025-12-13/
```

#### Episode 级别结构

每个 Episode 以时间戳命名（格式：`YYYY-MM-DD_HH-MM-SS`），包含以下内容：

```
{outcome}/{date}/{episode_timestamp}/
├── instruction.txt                    # 任务指令文本（如 "place cube with G into the purple bowl"）
├── L01_prefix_max.webm               # Layer 1 的最活跃 Head 视频（Prefix Phase）
├── L04_prefix_max.webm               # Layer 4 的最活跃 Head 视频
├── L05_prefix_max.webm               # Layer 5 的最活跃 Head 视频
├── L07_prefix_max.webm               # Layer 7 的最活跃 Head 视频
├── L10_prefix_max.webm               # Layer 10 的最活跃 Head 视频
├── L1/                               # Layer 1 的所有 8 个 Attention Head 视频
│   ├── H00_prefix.webm
│   ├── H01_prefix.webm
│   ├── H02_prefix.webm
│   ├── H03_prefix.webm
│   ├── H04_prefix.webm
│   ├── H05_prefix.webm
│   ├── H06_prefix.webm
│   └── H07_prefix.webm
├── L4/                               # Layer 4 的所有 8 个 Attention Head 视频
│   └── (H00-H07_prefix.webm)
├── L5/                               # Layer 5 的所有 8 个 Attention Head 视频
│   └── (H00-H07_prefix.webm)
├── L7/                               # Layer 7 的所有 8 个 Attention Head 视频
│   └── (H00-H07_prefix.webm)
└── L10/                              # Layer 10 的所有 8 个 Attention Head 视频
    └── (H00-H07_prefix.webm)
```

#### 文件说明

1. **`instruction.txt`**: 
   - 记录该 Episode 的任务指令
   - 示例: `"place cube with G into the purple bowl"`

2. **`L{layer}_prefix_max.webm`**:
   - 每个 Layer 中方差最大（最活跃）的 Attention Head 的可视化视频
   - 用于快速浏览各层的关键注意力模式
   - 文件大小: ~750-820 KB/视频

3. **`L{layer}/H{head}_prefix.webm`**:
   - 每个 Layer 的 8 个 Attention Head 的完整可视化
   - Head 编号: 00-07（共 8 个）
   - 用于精细分析每个 Head 的注意力分布差异

#### 命名规则

- **Layer**: `L1`, `L4`, `L5`, `L7`, `L10` 等（选择性可视化关键层）
- **Head**: `H00` 到 `H07`（PaliGemma 每层有 8 个 Attention Head）
- **Phase**: `prefix` 表示 Prefix Phase（图像+文本编码阶段）

#### 数据用途

1. **定性分析**: 观察不同层级和 Head 对图像区域的关注模式
2. **时序动态**: 通过视频观察 Attention 随 Episode 时间步的演化
3. **成功/失败对比**: 对比 `success/` 和 `failure/` 目录下的注意力差异
4. **多层级对比**: 比较浅层 (L1) vs 中层 (L5, L7) vs 深层 (L10) 的功能分化

### 5.4 数据流水线总结

整个 Attention 可视化流程的数据流向如下：

```
[Input] toy_cube_benchmark/              (原始 DROID 数据集, ~13 GB, 82 episodes)
   ├── recordings/frames/*.jpg           (视频帧序列, 每 episode ~181 帧)
   ├── trajectory.h5                     (机器人轨迹数据)
   └── instruction.txt                   (任务指令)
          ↓
    [Processing Pipeline]
    - 加载 Pi0.5 模型 + PaliGemma VLM
    - 提取 Attention Weights (修改 gemma_pytorch.py)
    - 关键帧推理 (每 8 帧, Open-Loop Control)
    - 生成热力图可视化
          ↓
[Output 1] results_toy_right/            (静态图片, ~973 MB, 32 episodes)
   ├── {frame_idx}/prefix_L*_attn_vis_max.jpg   (每层最活跃 Head)
   └── {frame_idx}/L*_prefix_heads/head_*.jpg   (所有 8 个 Head)
          ↓
    [Video Synthesis]
    - 合成关键帧为视频 (combine_video.py)
    - 自动筛选最活跃 Head
          ↓
[Output 2] results_toy_video_right/      (视频文件, ~1.2 GB, 48 episodes)
   ├── L*_prefix_max.webm                (每层最活跃 Head 的时序视频)
   └── L*/H*_prefix.webm                 (每个 Head 的完整视频)
```

#### 关键处理参数

- **关键帧间隔**: 8 帧 (对应 Open-Loop Control 的重新推理频率)
- **可视化层级**: L1, L4, L5, L7, L10 (代表浅层、中层、深层)
- **Attention Head 数量**: 每层 8 个 (PaliGemma 标准配置)
- **视频格式**: WebM (VP9 编码, 高压缩比)
- **图片格式**: JPEG (有损压缩, 平衡质量与存储)


#### 数据对应关系

- **Episode ID 一致性**: 三个目录使用相同的时间戳命名 (`YYYY-MM-DD_HH-MM-SS`)
- **Outcome 分类一致**: 均按 `success/` 和 `failure/` 分类
- **帧索引对应**: `results_toy_right/00000/` 对应 `toy_cube_benchmark/recordings/frames/00000.jpg`
- **处理覆盖率**: 
  - Input: 82 episodes (100%)
  - Attention Maps: 32 episodes (~39%, 部分处理)
  - Attention Videos: 48 episodes (~59%, 更多 episode 已完成)


---

## 6. Hypothesis 1.1: Attention-Object Correlation (2024-12-24)

### 6.1 研究问题

**核心假设**: VLM attention maps 会聚焦在 DINO-X 检测到的物体上，且不同层级的聚焦程度不同。

### 6.2 实现方案

- **数据**: DROID episodes + DINO-X object detection masks
- **关键修正**: 使用 **Wrist Camera** attention (tokens 256-511)，而非 Side Camera
- **指标**: 
  - Overlap Ratio: 物体上的 attention 比例
  - Attention Concentration: 物体内外 attention 均值比
  - IoU: 二值化 attention 与 mask 的空间重叠度

### 6.3 工具脚本

- **`viz/h1_1_object_detection.py`**: 单 episode 分析，生成逐帧可视化和视频
- **`viz/object_pipeline.py`**: 批量处理 pipeline，支持多 episode 聚合分析

### 6.4 主要发现 (初步)

- **Layer 5**: Overlap 和 Concentration 峰值，语义物体识别的关键层
- **Layer 8**: 第二峰值，任务导向的物体关注
- **Layer 14-17**: Concentration <1.0，attention 转向动作规划（gripper、轨迹）
- **分层功能**: 识别 (L3-5) → 转换 (L6) → 任务规划 (L7-8) → 动作执行 (L14+)


