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

我们开发了一套工具链用于分析 Attention Map。

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


