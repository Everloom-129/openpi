# Pi0.5 PaliGemma Attention Visualization 实践记录 '
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

```python
# 在 PaliGemmaWithExpertModel.forward 中:

# 1. 开启 Attention 输出
outputs = self.paligemma.language_model(..., output_attentions=True)

# 2. Hack: 直接保存每一层的 Attention 到磁盘 (仅非训练模式)
if not self.training:
    import numpy as np
    # 保存 Prefix Attention
    if prefix_output.attentions is not None:
        os.makedirs("results/layers_prefix", exist_ok=True)
        for i, layer_attn in enumerate(prefix_output.attentions):
            # layer_attn shape: [Batch, NumHeads, SeqLen, SeqLen]
            np.save(f"results/layers_prefix/attn_map_layer_{i}.npy", layer_attn.detach().cpu().float().numpy())
```

### 2.2 禁用 Torch Compile
**文件**: `src/openpi/models_pytorch/pi0_pytorch.py`

由于我们插入了 I/O 操作 (`numpy.save`) 且需要使用 `pdb` 调试，必须禁用 `torch.compile`。JIT 编译会隐藏中间变量并导致图捕获错误。

```python
# 注释掉 compile
# self.sample_actions = torch.compile(self._sample_actions, mode="max-autotune")
self.sample_actions = self._sample_actions
```

## 3. 可视化脚本逻辑
**脚本**: `viz/attn_map.py`

我们实现了两种可视化模式：

### 3.1 预处理对齐  
模型输入图片被 Resize 到了 224x224 (保持比例并 Padding)。
- [x] 为了确保 Attention Map 和图片对齐，我们在可视化时必须对原图进行相同的 `resize_with_pad` 操作。

### 3.2 模式 1: Summary Overlay (`visualize_attention`)
将 Attention Map 叠加在原图上，用于快速判断该层是否关注了正确物体。
- **操作**: 对 Heads 取平均 (或最大值)。
- **布局**: 2x2 网格 (原图 vs 叠加图)。
- **输出**: 保存为 `results/{name}/{prefix/suffix}_L{layer}_attn_vis_{mode}.jpg`。

### 3.3 模式 2: Detailed Heads (`visualize_heads`)
按照论文风格，单独展示每一层的 8 个 Head，观察不同的 Head 是否分化出了不同的功能（如：有的关注手，有的关注物体）。
- **操作**: 不进行叠加，直接绘制 `viridis` 热力图。
- **布局**: 3列布局 (Exterior Attention | Wrist Attention | Reference Images)。
- **输出**: 保存为 `results/{name}/L{layer}_heads/head_{i}.jpg`。

## 4. Attention 维度解析

PaliGemma 的 Attention Map 维度处理：

1. **Token 数量**:
   - 图像 Patch 大小为 14，输入 224x224 -> 16x16 = 256 Tokens。
   - Droid 模型有两个相机 (Exterior, Wrist)，共 512 Image Tokens。
   
2. **Prefix Attention (Context -> Context)**:
   - Shape: `[Batch, Heads, TotalLen, TotalLen]` (方阵)。
   - 我们关注: `Attention[TextTokens, ImageTokens]`。即文本 Prompt 看了图的哪里。

3. **Suffix Attention (Action -> Context)**:
   - Shape: `[Batch, Heads, ActionLen, TotalLen]` (长方形矩阵)。
   - 我们关注: `Attention[ActionTokens, ImageTokens]`。即生成的动作看了图的哪里。

## 5. 结果示例
生成的图片保存在 `results/` 目录下：
- `results/duck_left_0/prefix_L15_attn_vis_max.jpg`: 第15层 Prefix Attention 叠加图。
- `results/duck_left_0/L15_prefix_heads/head_03.jpg`: 第15层 第3个 Head 的原始热力图。

## 6. 结论
通过 Layer 4, 10, 15, 17 的可视化，我们可以观察到模型在深层网络中逐渐聚焦于与 Prompt 相关的物体（如 "duck" 或 "bowl"）。

