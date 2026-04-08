# Attention Masking 实验 & 确定性分析

日期: 2025-04-03

## Attention 保存机制对比

### PyTorch 版
- Eager 执行，写一行跑一行
- HuggingFace 原生 `output_attentions=True`，forward 完直接 `.detach().cpu().numpy()`
- 能同时抓 prefix + suffix attention
- 不需要改模型代码

### JAX 版
- Trace + Compile 模式：先把计算录成计算图（trace），再交给 XLA 编译优化后一口气执行
- 18 层 transformer 用 `nn.scan` 编译成一个 XLA 循环体，不能在里面 `list.append()`
- 用 `jax.debug.callback` 作为"逃生口"：XLA 执行到该点时暂停，把数据从 GPU 拷回 CPU，调用 Python 函数存储
- 每层一次 GPU→CPU 同步，共 18 次，**会拖慢推理**
- 只能抓 prefix attention（suffix 在 while_loop 里跑 10 步，抓的话 18×10=180 次回调，OOM）

## Attention Masking 实验设计

### 核心思路
Masking 发生在 XLA 计算图内部（softmax 之前改 logit），不需要 callback，不拖慢推理。

### 三种 masking 模式

| mode | 名称 | 效果 | 例子 (pct=10) |
|------|------|------|---------------|
| 1 | mask top | 把最强的 N% logit 设为 -inf | 砍掉最强 10% attention |
| 2 | mask bottom | 把最弱的 N% logit 设为 -inf | 砍掉最弱 10% attention |
| 3 | min filter | 只保留最强的 N%，其余设为 -inf | 保留 top 10%，砍掉 90% |

mode 3 是 mode 1 的反操作：mode 1 砍掉的 = mode 3 保留的。

### 实现位置
- `src/openpi/models/gemma.py` — `_mask_attn_percentile()` 函数
- `src/openpi/models/pi0.py` — `sample_actions()` 的 prefix forward 传入 `attn_mask_config`
- `src/openpi/models/pi0_config.py` — `attn_logit_mask_layers` 和 `attn_logit_mask_percentile`
- `src/openpi/models/attn_mask_test.py` — 12 个单元测试

### 实验脚本
`viz/test_attn_masking.py` — 对同一个输入（duck frame 40），用固定噪声跑 baseline + 多种 masking 配置，比较 action 偏移。

### Masking 只作用于 prefix pass
Prefix 是模型"理解"图片和文本的地方。Mask 掉 prefix 里的 attention logit → 改变 KV cache → 传递到后续所有去噪步 → 影响最终 action。

## 确定性分析

### JAX — 完全确定
1. **noise**: `sample_actions` 接受 `noise=` 参数，传固定值即可
2. **preprocess_observation**: `train=False` 不走随机增强
3. **模型计算**: 无 dropout；XLA 同设备同编译同输入 = bit-identical
4. **while_loop**: 固定 10 步迭代

结论：**JAX 实验天然确定，不需要额外设置。**

### PyTorch — 基本确定，但非 bit-identical
1. **noise**: 同样可传固定值（但默认用 `torch.normal` 依赖全局 RNG，不传就不确定）
2. **preprocess_observation**: 同上，`train=False` 确定
3. **cuDNN 非确定性**: matmul 和 attention 默认用非确定算法（更快），浮点抖动约 1e-6 ~ 1e-5
4. **强制确定**: 需要 `torch.use_deterministic_algorithms(True)` + `cudnn.deterministic=True`，但会降速且部分 op 不兼容

结论：**PyTorch 的浮点抖动远小于 masking 造成的 action 偏移（1e-2 ~ 1e-1 级），实际不影响实验结论，但严格来说不是 bit-identical。**

