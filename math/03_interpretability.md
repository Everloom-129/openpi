# VLA 可解释性：方法论框架

## 1. 核心研究问题层次

```
Level 1 (行为级):   "什么输入导致什么输出？"       → 相关性分析
Level 2 (因果级):   "什么输入是输出的原因？"       → 干预分析
Level 3 (机制级):   "内部哪个组件实现了这个功能？"  → 电路分析
```

我们目前的工具（attention heatmap）主要在 Level 1 游走。可解释性研究的目标是到达 Level 2 和 3。

---

## 2. 信息流图：VLA 的计算图

```
图像 (224×224×3)
    ↓ ViT patchify
image_tokens [512, d]  ←─────────────────────────────────────────┐
                                                                   │
语言指令 (string)                                                   │ cross-attn
    ↓ tokenize                                                     │ (action expert)
text_tokens  [~100, d]                                             │
                                                                   │
    ↓ PaliGemma backbone (18 layers self-attention)                │
                                                                   │
context_tokens [~600, d]  ──────────────────────────────────────→ │
                                                                   │
noise x_t [8, 8]  ──→  action_tokens [8, d]                      │
                              ↓                                    │
                         v_θ(x_t, t, s)  ←─────────────────────── ┘
                              ↓
                         x_{t+dt} = x_t + v_θ · dt
```

**关键节点**：
- `context_tokens` = PaliGemma 最后一层的 key/value（这是 action expert 的 "记忆库"）
- `action_tokens` 的残差流在积分过程中逐渐从噪声变为有意义的 action

---

## 3. 方法论对比

### 3.1 Attention Analysis（我们现在在做的）

**数学**：$A_{ij}^{(l,h)}$，第 $l$ 层第 $h$ head 的 attention weight 矩阵

**能回答**：
- 模型在某层某 head 的"路由偏好"是什么？
- 不同 token 之间的相关性结构如何？

**不能回答**：
- 这个路由是否对输出有因果影响？
- 被关注的信息是否真的被"使用"了？

**已有实现**：`grid_heatmap.py`, `action_view.py`, `trajectory.py`

---

### 3.2 Gradient-based Saliency（升级版）

**数学**：

$$s_j = \left\| \frac{\partial \, \mathcal{L}(a_\text{pred})}{\partial \, e_j} \right\|$$

其中 $e_j$ 是第 $j$ 个 image patch 的 embedding。

更稳定的版本 — **Integrated Gradients**：

$$\text{IG}_j = (e_j - e_j^\text{baseline}) \cdot \int_0^1 \frac{\partial F(\tilde{e}(\alpha))}{\partial \tilde{e}_j} \, d\alpha$$

满足**完备性**：$\sum_j \text{IG}_j = F(e) - F(\text{baseline})$（所有 patch 的贡献之和等于输出差）

**能回答**：
- 哪个 patch 的输入变化最影响 action 输出？（比 attention 更直接）
- 对于具体 action 维度（如 gripper），哪个区域最关键？

**需要**：反向传播到输入层，需要 PyTorch 模型支持（`gemma_pytorch.py` 已是 PyTorch）

---

### 3.3 Occlusion / Patch Ablation（你已有的 `image_saliency.py`）

**数学**：

$$\text{ablation\_score}_j = \| a_\text{pred} - a_\text{pred}^{(\neg j)} \|$$

其中 $a_\text{pred}^{(\neg j)}$ 是遮住第 $j$ 个 patch 后的预测。

**优点**：直接因果干预，无需梯度，模型无关

**缺点**：计算量大（$O(n_\text{patches})$ 次前向传播）；离散遮挡不够平滑

---

### 3.4 Activation Patching（最强因果工具）

**数学**：设有两个输入 $s_A$（正常场景）和 $s_B$（对比场景），

在 $s_B$ 的推理中，将第 $l$ 层第 $i$ 个 token 的激活替换为 $s_A$ 的激活：

$$h_i^{(l)} \leftarrow h_i^{(l,A)}$$

测量输出变化 $\Delta a = a_\text{pred}^{(B, \text{patched})} - a_\text{pred}^{(B)}$

**能回答**：
- 具体哪一层、哪个 token 的激活是成功/失败的因果因素？
- "知识"存在于哪里？

**需要**：一对有意义的对比 episode（成功 vs 失败，不同指令等）

**实现难度**：中等（需要注册 PyTorch forward hooks，保存/注入激活）

---

### 3.5 Probing Classifier（表征分析）

**数学**：在第 $l$ 层提取 token $i$ 的激活 $h_i^{(l)}$，训练线性分类器：

$$\hat{y} = W h_i^{(l)} + b, \quad \mathcal{L} = \text{CE}(\hat{y}, y)$$

其中 $y$ 是我们关心的 ground truth 标签（物体位置、夹爪状态、任务阶段等）。

**Probe 精度高** → 该层的 token 表征确实编码了 $y$

**能回答**：
- 哪一层首先出现"物体位置"的编码？
- image token vs action token，谁更"了解"夹爪状态？

**需要**：带标注的数据集（可以用 DROID 的 trajectory.h5 + 视觉识别 pipeline 自动生成）

---

## 4. 方法间的对比总结

| 方法 | 因果性 | 计算成本 | 需要梯度 | 你的现状 |
|------|--------|----------|----------|---------|
| Attention weight | 相关性（弱） | O(1) | 否 | ✅ 已有 |
| Value-weighted attention | 相关性（强） | O(1) | 否 | 🔧 一行升级 |
| Attention rollout | 相关性（多层） | O(L) | 否 | 🔧 可加 |
| Gradient saliency | 因果（局部线性）| O(1) backward | 是 | 📋 需开发 |
| Integrated Gradients | 因果（公理满足）| O(N) backward | 是 | 📋 需开发 |
| Occlusion ablation | 因果（直接）| O(n_patches) forward | 否 | ✅ 已有框架 |
| Activation patching | 因果（最强）| O(1) × 2 | 否 | 📋 需开发 |
| Probing | 表征分析 | 需要标注数据 | 否 | 📋 需开发 |

---

## 5. 针对 VLA 特有的研究问题

### Q1：指令遵循 vs 视觉 Shortcut
**假设**：模型可能在忽略指令，只靠图像中的物体位置来决定动作。

**验证方法**：
- Counterfactual prompt：换掉指令 → 看 action 变化量
- IG on text tokens vs image patches：比较各自的显著性
- 如果 IG 显示 text token 贡献接近 0 → shortcut 假设成立

### Q2：哪一层实现了 Action Grounding
**假设**：存在某几层，在这几层之后 action token 的表征才"知道"该做什么动作。

**验证方法**：
- 在每层对 action token 做 probing（预测动作方向）
- Probing accuracy 从某层开始急剧上升 → 该层是 grounding 发生的地方
- Activation patching 验证：patch 该层之后的激活影响大，之前的影响小

### Q3：Flow Matching 不同 t 下的注意力演化
**假设**：$t \approx 0$ 时 attention 最信息密集，$t \approx 1$ 时 attention 在做细节修正。

**验证方法**：
- Hook ODE 积分的每一步，记录 action→image attention
- 计算每步的 attention entropy，看它如何随 t 变化
- 找 "attention transition point"：从扩散性 → 集中性的转变时刻

### Q4：Failure Mode 的注意力特征
**假设**：失败帧有可识别的 attention 异常模式。

**验证方法**（不需要 attention-action 直接相关性）：
- 聚类所有帧的 attention 模式（PCA + k-means on flattened attention maps）
- 检查各聚类中 success/failure 的比例
- 找到 failure-enriched 聚类的 attention 共同特征

---

## 6. 推荐研究路径

```
阶段 1（1-2周）：基础升级
  ├── 实现 value-weighted attention（一行修改，立即更有意义）
  ├── 实现 attention rollout（跨层累积，看全局信息流）
  └── 系统化 occlusion → ΔAction（已有框架，补全批量 + 排名）

阶段 2（2-4周）：梯度工具
  ├── 实现 gradient saliency（PyTorch autograd, 需要 retain_graph）
  ├── 与 attention 对比：两者一致的 patch = 真正重要的 patch
  └── 对比 success vs failure 的梯度 saliency 图

阶段 3（1-2月）：因果分析
  ├── 收集成对的 success/failure episode（相同任务，不同结果）
  ├── 实现 activation patching（register_forward_hook）
  └── 找到 failure 的因果层/token
```
