# Attention: 第一性原理推导

## 1. 基础定义

设序列长度为 $n$，每个 token 的表征维度为 $d$。输入矩阵 $X \in \mathbb{R}^{n \times d}$。

**线性投影**（每个 head 独立）：

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

其中 $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_h}$，$d_h = d/H$（$H$ 为 head 数）。

**Scaled dot-product attention**：

$$A = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right) \in \mathbb{R}^{n \times n}, \qquad \text{output} = AV$$

$A_{ij}$ 是 token $i$ 对 token $j$ 的注意力权重，满足 $\sum_j A_{ij} = 1$。

---

## 2. 输出的本质：Value 的加权平均

$$\text{output}_i = \sum_{j=1}^n A_{ij} \cdot v_j$$

这里 $v_j = (XW_V)_j \in \mathbb{R}^{d_h}$ 是 token $j$ 的 **value 向量**。

**关键洞察**：attention 的输出是 value 向量的凸组合。$A_{ij}$ 决定"从哪里聚合"，但每个 $v_j$ 的**幅度** $\|v_j\|$ 决定"能贡献多少"。

> **对可解释性的含义**：
> - $A_{ij}$ 大 但 $\|v_j\|$ 小 → 实际贡献极小
> - $A_{ij}$ 小 但 $\|v_j\|$ 大 → 可能是主要信息来源
> - 单看 $A_{ij}$ 是不够的

---

## 3. Softmax 的温度与熵

令 $s_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_h}}$ 为原始 logit，则：

$$A_{ij} = \frac{e^{s_{ij}}}{\sum_k e^{s_{ik}}}$$

**注意力分布的熵**：

$$H_i = -\sum_j A_{ij} \log A_{ij}$$

- $H_i \to 0$：注意力集中在单一 token（sharp / focused）
- $H_i \to \log n$：注意力均匀分散（diffuse / uncertain）

这就是我们在 `trajectory.py` 中 `concentration = 1 - H / H_max` 的数学基础。

**但熵低不等于模型"有把握"**，只意味着这个 head 在当前层从单一来源读取信息。

---

## 4. 多头的作用

$$\text{MHA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W_O$$

每个 head 学习不同的 $(W_Q^h, W_K^h, W_V^h)$，即在不同的"关系子空间"中聚合信息。

**Head 之间的分歧**（我们的 `Max - Min` / `Std Dev` 指标）：

$$\text{disagreement}_i = \frac{1}{H}\sum_h \left\| A^h_{i,:} - \bar{A}_{i,:} \right\|_2$$

高分歧 = 不同 head 对"应该看哪里"有明显不同判断，可能意味着该 token 的语义在多个维度上都有关联。

---

## 5. Residual Stream 视角（Mechanistic Interpretability 核心框架）

Transformer 的本质不是"attention layer 的堆叠"，而是**残差流（residual stream）的累积写入**：

$$h_i^{(0)} = \text{embed}(x_i) + \text{pos\_embed}(i)$$

$$h_i^{(l+1)} = h_i^{(l)} + \underbrace{\text{MHA}^{(l)}(H^{(l)})_i}_{\text{attention 的写入}} + \underbrace{\text{MLP}^{(l)}(h_i^{(l)})}_{\text{MLP 的写入}}$$

最终的 action 预测从 action token 的残差流 $h_{\text{action}}^{(L)}$ 中读取。

**第一性原理推论**：
- 每个 attention layer 的作用是"把其他 token 的信息搬运到当前 token 的残差流里"
- 每个 MLP layer 的作用是"在当前 token 的残差流里做非线性变换（知识存储）"
- 可解释性的核心问题：**哪个组件，在哪一层，往 action token 的残差流里写入了决定性信息？**

---

## 6. 为什么 Attention 不等于 Explanation

形式化地，token $j$ 对 token $i$ 输出的**真实因果贡献**应该是：

$$\frac{\partial \, \text{output}_i}{\partial \, h_j^{(l)}} \quad \text{（梯度）}$$

而 attention weight $A_{ij}$ 只是：

$$A_{ij} = \frac{\partial \, \text{output}_i}{\partial \, s_{ij}} \cdot \frac{1}{\|v_j\|}  \quad \text{（近似，不精确）}$$

两者差异来源：
1. $V$ 矩阵投影改变了向量方向和幅度
2. Softmax 的归一化引入了所有 token 之间的耦合
3. 残差连接让信息可以绕过 attention 直接传递

**更好的 attribution 方法**：

| 方法 | 公式 | 含义 |
|------|------|------|
| Attention | $A_{ij}$ | 路由权重 |
| Value-weighted | $A_{ij} \cdot \|v_j\|$ | 信息量加权 |
| Gradient × Input | $\nabla_{h_j} \text{output}_i \odot h_j$ | 局部线性近似 |
| Integrated Gradients | $\int_0^1 \nabla_{h_j(\alpha)} d\alpha$ | 路径积分，满足完备性公理 |
| Activation Patching | $\Delta\text{output}$ when patching $h_j$ | 真正的因果干预 |

---

## 7. 对我们研究的直接指导

```
我们有的信号：
  text_to_img[layer, head, token, patch]    ← 原始 attention weight A_ij

我们可以升级到：
  text_to_img × ||V[patch]||                ← value-weighted，更有意义

我们真正想要的：
  ∂(action_output) / ∂(image_patch_embed)   ← 梯度，需要反向传播
  或: patch ablation → Δaction              ← 因果干预（已有 occlusion view）
```

**核心问题**：给定某一帧，模型预测 action $a^*$，哪个 image patch 的信息对 $a^*$ 的贡献最大？

这个问题用 attention 只能间接回答，用梯度或 activation patching 才能直接回答。
