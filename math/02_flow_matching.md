# Flow Matching: 第一性原理推导

## 1. 问题设置

VLA 的 action 生成问题：给定感知输入 $s$（图像 + 语言），学习条件分布 $p(a \mid s)$。

其中 $a \in \mathbb{R}^{T \times D}$，$T=8$ 步，$D=8$ 维（7 关节速度 + 1 夹爪）。

**目标**：从简单分布 $p_0 = \mathcal{N}(0, I)$ 出发，构造一条"流"把噪声变成 action 样本。

---

## 2. Continuous Normalizing Flow（CNF）

定义一条随时间变化的概率路径 $\{p_t\}_{t \in [0,1]}$：

$$p_0 = \mathcal{N}(0, I), \qquad p_1 \approx p_\text{data}(a \mid s)$$

这条路径由一个**向量场** $u_t : \mathbb{R}^D \to \mathbb{R}^D$ 生成，满足连续性方程（概率守恒）：

$$\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t \, u_t) = 0$$

给定初始点 $x_0 \sim p_0$，沿向量场积分得到轨迹 $\phi_t(x_0)$：

$$\frac{d}{dt}\phi_t(x) = u_t(\phi_t(x)), \qquad \phi_0(x) = x$$

**目标**：学习 $v_\theta \approx u_t$，使得 $\phi_1 \sim p_\text{data}$。

---

## 3. Flow Matching 目标函数

直接回归向量场：

$$\mathcal{L}_\text{FM}(\theta) = \mathbb{E}_{t, x \sim p_t} \left[ \| v_\theta(x, t, s) - u_t(x) \right\|^2 ]$$

**问题**：$u_t(x)$ 依赖整个 $p_t$，边际向量场难以直接计算。

---

## 4. Conditional Flow Matching（CFM）

**关键技巧**：条件化到数据点 $x_1 \sim p_\text{data}$，构造条件路径。

**最简单的条件路径**（线性插值，π₀ 使用这个）：

$$x_t = (1-t) \cdot x_0 + t \cdot x_1, \qquad x_0 \sim \mathcal{N}(0, I), \quad x_1 \sim p_\text{data}$$

对应的**条件向量场**（目标速度）：

$$u_t(x_t \mid x_0, x_1) = x_1 - x_0$$

注意：这个目标是**常数**！不依赖 $t$，也不依赖当前位置 $x_t$（在线性路径下）。

**CFM 目标函数**：

$$\mathcal{L}_\text{CFM}(\theta) = \mathbb{E}_{t \sim U[0,1],\; x_1 \sim p_\text{data},\; x_0 \sim \mathcal{N}(0,I)} \left[ \| v_\theta(x_t, t, s) - (x_1 - x_0) \right\|^2 ]$$

**可以证明**：$\nabla_\theta \mathcal{L}_\text{CFM} = \nabla_\theta \mathcal{L}_\text{FM}$（梯度相等，即训练等价）。

---

## 5. π₀ 的具体实现

π₀ 中，$v_\theta$ 由 **Action Expert**（小型 Transformer）实现：

```
输入: x_t (noisy action, shape [T,D])
      t   (noise level scalar)
      s   = concat(image_tokens, text_tokens)  ← 来自 PaliGemma backbone
输出: v_θ(x_t, t, s)  (predicted velocity, same shape as x_t)
```

**训练时**：
$$x_t = (1-t) x_0 + t a_\text{gt}, \quad \text{target} = a_\text{gt} - x_0$$

**推理时**：从 $x_0 \sim \mathcal{N}(0,I)$ 出发，用 ODE solver（如 Euler）积分：

$$x_{t+\Delta t} = x_t + v_\theta(x_t, t, s) \cdot \Delta t$$

走 $N$ 步（默认 $N=10$ 或 $N=100$），得到 $x_1 \approx a_\text{pred}$。

---

## 6. 从数学看 π₀ 的 Attention 结构

π₀ 的 Action Expert 对 noisy action token 做 cross-attention，query 来自 action，key/value 来自图像+文本：

$$\text{action\_attn}_{ij} = \text{softmax}\!\left(\frac{q_{\text{action}_i} \cdot k_{\text{img/text}_j}}{\sqrt{d_h}}\right)$$

**第一性原理推论**：

1. **Attention 在 flow matching 里的角色**：action expert 需要从图像/文本中"读取"条件信息 $s$，来预测 $v_\theta(x_t, t, s)$，即"噪声 action 应该往哪个方向走"。

2. **Noise level $t$ 的影响**：
   - $t \to 0$：$x_t \approx x_0$（纯噪声），action expert 需要从 $s$ 中读取大量信息，attention 应该更"认真看图像"
   - $t \to 1$：$x_t \approx x_1$（接近真实 action），已经有了方向，attention 可以做细微修正

   **这是一个可以验证的假设！** 可以在不同 $t$ 下观察 action→image attention 的分布变化。

3. **Flow matching 的确定性**：线性路径使得 $u_t = x_1 - x_0$ 是常数向量场，模型的任务是"记住"每种条件 $s$ 下对应哪个方向。这可以理解为：图像/语言 tokens 编码了"action 方向"的信息。

---

## 7. Optimal Transport Flow Matching（OT-CFM）

更好的路径选择：使用最优传输（OT）匹配 $x_0 \sim p_0$ 和 $x_1 \sim p_1$，最小化路径弯曲程度：

$$\min_\pi \mathbb{E}_{(x_0, x_1) \sim \pi} \|x_0 - x_1\|^2 \quad \text{s.t. } \pi \text{ 的边际为 } p_0, p_1$$

OT 路径比随机配对更直，减少路径交叉，训练更稳定。π₀.5 可能使用了这种变体。

---

## 8. 与 Diffusion 的关系

| | Flow Matching | DDPM/Score Matching |
|--|--|--|
| 路径 | 直线（线性插值）| 随机游走（SDE） |
| 目标 | 速度 $v = x_1 - x_0$ | 分数 $\nabla \log p_t$ |
| 推理步数 | 少（10步可以） | 多（100-1000步） |
| 概率路径 | 确定性 ODE | 随机 SDE |
| 本质 | 等价（特定参数化下）| 等价 |

Flow matching 可以理解为 diffusion 的确定性极限（噪声系数 → 0）。

---

## 9. 对可解释性研究的启示

**Flow matching 给了我们一个新的分析维度：噪声水平 $t$**

```
当 t 很小（接近纯噪声）：
  - 模型完全依赖条件 s = (image, text) 来确定方向
  - 此时 attention 最"信息密集"，最能反映模型真正在用什么

当 t 很大（接近真实 action）：
  - 模型在做小幅度修正
  - 此时 attention 可能更关注细节（精确位置、抓握方式）
```

**可行研究问题**：
1. 不同 $t$ 下，attention 模式如何变化？（需要在推理时 hook 不同步骤的 attention）
2. 哪些图像 patch 在 $t \approx 0$ 时被关注最多？（这最接近"真正用到了什么"）
3. 模型在什么 $t$ 值发生"注意力转移"？

**当前代码只捕获了最后一步的 attention**（`pipeline.py` 调用一次 `policy.infer()`），未来可以在 ODE 积分过程中每步都 hook attention。
