# H2.1 Counterfactual Prompt Analysis Pipeline

## 概述

在 `viz/object_pipeline.py` 中集成了 **Hypothesis 2.1: 文本 Prompt 控制视觉注意力** 的批处理分析功能。该功能通过反事实 (counterfactual) prompt 测试，验证 VLM backbone 是否真正理解并响应 prompt 中的物体名称变化。

## 核心假设

**Hypothesis 2.1**: 改变 Text Prompt 中的物体名词，Prefix Attention 的重心会转移到新物体上（即使该物体不存在），而不是停留在原物体上。

- **如果成立**: 说明 VLM 真正理解语义，attention 受 prompt 控制
- **如果不成立**: 说明 VLM 主要依赖视觉特征，忽略 prompt 语义

## 控制 FLAG

在 `viz/object_pipeline.py` 顶部添加了两个控制开关：

```python
# ============================================================================
# CONTROL FLAGS
# ============================================================================
ENABLE_OBJECT_DETECTION = True  # Run object detection correlation analysis
ENABLE_COUNTERFACTUAL = True    # Run counterfactual prompt analysis
```

### 使用场景

1. **两个都启用** (默认):
   ```python
   ENABLE_OBJECT_DETECTION = True
   ENABLE_COUNTERFACTUAL = True
   ```
   - 同时运行 object detection 和 counterfactual 分析
   - 适合全面评估

2. **只运行 Object Detection**:
   ```python
   ENABLE_OBJECT_DETECTION = True
   ENABLE_COUNTERFACTUAL = False
   ```
   - 仅分析 attention-object 相关性
   - 节省时间，适合快速验证

3. **只运行 Counterfactual**:
   ```python
   ENABLE_OBJECT_DETECTION = False
   ENABLE_COUNTERFACTUAL = True
   ```
   - 仅测试 prompt 控制能力
   - 适合专注于语义理解研究

## Counterfactual 配置

```python
# Counterfactual prompt configuration
COUNTERFACTUAL_LAYERS = [5, 10, 17]  # Key layers for counterfactual analysis
COUNTERFACTUAL_PROMPTS = {
    "baseline": "find the {object} and pick it up",  # 会被替换为实际物体
    "duck": "find the duck toy and pick it up",
    "banana": "find the banana and pick it up",
    "cat": "find the cat toy and pick it up",
    "bottle": "find the bottle and pick it up",
}
ANALYSIS_CAMERA = "wrist"  # Which camera's attention to analyze
```

### 参数说明

- **COUNTERFACTUAL_LAYERS**: 选择关键层进行分析（通常选择早期、中期、后期各一层）
- **COUNTERFACTUAL_PROMPTS**: 
  - `baseline`: 使用实际物体名称（从原始 prompt 提取）
  - 其他: 反事实物体名称（不存在于场景中）
- **ANALYSIS_CAMERA**: 分析哪个相机的 attention（`wrist` 或 `exterior`）

## 新增函数

### 1. `extract_attention_map_from_policy()`
```python
def extract_attention_map_from_policy(policy, example, layer: int, camera: str = "wrist")
```
- 从已运行 `policy.infer()` 的结果中提取 attention map
- 返回 16x16 的 attention 矩阵

### 2. `visualize_counterfactual_comparison()`
```python
def visualize_counterfactual_comparison(
    attention_maps: dict,
    prompts: dict,
    reference_image: np.ndarray,
    output_path: str,
    layer: int,
    camera: str = "wrist",
)
```
- 并排显示不同 prompt 的 attention maps
- 上排: 纯热力图
- 下排: 叠加在原图上

### 3. `visualize_counterfactual_difference()`
```python
def visualize_counterfactual_difference(
    attention_maps: dict,
    prompts: dict,
    reference_image: np.ndarray,
    output_path: str,
    layer: int,
    baseline_key: str,
    camera: str = "wrist",
)
```
- 计算并可视化差分图: `Δ = M_counterfactual - M_baseline`
- 红色区域: 反事实 prompt 增加的 attention
- 蓝色区域: 反事实 prompt 减少的 attention

### 4. `compute_counterfactual_statistics()`
```python
def compute_counterfactual_statistics(attention_maps: dict, prompts: dict, baseline_key: str)
```
- 计算统计指标:
  - **Mean Δ**: 平均差异（正值 = 整体 attention 增加）
  - **|Δ|**: 绝对差异（对 prompt 变化的敏感度）
  - **L2 Distance**: 欧氏距离（空间差异大小）
  - **Correlation**: 相关系数（1.0 = 完全相同，0.0 = 无关）

### 5. `run_counterfactual_analysis()`
```python
def run_counterfactual_analysis(
    policy,
    example_base: dict,
    frame_idx: int,
    episode_dir: Path,
    layers: list[int],
    prompts: dict,
    baseline_key: str,
    camera: str = "wrist",
    object_name: str = "object",
)
```
- 对单帧运行完整的 counterfactual 分析
- 自动提取 attention、生成可视化、计算统计

## 输出结构

```
{episode_dir}/
├── object/                              # Object Detection 结果 (如果 ENABLE_OBJECT_DETECTION=True)
│   └── {frame:05d}/
│       ├── L0.jpg, L1.jpg, ...
│       └── ...
│
├── counterfactual/                      # Counterfactual 结果 (如果 ENABLE_COUNTERFACTUAL=True)
│   └── {frame:05d}/
│       ├── frame_{frame:05d}_{camera}_reference.jpg
│       ├── L05_prompt_comparison.png    # 并排对比
│       ├── L05_difference_maps.png      # 差分图
│       ├── L10_prompt_comparison.png
│       ├── L10_difference_maps.png
│       ├── L17_prompt_comparison.png
│       └── L17_difference_maps.png
│
├── videos/                              # Object Detection 视频 (如果启用)
│   ├── L00_object_attn.webm
│   ├── L01_object_attn.webm
│   └── ...
│
├── h1_1_obj_attn_results.json          # Object Detection JSON 结果
├── h1_1_obj_attn_summary.png           # Object Detection 汇总图
├── h2_1_counterfactual_results.json    # Counterfactual JSON 结果
├── h2_1_counterfactual_report.md       # Counterfactual 报告
└── object_detection.md                  # 标记文件
```

## 统计指标解释

### Counterfactual 分析指标

| 指标 | 含义 | 预期 (Hypothesis 成立) |
|------|------|----------------------|
| **Mean Δ** | 平均差异 | 显著非零 |
| **\|Δ\|** | 绝对差异 (敏感度) | 较大 (> 0.1) |
| **L2 Distance** | 欧氏距离 | 较大 (> 1.0) |
| **Correlation** | 空间相关性 | 较低 (< 0.7) |

### 解读示例

**Hypothesis 成立** (Prompt 真正控制 attention):
```
| Prompt  | Mean Δ  | |Δ|   | L2 Distance | Correlation |
|---------|---------|-------|-------------|-------------|
| duck    | +0.0234 | 0.156 | 2.341       | 0.523       |
| banana  | -0.0187 | 0.142 | 2.198       | 0.567       |
| cat     | +0.0312 | 0.178 | 2.567       | 0.489       |
```
✅ L2 距离大，相关性低，说明 attention 发生了显著转移

**Hypothesis 不成立** (模型忽略 prompt):
```
| Prompt  | Mean Δ  | |Δ|   | L2 Distance | Correlation |
|---------|---------|-------|-------------|-------------|
| duck    | +0.0012 | 0.023 | 0.234       | 0.967       |
| banana  | -0.0008 | 0.018 | 0.198       | 0.973       |
| cat     | +0.0015 | 0.021 | 0.221       | 0.971       |
```
❌ L2 距离小，相关性极高，说明 attention 几乎不变

## Pipeline 集成

在 `main()` 函数中，对每一帧的处理流程：

```python
# 1. Object Detection Analysis (如果启用)
if ENABLE_OBJECT_DETECTION:
    _ = policy.infer(example)
    frame_results = run_object_detection(...)
    all_results[frame_idx] = frame_results

# 2. Counterfactual Prompt Analysis (如果启用)
if ENABLE_COUNTERFACTUAL:
    # 提取物体名称
    object_name = extract_object_from_prompt(example["prompt"])
    
    # 运行 counterfactual 分析
    cf_results = run_counterfactual_analysis(
        policy, example, frame_idx, episode_dir,
        layers=COUNTERFACTUAL_LAYERS,
        prompts=COUNTERFACTUAL_PROMPTS,
        baseline_key="baseline",
        camera=ANALYSIS_CAMERA,
        object_name=object_name,
    )
    all_cf_results[frame_idx] = cf_results
```

## 生成的报告

### `h2_1_counterfactual_report.md` 示例

```markdown
# Counterfactual Prompt Analysis Report

**Episode**: 2024_01_15_episode_003
**Outcome**: success
**Camera**: wrist
**Frames Analyzed**: 5

## Prompts Tested

- **baseline** (baseline): "find the pineapple toy and pick it up"
- **duck**: "find the duck toy and pick it up"
- **banana**: "find the banana and pick it up"
- **cat**: "find the cat toy and pick it up"
- **bottle**: "find the bottle and pick it up"

## Results Summary

### Frame 50

#### Layer 5

| Prompt | Mean Δ | |Δ| | L2 Distance | Correlation |
|--------|---------|------|-------------|-------------|
| duck | +0.0234 | 0.156 | 2.341 | 0.523 |
| banana | -0.0187 | 0.142 | 2.198 | 0.567 |
| cat | +0.0312 | 0.178 | 2.567 | 0.489 |
| bottle | +0.0098 | 0.134 | 2.089 | 0.601 |

...
```

## 使用示例

### 运行完整 Pipeline

```bash
cd /home/exx/Project_pi05/openpi
python viz/object_pipeline.py
```

### 只运行 Counterfactual 分析

修改 `viz/object_pipeline.py`:
```python
ENABLE_OBJECT_DETECTION = False
ENABLE_COUNTERFACTUAL = True
```

然后运行:
```bash
python viz/object_pipeline.py
```

## 预期发现

### 如果 Hypothesis 2.1 成立

1. **不同 prompt 产生显著不同的 attention maps**
2. **L2 距离较大** (> 1.0)
3. **相关系数较低** (< 0.7)
4. **差分图显示明确的空间转移**

这将证明：
- ✅ VLM backbone 真正理解 prompt 语义
- ✅ Text-to-image attention 受 prompt 控制
- ✅ 模型具有跨模态语义对齐能力

### 如果 Hypothesis 2.1 不成立

1. **所有 prompt 产生几乎相同的 attention maps**
2. **L2 距离很小** (< 0.5)
3. **相关系数很高** (> 0.9)
4. **差分图接近零**

这将说明：
- ❌ VLM 主要依赖视觉特征
- ❌ Prompt 对 attention 影响很小
- ❌ 模型可能只是记忆训练数据模式

## 与 H1.1 的关系

- **H1.1 (Object Detection)**: 验证 attention 是否聚焦在真实存在的物体上
- **H2.1 (Counterfactual)**: 验证 attention 是否受 prompt 中物体名称控制

两者结合可以全面评估：
1. VLM 的视觉理解能力 (H1.1)
2. VLM 的语义理解能力 (H2.1)
3. 跨模态对齐质量

## Multi-Episode Aggregation

### `aggregate_counterfactual_analysis()`

Pipeline 会自动聚合所有 episode 的 counterfactual 结果，生成综合分析。

#### 输出文件

```
{RESULTS_ROOT}/counterfactual_aggregate/
├── cf_l2_distance_comparison.png      # L2 距离对比图 (按 prompt)
├── cf_correlation_comparison.png      # 相关系数对比图 (按 prompt)
├── cf_l2_heatmap.png                  # L2 距离热力图 (Prompt x Layer)
├── cf_multi_episode_report.md         # 综合报告
└── cf_multi_episode_aggregate.json    # 原始数据
```

#### 可视化说明

**1. L2 Distance 对比图**
- 每个 prompt 一个子图
- X 轴: Layers
- Y 轴: L2 Distance
- 蓝色: Success episodes
- 橙色: Failure episodes
- 红色虚线: 阈值 (1.0)

**2. Correlation 对比图**
- 每个 prompt 一个子图
- X 轴: Layers
- Y 轴: Correlation
- 蓝色: Success episodes
- 橙色: Failure episodes
- 红色虚线: 阈值 (0.7)

**3. L2 Distance 热力图**
- 两个子图: Success 和 Failure
- 行: Counterfactual prompts
- 列: Layers
- 颜色: 黄色→橙色→红色 (值越大越红)
- 数值标注在每个格子中

#### 综合报告示例

```markdown
# Counterfactual Prompt Analysis - Multi-Episode Report

**Total Episodes**: 10
- Success: 6
- Failure: 4

**Layers Analyzed**: [5, 10, 17]
**Prompts Tested**: ['duck', 'banana', 'cat', 'bottle']

## Hypothesis 2.1: Text Prompt Controls Visual Attention

### Interpretation Criteria

**Hypothesis SUPPORTED** (Prompt controls attention):
- L2 Distance > 1.0
- Correlation < 0.7

**Hypothesis NOT SUPPORTED** (Prompt ignored):
- L2 Distance < 0.5
- Correlation > 0.9

## Results Summary

### Success Episodes

#### Layer 5

| Prompt | L2 Distance | Correlation | |Δ| | Samples |
|--------|-------------|-------------|------|----------|
| ✅ duck | 2.341 | 0.523 | 0.156 | 45 |
| ✅ banana | 2.198 | 0.567 | 0.142 | 45 |
| ✅ cat | 2.567 | 0.489 | 0.178 | 45 |
| ❌ bottle | 0.834 | 0.812 | 0.089 | 45 |

...

## Overall Findings

### Success Episodes
- Average L2 Distance: 2.123 ± 0.456
- Average Correlation: 0.543 ± 0.123
- **Hypothesis 2.1**: ✅ SUPPORTED

### Failure Episodes
- Average L2 Distance: 1.987 ± 0.389
- Average Correlation: 0.589 ± 0.145
- **Hypothesis 2.1**: ✅ SUPPORTED

### Prompt Effectiveness Ranking

Ranked by L2 Distance (higher = stronger prompt control):

1. **cat**: 2.567 - "find the cat toy and pick it up"
2. **duck**: 2.341 - "find the duck toy and pick it up"
3. **banana**: 2.198 - "find the banana and pick it up"
4. **bottle**: 0.834 - "find the bottle and pick it up"
```

#### 关键发现解读

**Hypothesis 支持度判断**:
- ✅ **SUPPORTED**: L2 > 1.0 且 Correlation < 0.7
  - 说明 prompt 变化显著改变了 attention 分布
  - VLM 真正理解并响应语义变化
  
- ❌ **NOT SUPPORTED**: L2 < 0.5 或 Correlation > 0.9
  - 说明 prompt 变化对 attention 影响很小
  - VLM 主要依赖视觉特征，忽略语义

**Prompt 效果排名**:
- 排名越高，说明该 prompt 对 attention 的控制力越强
- 可以发现哪些物体名称更容易引起 attention 转移
- 可能反映训练数据中的物体分布

#### 自动运行

在 `main()` 函数最后自动调用：

```python
# Counterfactual aggregation
if ENABLE_COUNTERFACTUAL:
    print("\n" + "=" * 60)
    print("Generating multi-episode aggregated analysis (Counterfactual)...")
    print("=" * 60)
    aggregate_counterfactual_analysis(
        RESULTS_ROOT, COUNTERFACTUAL_LAYERS, COUNTERFACTUAL_PROMPTS, output_dir=RESULTS_ROOT
    )
```

## 完整工作流程

### 1. 单 Episode 处理

对每个 episode 的每一帧：
```
Frame 50:
  ├── Object Detection (如果启用)
  │   └── 生成 object/{frame:05d}/L*.jpg
  └── Counterfactual Analysis (如果启用)
      └── 生成 counterfactual/{frame:05d}/L*_*.png
```

### 2. Episode 级汇总

每个 episode 结束后：
```
Episode Results:
  ├── h1_1_obj_attn_results.json      # Object detection
  ├── h1_1_obj_attn_summary.png
  ├── h2_1_counterfactual_results.json # Counterfactual
  └── h2_1_counterfactual_report.md
```

### 3. Multi-Episode 聚合

所有 episode 处理完后：
```
Aggregated Results:
  ├── multi_episode_comparison.png     # Object detection aggregate
  ├── multi_episode_report.md
  └── counterfactual_aggregate/        # Counterfactual aggregate
      ├── cf_l2_distance_comparison.png
      ├── cf_correlation_comparison.png
      ├── cf_l2_heatmap.png
      ├── cf_multi_episode_report.md
      └── cf_multi_episode_aggregate.json
```

## 未来扩展

1. **多物体场景**: 测试场景中有多个物体时的 attention 转移
2. **属性变化**: 不仅改变物体名称，还改变颜色、大小等属性
3. **空间关系**: 测试 "left/right", "above/below" 等空间关系词
4. **动作变化**: "pick up" vs "push" vs "move" 等动作词的影响
5. **跨 Episode 对比**: 比较不同任务、不同物体的 prompt 控制能力
6. **时序分析**: 分析 attention 转移的时间动态

## 参考

- 原始实现: `viz/h2_1_location_prompt.py`
- Pipeline 集成: `viz/object_pipeline.py`
- Object Detection: `viz/h1_1_object_detection.py`
- 相关文档: `docs/1224_h1_1_object_detection.md`

