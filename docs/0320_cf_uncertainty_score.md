# Counterfactual Uncertainty Score — Design Doc

**Date**: 2026-03-20
**Status**: Planning

---

## Background

This repo already runs counterfactual prompts during batch inference (`pipeline.py`) and saves one HDF5 per (frame, prompt). Each file contains `pred_action` (8×8 float32) and `prefix/layer_{i}/text_to_img` attention. The data pipeline is complete; what is missing is a way to **aggregate those runs into interpretable uncertainty metrics**.

"Memory-related task" in the VLA context means: how much does the model rely on the language instruction versus pure visual input? A model that "remembers" the instruction well should change its predicted actions significantly when the instruction is removed or negated.

---

## Goals

1. Quantify **how uncertain** the model is about which action to take, as a function of prompt variation.
2. Quantify **how much the model relies on the instruction** (memory reliance).
3. Surface high-uncertainty frames in the dashboard to guide data-science investigation.

---

## Score Definitions

### 1. Memory Reliance Score

**Question**: If the instruction is removed, how much does the predicted action change?

**Formula**:
```
memory_reliance(frame) = || pred_action_baseline − pred_action_empty ||_F
```
where `||·||_F` is the Frobenius norm over the (8 steps × 8 dims) action matrix.

- High score → model heavily depends on the text instruction (good memory)
- Low score → model ignores the instruction (vision-only behavior)

**Requires**: a `key: empty` entry in `counterfactual.yaml` (already present).

---

### 2. Instruction Sensitivity Score

**Question**: Across all counterfactual prompts, how much does the predicted action vary?

**Formula**:
```
sensitivity(frame) = mean( Var_{prompts}( pred_action ) )
```
Variance is taken over the prompt dimension, then averaged over action dimensions.

- High score → model is uncertain / sensitive to wording
- Low score → model is robust to prompt variation

---

### 3. Attention Shift Score

**Question**: When the prompt changes, how much does the image attention pattern shift?

**Formula**:
```
attn_shift(frame, slug) = || mean_heads(text_to_img_baseline) − mean_heads(text_to_img_cf) ||_2
```
Computed per (frame, CF-prompt) pair, using a fixed representative layer (default: layer 8).

- High shift → the model is looking at very different image patches
- Useful for understanding *where* the uncertainty comes from (camera, object region, etc.)

---

### 4. (Optional) KL Divergence Version of Attention Shift

For a more probabilistic interpretation, normalize each attention vector to a distribution and compute:
```
attn_kl(frame, slug) = KL( softmax(attn_baseline) || softmax(attn_cf) )
```

---

## Data Flow

```
pipeline.py  (already runs CF prompts)
    ├── {frame:05d}.h5              → pred_action (baseline)
    └── {frame:05d}_{key}.h5       → pred_action + text_to_img (per CF prompt)
                ↓
    uncertainty.py  (new — aggregates scores)
                ↓
    uncertainty_view.py  (new — Streamlit tab)
```

All required data already exists in the HDF5 files after a normal batch pipeline run. No changes to inference are needed.

---

## New Files to Create

| File | Purpose |
|------|---------|
| `viz/dashboard/uncertainty.py` | Score computation: load CF H5s, compute the three metrics, return structured results |
| `viz/dashboard/views/uncertainty_view.py` | Streamlit tab: time-series charts, per-prompt heatmap, top-N uncertain frames table |

## Existing Files to Modify

| File | Change |
|------|--------|
| `viz/dashboard/app.py` | Register the new "Uncertainty" tab in Results mode |
| `viz/config/counterfactual.yaml` | Verify `empty` and `negation` entries are present (already are) |

---

## Dashboard View Spec

The new **Uncertainty** tab (Results mode only) should show:

### Panel A — Episode-level Time Series
- X axis: frame index
- Y axis (multi-line): `memory_reliance`, `sensitivity` (normalized 0–1 for comparability)
- Purpose: identify which frames are most uncertain across the episode

### Panel B — Per-prompt Action Heatmap
- Rows: CF prompt keys (cube, pen, empty, negation, …)
- Columns: action dimensions (j0–j6, gripper)
- Cell value: `|| pred_action_cf − pred_action_baseline ||` per dimension
- Purpose: understand which prompts and which action dims drive the most change

### Panel C — Top-N Uncertain Frames
- Sortable table: frame idx, memory_reliance, sensitivity, top-shifting prompt
- Clicking a row navigates to that frame in the existing Counterfactual tab

### Panel D — Attention Shift Map (per-prompt)
- For the currently selected frame: side-by-side 16×16 delta heatmaps (exterior + wrist)
- Shows *where* on the image the attention shifted for each CF prompt

---

## Open Questions

1. **Which layer to use for attention shift?** Layer 8 is a reasonable default (mid-network), but we could expose a slider or compute the mean across all 18 layers.
2. **Normalization**: Should scores be normalized per-episode or globally across all episodes in RESULTS_ROOT? Global normalization makes cross-episode comparison easier.
3. **Grouping by CF method**: `counterfactual.yaml` has a `method` field (`object_swap`, `negation`, `empty`, etc.). The sensitivity score could be broken down by method group for richer analysis.
4. **Trajectory-level aggregation**: Should we expose an episode-level scalar (e.g., mean sensitivity across all frames) for ranking episodes? This would be useful in the Results browser.
 
● 基于对代码库的深入阅读，以下是构建 Counterfactual Uncertainty
   Score（反事实不确定性分数） 用于 memory-related
  任务的开发方案（用中文回答）：

  ---
  核心思路

  "Memory-related task" 在这个 VLA
  模型语境下，指模型对语言指令的记忆与依赖程度。通过对比
  baseline prompt 和各种 counterfactual
  prompt（空指令、否定指令、物体替换等）下的预测动作 +
  注意力分布，可以量化模型的不确定性。

  ---
  三个核心分数定义

  1. Memory Reliance Score（指令依赖分）

  ▎ "如果我把指令清空，预测动作变化多大？"

  # 对比 baseline vs empty prompt 的 pred_action
  baseline_action = load_pred_action(h5_path(ckpt, ep, frame))
  empty_action    = load_pred_action(h5_path_cf(ckpt, ep,
  frame, "empty"))
  memory_reliance = np.linalg.norm(baseline_action -
  empty_action)  # (8,8) → scalar

  - 高分 = 模型高度依赖指令（有记忆）
  - 低分 = 模型忽略指令（靠视觉走）

  ---
  2. Instruction Sensitivity Score（指令敏感度）

  ▎ "在所有 CF prompts 下，预测动作的方差有多大？"

  # 汇聚所有 CF prompt 的 pred_action
  cf_actions = []
  for slug in list_cf_slugs(ckpt, ep, frame):
      cf_actions.append(load_pred_action(h5_path_cf(ckpt, ep,
  frame, slug)))
  cf_actions = np.stack(cf_actions)       # (n_prompts, 8, 8)
  sensitivity = cf_actions.var(axis=0).mean()  # scalar

  - 高分 = 模型不确定应该执行哪个动作

  ---
  3. Attention Shift Score（注意力漂移分）

  ▎ "换了 prompt 之后，模型看哪里发生了多大变化？"

  # text→image 注意力均值 (8, 512) 均值过 heads
  baseline_attn = load_text_to_img(h5_path(...),
  layer=8).mean(axis=(0, 1))  # (512,)
  cf_attn       = load_text_to_img(h5_path_cf(..., slug),
  layer=8).mean(axis=(0, 1))
  attn_shift    = np.linalg.norm(baseline_attn - cf_attn)

  ---
  开发步骤

  Step 1 — 新建 viz/dashboard/uncertainty.py

  集中计算上述三个分数：

  # viz/dashboard/uncertainty.py
  from viz.dashboard.loader import (
      load_pred_action, load_text_to_img, h5_path, h5_path_cf,
  list_cf_slugs
  )

  def compute_uncertainty_scores(ckpt, ep, frame,
  attn_layer=8):
      """
      Returns dict:
        memory_reliance: float
        sensitivity: float
        attn_shift: dict[slug → float]
      """
      ...

  Step 2 — 修改 viz/pipeline.py

  批量推理已支持 CF prompts（每帧都会写
  {frame}_{cf_key}.h5），数据已经有了，不需要修改推理部分。

  只需确认 counterfactual.yaml 里包含 empty 这条
  prompt（已有）。

  Step 3 — 新建 viz/dashboard/views/uncertainty_view.py

  在 dashboard 中增加 "Uncertainty" tab，显示：

  [时序折线图] 每帧的 memory_reliance / sensitivity score
  [热力图]    各 CF prompt × 各动作维度 的 pred_action 偏移
  [摘要表]    最不确定的 Top-N 帧 + 对应 CF prompt

  Step 4 — 接入 viz/dashboard/app.py

  在 sidebar 的 "Results" 模式下新增 tab，与现有 Trajectory tab
   并列。

  ---
  关键数据流

  pipeline.py 批量推理
      ├── {frame:05d}.h5          → baseline pred_action
      └── {frame:05d}_{key}.h5   → CF pred_action (已存在)
              ↓
  uncertainty.py 计算分数
              ↓
  uncertainty_view.py 可视化

  ---
  文件修改清单

  ┌──────────────────────────────────┬─────┬───────────────┐
  │               文件               │ 操  │     说明      │
  │                                  │ 作  │               │
  ├──────────────────────────────────┼─────┼───────────────┤
  │ viz/dashboard/uncertainty.py     │ 新  │ 三个分数计算  │
  │                                  │ 建  │ 逻辑          │
  ├──────────────────────────────────┼─────┼───────────────┤
  │ viz/dashboard/views/uncertainty_ │ 新  │ Streamlit     │
  │ view.py                          │ 建  │ 可视化 tab    │
  ├──────────────────────────────────┼─────┼───────────────┤
  │ viz/dashboard/app.py             │ 小  │ 注册新 tab    │
  │                                  │ 改  │               │
  ├──────────────────────────────────┼─────┼───────────────┤
  │                                  │ 确  │ 确保有 empty  │
  │ viz/config/counterfactual.yaml   │ 认  │ + negation    │
  │                                  │     │ 条目          │
  └──────────────────────────────────┴─────┴───────────────┘

  ---
  可选扩展

  - Per-layer 分析：在不同 transformer layer 上比较
  attn_shift，看哪层对指令最敏感
  - Trajectory-level 聚合：对整条 episode
  的不确定性做时序可视化（类似 trajectory.py）
  - KL 散度版本：用 KL(baseline_attn ∥ cf_attn) 替代
  L2，更符合概率意义

  需要我开始写某个具体模块的代码吗？

✻ Sautéed for 1m 29s

❯ put these plan into a doc, do not write code first           

  Read 1 file (ctrl+o to expand)    
                                                
● Write(docs/0320_counterfactual_uncertainty_score.md)
  ⎿  Wrote 144 lines to                               
     docs/0320_counterfactual_uncertainty_score.md
       1 # Counterfactual Uncertainty Score — Design Doc       
       2 
       3 **Date**: 2026-03-20                                  
       4 **Status**: Planning                               
       5                                                       
       6 ---                                                
       7
       8 ## Background
       9
      10 This repo already runs counterfactual prompts d
         uring batch inference (`pipeline.py`) and saves
          one HDF5 per (frame, prompt). Each file contai
         ns `pred_action` (8×8 float32) and `prefix/laye
         r_{i}/text_to_img` attention. The data pipeline
          is complete; what is missing is a way to **agg
         regate those runs into interpretable uncertaint
         y metrics**.
     … +134 lines (ctrl+o to expand)

● 文档已创建：docs/0320_counterfactual_uncertainty_score.md

  内容包括：                                                   
  - 背景 — 说明现有数据流，明确哪些已有、哪些缺失
  - 三个分数定义 — Memory Reliance、Instruction                
  Sensitivity、Attention Shift（含可选 KL 版本）            
  - 数据流图 — 说明不需要修改推理代码                          
  - 文件清单 — 新建 2 个 / 修改 2 个                        
  - Dashboard 视图规格 — Panel A~D 的具体内容                  
  - Open Questions — layer 选择、归一化策略、按 method         
  分组、episode 级聚合                                         
                                                               
  确认方向没问题后可以开始写代码。                             
                                 