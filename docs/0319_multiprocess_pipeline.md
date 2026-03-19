# Multi-Process Attention Pipeline

## 概述

`viz/pipeline_mp.py` 是 `viz/pipeline.py` 的多进程增强版本，通过并行处理多个 episode 来充分利用多 GPU 资源。

## 核心改进

### 单进程 vs 多进程对比

| 特性 | `pipeline.py` (单进程) | `pipeline_mp.py` (多进程) |
|------|------------------------|---------------------------|
| **并行级别** | 无 | Episode 级别 |
| **GPU 利用** | 1个GPU，顺序处理 | N个GPU，并行处理 |
| **理论吞吐** | 1× | N× (N = worker数量) |
| **内存占用** | 1个模型实例 (~16GB) | N个模型实例 (~16N GB) |
| **适用场景** | 单GPU或小数据集 | 多GPU + 大数据集 (>10 episodes) |

### 并行策略

```
Task Queue (所有 episode)
    ↓
┌───────┬───────┬───────┬───────┐
│Worker0│Worker1│Worker2│Worker3│
│ GPU0  │ GPU1  │ GPU0  │ GPU1  │  ← Round-robin GPU 分配
└───┬───┴───┬───┴───┬───┴───┬───┘
    ↓       ↓       ↓       ↓
  ep_01   ep_02   ep_03   ep_04   ← 同时处理不同 episode
```

**为什么选择 Episode 级别并行？**
1. ✅ **无依赖**：每个 episode 完全独立
2. ✅ **粒度合适**：每个 episode 处理时间 5-30s，负载均衡好
3. ✅ **简单实现**：不需要复杂的同步机制
4. ❌ Frame 级别并行：需要 batch inference 改写，复杂度高

## 架构设计

### 进程模型

```python
Main Process
  │
  ├─ Collect all episodes → Task Queue
  │
  ├─ Spawn Worker 0 (GPU 0)
  │   ├─ Load policy once
  │   └─ Loop: get task → process_episode() → report
  │
  ├─ Spawn Worker 1 (GPU 1)
  │   └─ ...
  │
  └─ Wait all workers → Print summary
```

### 关键组件

#### 1. GPU 自动检测

```python
def get_available_gpus() -> list[int]:
    """返回有 >20GB 可用显存的 GPU ID 列表"""
    # 使用 pynvml 查询每个 GPU 的可用显存
    # 过滤掉显存不足的 GPU
```

#### 2. Worker 初始化

```python
def worker_init(gpu_id: int, checkpoint: str, rank: int):
    # 设置 CUDA_VISIBLE_DEVICES 隔离 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 加载 policy 到该 GPU (只加载一次！)
    policy = get_policy(checkpoint, device="cuda:0")
    
    return {"policy": policy, "gpu_id": gpu_id}
```

**关键点**：每个 worker 只加载模型**一次**，然后复用处理所有分配的 episode，避免重复加载开销。

#### 3. 任务分发

```python
# 主进程收集所有 episode
tasks = collect_tasks(DATA_ROOT, RESULTS_ROOT)

# 放入共享队列
for task in tasks:
    task_queue.put(task)

# 添加终止信号 (poison pills)
for _ in range(num_workers):
    task_queue.put(None)
```

#### 4. 进度共享

```python
# 使用 multiprocessing.Manager 共享统计
shared_stats = manager.dict()
shared_stats["lock"] = manager.Lock()
shared_stats["processed"] = 0

# Worker 更新时加锁
with shared_stats["lock"]:
    shared_stats["processed"] += 1
```

## 使用方法

### 基础用法

```bash
# 自动检测 GPU，每个 GPU 启动 1 个 worker
uv run python viz/pipeline_mp.py \
    /data/droid_episodes \
    /results/attn_h5

# 或使用 shell 脚本
bash viz/process_toy_mp.sh
```

### 高级用法

```bash
# 手动指定 4 个 worker（适合有2个GPU，每个GPU跑2个worker）
uv run python viz/pipeline_mp.py /data /results --workers 4

# 只使用 GPU 0 和 GPU 2
uv run python viz/pipeline_mp.py /data /results --gpus 0,2

# 8个 worker + 4个 GPU
uv run python viz/pipeline_mp.py /data /results --workers 8 --gpus 0,1,2,3

# 跳过反事实提示
uv run python viz/pipeline_mp.py /data /results --no-counterfactual
```

### 修改 shell 脚本配置

编辑 `viz/process_toy_mp.sh`：

```bash
# 手动设置 worker 数量
NUM_WORKERS=4

# 手动指定 GPU
GPUS="0,1"
```

## 性能分析

### 为什么原版 GPU 利用率低？

单进程版本的瓶颈分析（每帧处理时间分布）：

```
┌─────────────────────────────────────────┐
│ 磁盘读图像   GPU推理   CPU压缩写HDF5      │
│   100ms      50ms       150ms          │
│ ████████     ████       ████████████    │
└─────────────────────────────────────────┘
Total: ~300ms，GPU 只工作 17% 时间
```

**I/O 瓶颈**：
- 读取 JPG 图像：磁盘 I/O
- 写入 HDF5 + gzip 压缩：CPU 密集
- GPU 推理：反而最快

**结果**：GPU 大部分时间在等待，利用率 <10%

### 多进程如何提升？

```
Worker 0: [读图] [推理] [写盘] [读图] [推理] [写盘]
Worker 1:       [读图] [推理] [写盘] [读图] [推理]
Worker 2:             [读图] [推理] [写盘] [读图]
Worker 3:                   [读图] [推理] [写盘]
          ────────────────────────────────────►
                   时间轴
```

虽然单个 worker 的 GPU 利用率仍然低，但**多个 worker 的推理时间错开**，总体 GPU 利用率显著提升。

### 预期性能提升

| Worker 数 | GPU 配置 | 理论加速比 | 实际加速比* |
|----------|----------|-----------|------------|
| 1 | 1×GPU | 1.0× | 1.0× |
| 2 | 2×GPU | 2.0× | 1.7-1.9× |
| 4 | 2×GPU | 2.0× | 1.8-2.0× |
| 4 | 4×GPU | 4.0× | 3.2-3.6× |

*实际加速比受限于任务队列调度开销和 episode 处理时间差异（负载不均）

## 资源需求

### 显存需求

每个 worker 需要 **~16-18 GB** 显存：
- Pi0.5 模型权重：~8 GB
- KV cache + 激活值：~6 GB
- 注意力 buffer：~2 GB

**安全配置**：
- RTX A6000 (48GB) → 最多 2 workers/GPU
- A100 (80GB) → 最多 4 workers/GPU
- RTX 4090 (24GB) → 1 worker/GPU

### CPU 和内存

- 主机内存：每 worker ~5 GB（图像解码 + HDF5 压缩）
- CPU 核心：建议 ≥ 2×worker 数量（用于 I/O 并发）

## 监控和调试

### 实时监控

```bash
# 另一个终端运行 nvtop 监控 GPU
nvtop

# 或使用 watch + nvidia-smi
watch -n 1 nvidia-smi
```

**预期现象**：
- ✅ 显存占用稳定（每个 worker ~16GB）
- ✅ GPU 利用率间歇性跳到 20-60%（推理时）
- ✅ 功耗间歇性上升（推理时 100-150W）

### 进度追踪

Worker 会实时打印进度：

```
[W0|GPU0] Processing success/2024_01_15/episode_001
[W1|GPU1] Processing success/2024_01_15/episode_002
[W0|GPU0]   done 8s  ok=5 skip=0 err=0
[W1|GPU1]   done 12s  ok=7 skip=0 err=0
...
```

### 常见问题

#### 1. OOM (Out of Memory)

```
RuntimeError: CUDA out of memory
```

**解决**：
- 减少 worker 数量：`--workers 2`
- 或只用部分 GPU：`--gpus 0`

#### 2. 进程挂起

某个 worker 卡住不动（可能是数据损坏或模型 bug）。

**排查**：
```bash
# 查看哪个 worker 的 CPU 占用异常
htop -p $(pgrep -f pipeline_mp.py | tr '\n' ',' | sed 's/,$//')

# 如果必要，kill 特定 worker（其他 worker 继续）
kill <PID>
```

#### 3. GPU 分配冲突

```
[Worker 0] CUDA error: all CUDA-capable devices are busy
```

**原因**：你之前的 6 个进程还在占用 GPU。

**解决**：先清理不需要的进程（见下方）。

## 与现有流程集成

### 清理现有进程（可选）

你当前有 6 个进程占用 GPU，建议先清理：

```bash
# 1. 查看所有 Python GPU 进程
ps aux | grep python | grep -E "streamlit|pipeline|server"

# 2. 关闭不需要的 Streamlit (保留 1 个即可)
kill 1032822  # 或 3003816

# 3. 关闭其他模型服务器（如果不用）
kill 3269903  # GR00T server
kill 1963901  # Curobo server

# 这样能释放 GPU0: ~19GB，GPU1: ~6GB
```

### 推荐配置

**场景 1：你有 2 个 RTX A6000 (48GB each)**

```bash
# 清理后，每个 GPU 可跑 2 个 worker
uv run python viz/pipeline_mp.py /data /results --workers 4 --gpus 0,1
```

预期：
- GPU0: 2 workers × 16GB = 32GB 占用
- GPU1: 2 workers × 16GB = 32GB 占用
- GPU 利用率：20-40%（已经比原来的 5% 好很多）
- 加速比：~3.5×

**场景 2：保守配置（保留其他服务）**

```bash
# 只用 GPU1，启动 1 个 worker（避免干扰 GPU0 的其他服务）
uv run python viz/pipeline_mp.py /data /results --workers 1 --gpus 1
```

## 实际测试

用小数据集测试：

```bash
# 假设你的 toy_cube_benchmark 有 10 个 episode

# 单进程基准测试
time uv run python viz/pipeline.py /data /results --no-counterfactual

# 多进程测试 (2 workers)
time uv run python viz/pipeline_mp.py /data /results --no-counterfactual --workers 2

# 计算加速比
# speedup = single_process_time / multi_process_time
```

## 技术细节

### 为什么用 `spawn` 而不是 `fork`？

```python
mp.set_start_method("spawn", force=True)
```

- **spawn**：每个子进程独立启动，重新导入模块
- **fork**：复制父进程的内存空间（包括 CUDA 上下文）

PyTorch + CUDA 不支持 fork（会导致死锁），必须用 spawn。

### CUDA_VISIBLE_DEVICES 隔离

```python
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
device = "cuda:0"  # 总是 0，因为其他 GPU 被隐藏
```

每个 worker 只"看到"自己的 GPU，避免多进程竞争同一 GPU。

### 进程间通信

- **任务分发**：`multiprocessing.Queue`（线程安全）
- **统计共享**：`multiprocessing.Manager.dict()`
- **同步原语**：`Manager.Lock()`

### 优雅退出

```python
# 主进程捕获 Ctrl+C
try:
    main()
except KeyboardInterrupt:
    print("Interrupted by user. Exiting...")
```

Worker 会自动清理（因为 Python 进程退出时 CUDA 资源自动释放）。

## 局限性和未来改进

### 当前局限

1. **显存需求高**：N个worker = N×16GB
2. **冷启动慢**：每个 worker 都要加载模型（~30s）
3. **负载不均**：episode 长度不同，可能有 worker 提前空闲

### 可能的改进

#### 方案 A：Batch-level 并行（更复杂）

在单个 GPU 上用更大 batch size 处理多帧：

```python
# 一次推理 8 帧
batch_example = stack_examples([
    load_example(data_dir, 0),
    load_example(data_dir, 8),
    ...
])
result = policy.infer_batch(batch_example)  # 需要改写 policy
```

优点：
- 显存效率更高（共享模型权重）
- GPU 利用率更高（矩阵乘法更大）

缺点：
- 需要大幅改写 `policy.infer()` 和 `gemma_pytorch.py`
- Transformer 的 KV cache 机制不适合 batch（因为 sequence 长度不同）

#### 方案 B：预加载 + 异步写入

在当前架构下优化 I/O：

```python
# 预加载线程
image_queue = Queue(maxsize=16)
Thread(target=preload_images, args=(keyframes, image_queue)).start()

# 主循环
for frame_idx in keyframes:
    example = image_queue.get()  # 已经在内存中
    result = policy.infer(example)
    # 异步写入 HDF5（另一个线程）
    write_queue.put((buf, h5_path, ...))
```

预期提升：~20-30%（减少 GPU 等待时间）

#### 方案 C：GPU 共享（需要硬件支持）

使用 MPS (Multi-Process Service) 让多个进程共享一个 GPU：

```bash
# 启动 MPS daemon
nvidia-cuda-mps-control -d

# 然后运行多进程脚本（所有 worker 共享 GPU）
```

适用场景：单 GPU 系统，想并行多个轻量任务。

## 总结

当前多进程版本的 **最佳使用场景**：
- ✅ 有 2+ 个 GPU
- ✅ 有 10+ 个 episode 要处理
- ✅ 每个 episode 有足够的帧数（>10 帧）

如果只有 1 个 GPU 或 episode 数量很少（<5），单进程版本已经够用。
