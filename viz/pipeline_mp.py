"""Multi-process batch inference pipeline for attention visualization.

Parallelizes at the episode level — each worker process handles different episodes
and is assigned to a specific GPU. This maximizes GPU utilization when processing
large datasets with many episodes.

Key improvements over pipeline.py:
  - N workers × N GPUs = N× throughput (理论上)
  - Automatic GPU load balancing
  - Shared progress tracking with multiprocessing.Manager
  - Each worker loads policy once and reuses for all episodes

CRITICAL DESIGN:
  - DELAYED IMPORTS: torch/CUDA modules are imported INSIDE worker processes
    AFTER setting CUDA_VISIBLE_DEVICES. This prevents CUDA context sharing bugs.
  - Main process only handles task collection and worker spawning.

Usage:
    # Auto-detect GPUs, use 1 worker per GPU
    uv run python viz/pipeline_mp.py <DATA_ROOT> <RESULTS_ROOT>
    
    # Manually specify number of workers (will cycle through available GPUs)
    uv run python viz/pipeline_mp.py <DATA_ROOT> <RESULTS_ROOT> --workers 4
    
    # Specify which GPUs to use
    uv run python viz/pipeline_mp.py <DATA_ROOT> <RESULTS_ROOT> --gpus 0,1,3
    
    # Skip counterfactual prompts
    uv run python viz/pipeline_mp.py <DATA_ROOT> <RESULTS_ROOT> --no-counterfactual
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

# ── Constants (safe to import, no CUDA) ────────────────────────────────────────
OPEN_LOOP_HORIZON: int = 8
DEFAULT_CHECKPOINT: str = "./checkpoints/viz/pi05_droid_pytorch"
DEFAULT_CF_CONFIG: str = str(Path(__file__).parent / "config" / "counterfactual.yaml")


# ── GPU utilities (main process, no CUDA) ──────────────────────────────────────

def get_available_gpus() -> list[int]:
    """Return list of GPU device IDs with sufficient memory (>25GB free).
    
    IMPORTANT: This checks current free memory. Be conservative!
    Each worker needs ~16-18GB (model + buffer + activations).
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        gpu_ids = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_gb = mem_info.free / (1024**3)
            total_gb = mem_info.total / (1024**3)
            used_gb = mem_info.used / (1024**3)
            
            # Very conservative: 25GB free minimum
            # (16GB model + 6GB buffer + 3GB safety margin)
            if free_gb > 25:
                gpu_ids.append(i)
                print(f"  GPU {i}: {free_gb:.1f}GB free / {total_gb:.1f}GB total → OK")
            else:
                print(f"  GPU {i}: {free_gb:.1f}GB free / {total_gb:.1f}GB total → SKIP ({used_gb:.1f}GB in use)")
        
        pynvml.nvmlShutdown()
        
        if not gpu_ids:
            print("\nWARNING: No GPU with >25GB free memory!")
            print("         You may need to kill other processes first.")
            print("         Attempting to use GPU 0 anyway (may OOM)...")
            return [0]
        
        return gpu_ids
        
    except Exception as e:
        print(f"WARNING: Failed to query GPU info: {e}")
        print("         Falling back to GPU 0")
        return [0]


# ── Worker process (CUDA imports happen here) ──────────────────────────────────

def worker_main(
    worker_id: int,
    gpu_id: int,
    task_queue: mp.Queue,
    checkpoint: str,
    cf_prompts: list[dict],
    shared_stats: dict,
    camera: str,
) -> None:
    """Main loop for a worker process.
    
    CRITICAL: All CUDA/torch imports must happen INSIDE this function,
    AFTER setting CUDA_VISIBLE_DEVICES. This ensures each worker gets
    an isolated CUDA context on the correct GPU.
    """
    import os
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 1: Set GPU BEFORE any CUDA imports
    # ═══════════════════════════════════════════════════════════════════════════
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[Worker {worker_id}] Starting on GPU {gpu_id}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 2: Now safe to import torch/CUDA modules
    # ═══════════════════════════════════════════════════════════════════════════
    import torch
    import h5py
    import numpy as np
    from PIL import Image
    
    # Import pipeline utilities (these import torch internally)
    from pipeline import get_video_length, load_example, infer_and_save
    from attn_map import get_keyframes, get_policy
    
    # Verify isolation
    if torch.cuda.is_available():
        actual_count = torch.cuda.device_count()
        if actual_count != 1:
            print(f"[Worker {worker_id}] ⚠️  WARNING: Expected 1 visible GPU, got {actual_count}")
        torch.cuda.set_device(0)  # Should be the only visible GPU
        print(f"[Worker {worker_id}] CUDA isolated: {torch.cuda.get_device_name(0)}")
    else:
        print(f"[Worker {worker_id}] ⚠️  WARNING: CUDA not available!")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 3: Load policy once (expensive, ~30s)
    # ═══════════════════════════════════════════════════════════════════════════
    device = "cuda:0"  # Always 0 due to CUDA_VISIBLE_DEVICES masking
    print(f"[Worker {worker_id}] Loading policy from {checkpoint} ...")
    policy = get_policy(checkpoint, device=device)
    print(f"[Worker {worker_id}] Policy loaded ✓\n")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 4: Process episodes from queue
    # ═══════════════════════════════════════════════════════════════════════════
    episodes_processed = 0
    while True:
        try:
            task = task_queue.get(timeout=1)
            if task is None:  # Poison pill
                print(f"[Worker {worker_id}] Shutting down (processed {episodes_processed} episodes)")
                break
            
            # Process one episode
            data_dir = task["data_dir"]
            episode_dir = task["episode_dir"]
            episode_id = task["episode_id"]
            outcome = task["outcome"]
            date = task["date"]
            
            # Check if already done
            marker = episode_dir / "pi05.md"
            if marker.exists():
                print(f"[W{worker_id}|GPU{gpu_id}] [skip] {outcome}/{date}/{episode_id}")
                with shared_stats["lock"]:
                    shared_stats["skipped"] += 1
                continue
            
            print(f"[W{worker_id}|GPU{gpu_id}] Processing {outcome}/{date}/{episode_id}")
            t0 = time.perf_counter()
            
            try:
                # Process all keyframes in this episode
                total_frames = get_video_length(data_dir)
                if total_frames == 0:
                    print(f"[W{worker_id}|GPU{gpu_id}]   no frames found, skipping")
                    continue
                
                keyframes = get_keyframes(total_frames, OPEN_LOOP_HORIZON)
                frame_ok = frame_skip = frame_err = 0
                
                for frame_idx in keyframes:
                    frame_dir = episode_dir / f"{frame_idx:05d}"
                    frame_dir.mkdir(exist_ok=True)
                    
                    try:
                        example = load_example(data_dir, frame_idx, camera=camera)
                        
                        # Main inference
                        h5_main = frame_dir / f"{frame_idx:05d}.h5"
                        if h5_main.exists():
                            frame_skip += 1
                        else:
                            infer_and_save(policy, example, h5_main, frame_idx)
                            print(f"[W{worker_id}|GPU{gpu_id}]   {frame_idx:05d}.h5 ✓")
                        
                        # Counterfactual prompts
                        for cf in cf_prompts:
                            h5_cf = frame_dir / f"{frame_idx:05d}_{cf['key']}.h5"
                            if h5_cf.exists():
                                continue
                            cf_example = {**example, "prompt": cf["prompt"]}
                            infer_and_save(policy, cf_example, h5_cf, frame_idx)
                            print(f"[W{worker_id}|GPU{gpu_id}]   {frame_idx:05d}_{cf['key']}.h5 ✓")
                        
                        frame_ok += 1
                        
                    except Exception as e:
                        print(f"[W{worker_id}|GPU{gpu_id}]   frame {frame_idx:05d} ERROR: {e}")
                        frame_err += 1
                
                # Write marker if any frames succeeded
                elapsed = time.perf_counter() - t0
                if frame_err > 0 and frame_ok == 0:
                    print(f"[W{worker_id}|GPU{gpu_id}]   all frames failed, skipping marker")
                    with shared_stats["lock"]:
                        shared_stats["errors"] += 1
                else:
                    marker.write_text(
                        f"# Processing Complete\n\n"
                        f"Episode: {episode_id}\n"
                        f"Outcome: {outcome}\n"
                        f"Date: {date}\n"
                        f"Total Frames: {total_frames}\n"
                        f"Keyframes: {keyframes}\n"
                        f"Counterfactuals: {[c['key'] for c in cf_prompts]}\n"
                    )
                    print(f"[W{worker_id}|GPU{gpu_id}]   done {elapsed:.0f}s  "
                          f"ok={frame_ok} skip={frame_skip} err={frame_err}")
                    with shared_stats["lock"]:
                        shared_stats["processed"] += 1
                
                episodes_processed += 1
                
            except Exception as e:
                import traceback
                print(f"[W{worker_id}|GPU{gpu_id}]   EPISODE ERROR: {e}")
                traceback.print_exc()
                with shared_stats["lock"]:
                    shared_stats["errors"] += 1
        
        except Exception as e:
            if "Empty" not in str(type(e).__name__):  # Ignore queue.Empty
                print(f"[Worker {worker_id}] Unexpected queue error: {e}")


# ── Task collection (main process, no CUDA) ────────────────────────────────────

def collect_tasks(data_root: Path, results_root: Path) -> list[dict]:
    """Collect all episodes that need processing.
    
    Returns list of task dicts with keys: data_dir, episode_dir, episode_id, outcome, date
    """
    import h5py
    
    tasks = []
    
    for outcome in ("success", "failure"):
        outcome_dir = data_root / outcome
        if not outcome_dir.exists():
            continue
        
        for date_dir in sorted(outcome_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            
            for traj_path in sorted(date_dir.rglob("trajectory.h5")):
                data_dir = traj_path.parent
                episode_id = data_dir.name
                
                rel_path = data_dir.relative_to(data_root)
                episode_dir = results_root / rel_path
                episode_dir.mkdir(parents=True, exist_ok=True)
                
                tasks.append({
                    "data_dir": data_dir,
                    "episode_dir": episode_dir,
                    "episode_id": episode_id,
                    "outcome": outcome,
                    "date": date_dir.name,
                })
    
    return tasks


def load_cf_config(config_path: str | Path) -> list[dict]:
    """Load counterfactual prompt list from YAML.
    
    Returns a list of dicts with keys: key, prompt, method.
    Returns [] if the file is missing or 'prompts' is absent.
    """
    import yaml
    p = Path(config_path)
    if not p.exists():
        print(f"[warn] CF config not found: {p} — skipping counterfactuals")
        return []
    with open(p) as f:
        data = yaml.safe_load(f)
    return data.get("prompts", [])


# ── Main entry point ───────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Multi-process Pi0.5 batch attention pipeline"
    )
    parser.add_argument("data_root", help="Root dir with success/ and failure/ subdirs")
    parser.add_argument("results_root", help="Output root directory")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cf-config", default=DEFAULT_CF_CONFIG,
                        help="Path to counterfactual YAML config")
    parser.add_argument("--no-counterfactual", dest="counterfactual",
                        action="store_false",
                        help="Skip counterfactual prompt inference")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes (default: 1 per GPU)")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs to use (e.g., '0,1,3')")
    parser.add_argument("--camera", type=str, default="right",
                        help="Camera side: 'right' or 'left' (default: right)")
    args = parser.parse_args(argv)
    
    DATA_ROOT = Path(args.data_root)
    RESULTS_ROOT = Path(args.results_root)
    
    # Load counterfactual config
    cf_prompts = load_cf_config(args.cf_config) if args.counterfactual else []
    if cf_prompts:
        print(f"Loaded {len(cf_prompts)} counterfactual prompts from {args.cf_config}")
        for cf in cf_prompts:
            print(f"  [{cf['method']}] {cf['key']!r}: {cf['prompt']!r}")
        print()
    
    # Determine GPU allocation
    print("=" * 60)
    print("GPU DETECTION")
    print("=" * 60)
    if args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
        print(f"Using user-specified GPUs: {gpu_ids}")
    else:
        gpu_ids = get_available_gpus()
        if not gpu_ids:
            print("\nERROR: No suitable GPUs found!")
            print("Try: python viz/pipeline_mp.py --gpus 0 ...")
            return
    
    # Determine number of workers
    if args.workers:
        num_workers = args.workers
    else:
        # Conservative default: 1 worker per GPU
        num_workers = len(gpu_ids)
    
    # Safety check: prevent too many workers
    # Each worker needs ~16-18GB. A6000 has 48GB → max 2 workers/GPU safely
    max_safe_workers = len(gpu_ids) * 2
    if num_workers > max_safe_workers:
        print(f"\n⚠️  WARNING: {num_workers} workers × 16GB = {num_workers * 16}GB memory needed")
        print(f"            but only {len(gpu_ids)} GPU(s) with ~48GB each available.")
        print(f"            Limiting to {max_safe_workers} workers (2 per GPU).")
        num_workers = max_safe_workers
    
    print(f"\n{'=' * 60}")
    print("WORKER CONFIGURATION")
    print(f"{'=' * 60}")
    print(f"  GPUs:            {gpu_ids}")
    print(f"  Workers:         {num_workers}")
    print(f"  Workers per GPU: ~{num_workers / len(gpu_ids):.1f}")
    print(f"  Camera:          {args.camera}")
    print()
    
    # Collect all tasks
    print("Collecting episodes...")
    tasks = collect_tasks(DATA_ROOT, RESULTS_ROOT)
    total_episodes = len(tasks)
    
    if total_episodes == 0:
        print("No episodes found!")
        return
    
    print(f"Found {total_episodes} episode(s) to process\n")
    
    # Warn if task count is low
    if total_episodes < num_workers:
        print(f"⚠️  WARNING: Only {total_episodes} episodes but {num_workers} workers.")
        print(f"            Some workers will be idle. Consider using --workers {total_episodes}")
        print()
    
    # Setup multiprocessing
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set
    
    manager = mp.Manager()
    task_queue = manager.Queue()
    
    # Shared stats
    shared_stats = manager.dict()
    shared_stats["lock"] = manager.Lock()
    shared_stats["processed"] = 0
    shared_stats["skipped"] = 0
    shared_stats["errors"] = 0
    
    # Populate task queue
    for task in tasks:
        task_queue.put(task)
    
    # Add poison pills (one per worker)
    for _ in range(num_workers):
        task_queue.put(None)
    
    print(f"{'=' * 60}")
    print("LAUNCHING WORKERS")
    print(f"{'=' * 60}")
    
    # Launch workers
    processes = []
    for worker_id in range(num_workers):
        gpu_id = gpu_ids[worker_id % len(gpu_ids)]  # Round-robin GPU assignment
        p = mp.Process(
            target=worker_main,
            args=(worker_id, gpu_id, task_queue, args.checkpoint, cf_prompts, shared_stats, args.camera),
        )
        p.start()
        processes.append(p)
        print(f"  Launched Worker {worker_id} → GPU {gpu_id}")
        time.sleep(0.5)  # Stagger startup to reduce memory spike
    
    print(f"\n{'=' * 60}")
    print("Workers are running... Press Ctrl+C to stop")
    print(f"{'=' * 60}\n")
    
    # Wait for all workers to finish
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Terminating workers...")
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(timeout=5)
        print("All workers terminated.")
        return
    
    # Final summary
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total episodes:         {total_episodes}")
    print(f"Successfully processed: {shared_stats['processed']}")
    print(f"Skipped (done):         {shared_stats['skipped']}")
    print(f"Errors:                 {shared_stats['errors']}")
    print(f"Results:                {RESULTS_ROOT}")


if __name__ == "__main__":
    t_start = time.perf_counter()
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\nTotal wall time: {time.perf_counter() - t_start:.1f}s")
