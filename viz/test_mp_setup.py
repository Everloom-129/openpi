#!/usr/bin/env python3
"""Quick test script to verify multi-process GPU isolation works correctly.

Launches N workers and checks that each sees only 1 GPU (via CUDA_VISIBLE_DEVICES).
Does NOT load heavy models — just tests the CUDA context isolation.

Usage:
    uv run python viz/test_mp_setup.py --workers 2 --gpus 0,1
    uv run python viz/test_mp_setup.py --workers 1 --gpus 1
"""
import argparse
import multiprocessing as mp
import time
import sys


def test_worker(worker_id: int, gpu_id: int, result_queue: mp.Queue) -> None:
    """Test worker: verify GPU isolation."""
    import os
    
    # Set GPU BEFORE any CUDA imports
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Now import torch
    try:
        import torch
    except ImportError:
        result_queue.put({
            "worker_id": worker_id,
            "gpu_id": gpu_id,
            "success": False,
            "error": "torch not installed",
        })
        return
    
    try:
        # Check isolation
        if not torch.cuda.is_available():
            result_queue.put({
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "success": False,
                "error": "CUDA not available",
            })
            return
        
        visible_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if visible_count > 0 else "N/A"
        
        # Allocate a small tensor to verify GPU access
        torch.cuda.set_device(0)
        test_tensor = torch.zeros(1000, 1000, device="cuda:0")
        mem_allocated = torch.cuda.memory_allocated(0) / (1024**2)  # MB
        del test_tensor
        torch.cuda.empty_cache()
        
        success = (visible_count == 1)
        
        result_queue.put({
            "worker_id": worker_id,
            "gpu_id": gpu_id,
            "success": success,
            "visible_count": visible_count,
            "device_name": device_name,
            "mem_test": f"{mem_allocated:.1f}MB allocated",
            "error": None if success else f"Expected 1 GPU, got {visible_count}",
        })
        
    except Exception as e:
        result_queue.put({
            "worker_id": worker_id,
            "gpu_id": gpu_id,
            "success": False,
            "error": str(e),
        })


def main():
    parser = argparse.ArgumentParser(description="Test multi-process GPU setup")
    parser.add_argument("--workers", type=int, default=2, help="Number of workers")
    parser.add_argument("--gpus", type=str, default="0,1", help="Comma-separated GPU IDs")
    args = parser.parse_args()
    
    gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
    num_workers = args.workers
    
    print("=" * 60)
    print("Multi-Process GPU Isolation Test")
    print("=" * 60)
    print(f"Workers: {num_workers}")
    print(f"GPUs:    {gpu_ids}")
    print()
    
    # Setup multiprocessing
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    
    manager = mp.Manager()
    result_queue = manager.Queue()
    
    # Launch workers
    print("Launching workers...")
    processes = []
    for worker_id in range(num_workers):
        gpu_id = gpu_ids[worker_id % len(gpu_ids)]
        p = mp.Process(
            target=test_worker,
            args=(worker_id, gpu_id, result_queue),
        )
        p.start()
        processes.append(p)
        print(f"  Worker {worker_id} → GPU {gpu_id}")
    
    # Wait and collect results
    for p in processes:
        p.join(timeout=10)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    all_success = True
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    results.sort(key=lambda x: x["worker_id"])
    
    for res in results:
        wid = res["worker_id"]
        gid = res["gpu_id"]
        
        if res["success"]:
            print(f"✓ Worker {wid} (GPU {gid}): PASS")
            print(f"    Device: {res['device_name']}")
            print(f"    Visible GPUs: {res['visible_count']}")
            print(f"    Memory test: {res['mem_test']}")
        else:
            print(f"✗ Worker {wid} (GPU {gid}): FAIL")
            print(f"    Error: {res['error']}")
            all_success = False
    
    print("\n" + "=" * 60)
    if all_success:
        print("✓ ALL TESTS PASSED")
        print("  GPU isolation is working correctly.")
        print("  You can now run: bash viz/process_toy_mp.sh")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        print("  GPU isolation is NOT working.")
        print("  Do NOT run the full pipeline yet!")
        print("\n  Possible causes:")
        print("    1. torch imported before CUDA_VISIBLE_DEVICES set")
        print("    2. GPU IDs invalid or not accessible")
        print("    3. CUDA driver issue")
        sys.exit(1)


if __name__ == "__main__":
    main()
