"""Producer-consumer pipeline with single VLA instance.

Design goals:
1) Keep only ONE policy model instance in memory.
2) Use thread-based workers for disk I/O (frame loading + HDF5 writing).
3) Keep model inference single-threaded to avoid CUDA contention.

Stages:
    producer (I/O threads): load frame data -> infer_queue
    consumer (single thread): policy.infer + capture attn -> write_queue
    writers (I/O threads): write HDF5 to disk
"""
from __future__ import annotations

import argparse
import dataclasses
import queue
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from attn_h5_writer import write_attn_h5_from_buffer
from attn_map import get_keyframes, get_policy, select_best_gpu
from pipeline import (
    CAMERA,
    DEFAULT_CF_CONFIG,
    DEFAULT_CHECKPOINT,
    OPEN_LOOP_HORIZON,
    get_video_length,
    load_cf_config,
    load_example,
)


@dataclasses.dataclass(frozen=True)
class EpisodeInfo:
    episode_key: str
    data_dir: Path
    episode_dir: Path
    outcome: str
    date: str
    episode_id: str
    total_frames: int
    keyframes: list[int]


@dataclasses.dataclass(frozen=True)
class FramePlan:
    episode_key: str
    data_dir: Path
    frame_idx: int
    outputs: list[tuple[Path, str]]
    # outputs: [(h5_path, prompt_text), ...]


@dataclasses.dataclass(frozen=True)
class InferTask:
    episode_key: str
    frame_idx: int
    h5_path: Path
    prompt: str
    example: dict


@dataclasses.dataclass(frozen=True)
class WriteTask:
    episode_key: str
    frame_idx: int
    h5_path: Path
    prompt: str
    example: dict
    attn_buffer: dict


def collect_work(
    data_root: Path, results_root: Path, cf_prompts: list[dict]
) -> tuple[dict[str, EpisodeInfo], list[FramePlan], int]:
    """Collect episodes and planned frame outputs.

    Returns:
      - episode metadata map
      - frame plans that still need processing (missing h5 only)
      - number of episodes skipped by marker
    """
    episodes: dict[str, EpisodeInfo] = {}
    plans: list[FramePlan] = []
    skipped_by_marker = 0

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
                marker = episode_dir / "pi05.md"
                if marker.exists():
                    skipped_by_marker += 1
                    continue

                total_frames = get_video_length(data_dir)
                if total_frames <= 0:
                    continue
                keyframes = get_keyframes(total_frames, OPEN_LOOP_HORIZON)

                episode_key = str(rel_path)
                episodes[episode_key] = EpisodeInfo(
                    episode_key=episode_key,
                    data_dir=data_dir,
                    episode_dir=episode_dir,
                    outcome=outcome,
                    date=date_dir.name,
                    episode_id=episode_id,
                    total_frames=total_frames,
                    keyframes=keyframes,
                )

                for frame_idx in keyframes:
                    frame_dir = episode_dir / f"{frame_idx:05d}"
                    frame_dir.mkdir(exist_ok=True)

                    outputs: list[tuple[Path, str]] = []
                    main_h5 = frame_dir / f"{frame_idx:05d}.h5"
                    if not main_h5.exists():
                        instruction_path = data_dir / "instruction.txt"
                        prompt = instruction_path.read_text().strip() if instruction_path.exists() else ""
                        outputs.append((main_h5, prompt))

                    for cf in cf_prompts:
                        cf_h5 = frame_dir / f"{frame_idx:05d}_{cf['key']}.h5"
                        if not cf_h5.exists():
                            outputs.append((cf_h5, cf["prompt"]))

                    if outputs:
                        plans.append(
                            FramePlan(
                                episode_key=episode_key,
                                data_dir=data_dir,
                                frame_idx=frame_idx,
                                outputs=outputs,
                            )
                        )
    return episodes, plans, skipped_by_marker


def run_infer(policy, example: dict) -> dict:
    """Single inference with in-RAM attention capture."""
    from openpi.models_pytorch import gemma_pytorch as _gpt

    _gpt.enable_attn_buffer()
    try:
        _ = policy.infer(example)
        return _gpt.get_attn_buffer() or {}
    finally:
        _gpt.clear_attn_buffer()


def write_marker(ep: EpisodeInfo, cf_prompts: list[dict]) -> None:
    marker = ep.episode_dir / "pi05.md"
    marker.write_text(
        f"# Processing Complete\n\n"
        f"Episode: {ep.episode_id}\n"
        f"Outcome: {ep.outcome}\n"
        f"Date: {ep.date}\n"
        f"Total Frames: {ep.total_frames}\n"
        f"Keyframes: {ep.keyframes}\n"
        f"Counterfactuals: {[c['key'] for c in cf_prompts]}\n"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Single-model producer-consumer Pi0.5 attention pipeline"
    )
    parser.add_argument("data_root", help="Root dir with success/ and failure/ subdirs")
    parser.add_argument("results_root", help="Output root directory")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cf-config", default=DEFAULT_CF_CONFIG, help="Counterfactual YAML config")
    parser.add_argument("--no-counterfactual", dest="counterfactual", action="store_false")
    parser.add_argument("--camera", default=CAMERA, choices=("left", "right"))
    parser.add_argument(
        "--io-workers",
        type=int,
        default=8,
        help="Number of threads for disk I/O (frame loading and HDF5 writing).",
    )
    parser.add_argument(
        "--max-inflight-loads",
        type=int,
        default=32,
        help="Max number of frame-load jobs in flight.",
    )
    parser.add_argument(
        "--infer-queue-size",
        type=int,
        default=128,
        help="Bounded queue size between producer and inference consumer.",
    )
    parser.add_argument(
        "--write-queue-size",
        type=int,
        default=128,
        help="Bounded queue size between inference consumer and write workers.",
    )
    args = parser.parse_args(argv)

    data_root = Path(args.data_root)
    results_root = Path(args.results_root)
    cf_prompts = load_cf_config(args.cf_config) if args.counterfactual else []
    if cf_prompts:
        print(f"Loaded {len(cf_prompts)} counterfactual prompts from {args.cf_config}")

    episodes, frame_plans, skipped_marker = collect_work(data_root, results_root, cf_prompts)
    if not episodes:
        print("No episodes to process.")
        return

    expected_outputs_by_episode = {k: 0 for k in episodes}
    for fp in frame_plans:
        expected_outputs_by_episode[fp.episode_key] += len(fp.outputs)

    # Episodes that have no missing outputs: write marker immediately.
    processed_early = 0
    for ep_key, ep in episodes.items():
        if expected_outputs_by_episode[ep_key] == 0:
            write_marker(ep, cf_prompts)
            processed_early += 1

    device_id = select_best_gpu()
    device = f"cuda:{device_id}"
    print(f"Loading one policy instance on {device} ...")
    policy = get_policy(args.checkpoint, device=device)
    is_pi05 = bool(getattr(getattr(policy, "_model", None), "pi05", True))
    print(f"Policy loaded (pi05={is_pi05}).\n")

    infer_q: queue.Queue[InferTask | None] = queue.Queue(maxsize=args.infer_queue_size)
    write_q: queue.Queue[WriteTask | None] = queue.Queue(maxsize=args.write_queue_size)

    lock = threading.Lock()
    completed_outputs_by_episode = {k: 0 for k in episodes}
    failed_outputs_by_episode = {k: 0 for k in episodes}
    error_count = 0

    def producer() -> None:
        nonlocal error_count
        if not frame_plans:
            infer_q.put(None)
            return

        def load_for_plan(plan: FramePlan) -> tuple[FramePlan, dict]:
            ex = load_example(plan.data_dir, plan.frame_idx, camera=args.camera)
            return plan, ex

        with ThreadPoolExecutor(max_workers=max(1, args.io_workers)) as ex_pool:
            plan_iter = iter(frame_plans)
            inflight = set()

            # fill initial inflight
            while len(inflight) < max(1, args.max_inflight_loads):
                try:
                    p = next(plan_iter)
                except StopIteration:
                    break
                inflight.add(ex_pool.submit(load_for_plan, p))

            while inflight:
                done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                for fut in done:
                    try:
                        plan, example = fut.result()
                        for h5_path, prompt in plan.outputs:
                            infer_q.put(
                                InferTask(
                                    episode_key=plan.episode_key,
                                    frame_idx=plan.frame_idx,
                                    h5_path=h5_path,
                                    prompt=prompt,
                                    example=example,
                                )
                            )
                    except Exception as e:
                        with lock:
                            error_count += 1
                        print(f"[producer] load error: {e}")

                    try:
                        p = next(plan_iter)
                        inflight.add(ex_pool.submit(load_for_plan, p))
                    except StopIteration:
                        pass

        infer_q.put(None)

    def infer_consumer() -> None:
        nonlocal error_count
        while True:
            task = infer_q.get()
            if task is None:
                infer_q.task_done()
                break
            try:
                example = {**task.example, "prompt": task.prompt}
                attn_buffer = run_infer(policy, example)
                write_q.put(
                    WriteTask(
                        episode_key=task.episode_key,
                        frame_idx=task.frame_idx,
                        h5_path=task.h5_path,
                        prompt=task.prompt,
                        example=example,
                        attn_buffer=attn_buffer,
                    )
                )
            except Exception as e:
                with lock:
                    error_count += 1
                    failed_outputs_by_episode[task.episode_key] += 1
                print(f"[infer] {task.h5_path.name} error: {e}")
            finally:
                infer_q.task_done()

        # stop all writer threads
        for _ in range(max(1, args.io_workers)):
            write_q.put(None)

    def writer_worker(worker_id: int) -> None:
        nonlocal error_count
        while True:
            task = write_q.get()
            if task is None:
                write_q.task_done()
                break
            try:
                ok = write_attn_h5_from_buffer(
                    attn_buffer=task.attn_buffer,
                    h5_path=task.h5_path,
                    ext_img=task.example["observation/exterior_image_1_left"],
                    wrist_img=task.example["observation/wrist_image_left"],
                    instruction=task.prompt,
                    frame_idx=task.frame_idx,
                    is_pi05=is_pi05,
                )
                with lock:
                    if ok:
                        completed_outputs_by_episode[task.episode_key] += 1
                    else:
                        failed_outputs_by_episode[task.episode_key] += 1
                print(f"[writer-{worker_id}] {task.h5_path.name} {'✓' if ok else 'x'}")
            except Exception as e:
                with lock:
                    error_count += 1
                    failed_outputs_by_episode[task.episode_key] += 1
                print(f"[writer-{worker_id}] {task.h5_path.name} error: {e}")
            finally:
                write_q.task_done()

    print(
        f"Producer-consumer config: io_workers={args.io_workers}, "
        f"max_inflight_loads={args.max_inflight_loads}"
    )
    print(f"Episodes: {len(episodes)} | Frame plans: {len(frame_plans)}\n")

    t0 = time.perf_counter()
    t_producer = threading.Thread(target=producer, name="producer", daemon=True)
    t_infer = threading.Thread(target=infer_consumer, name="infer-consumer", daemon=True)
    writer_threads = [
        threading.Thread(target=writer_worker, args=(i,), name=f"writer-{i}", daemon=True)
        for i in range(max(1, args.io_workers))
    ]

    for t in writer_threads:
        t.start()
    t_infer.start()
    t_producer.start()

    t_producer.join()
    infer_q.join()
    t_infer.join()
    write_q.join()
    for t in writer_threads:
        t.join()

    processed = processed_early
    skipped = skipped_marker
    failed_episodes = 0

    for ep_key, ep in episodes.items():
        expected = expected_outputs_by_episode[ep_key]
        completed = completed_outputs_by_episode[ep_key]
        failed = failed_outputs_by_episode[ep_key]

        if expected == 0:
            continue
        if completed == expected and failed == 0:
            write_marker(ep, cf_prompts)
            processed += 1
        elif completed > 0:
            # partial success: keep old behavior (mark as done to avoid redoing successful files)
            write_marker(ep, cf_prompts)
            processed += 1
            failed_episodes += 1
        else:
            failed_episodes += 1

    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 50}")
    print(f"Total episodes discovered: {len(episodes) + skipped_marker}")
    print(f"Successfully processed:    {processed}")
    print(f"Skipped (marker exists):   {skipped}")
    print(f"Episode failures/partial:  {failed_episodes}")
    print(f"Output-level errors:       {error_count}")
    print(f"Results:                   {results_root}")
    print(f"Total wall time:           {elapsed:.1f}s")


if __name__ == "__main__":
    main()
