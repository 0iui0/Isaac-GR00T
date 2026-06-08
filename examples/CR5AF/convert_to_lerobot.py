#!/usr/bin/env python3
"""Convert CR5AF .npz recordings to LeRobot v2 format for GR00T finetuning.

Processes episodes one at a time to avoid memory issues.

Usage:
    python examples/CR5AF/convert_to_lerobot.py \
        --input-dir recordings/cr5af_demos \
        --output-dir datasets/cr5af_pick_place \
        --fps 30
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def encode_video_mp4(frames: np.ndarray, output_path: str, fps: int = 30):
    """Encode (T, H, W, C) uint8 RGB frames to MP4."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    T, H, W, C = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    for i in range(T):
        writer.write(cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
    writer.release()


def compute_actions(states: np.ndarray) -> np.ndarray:
    """Compute actions from consecutive states.

    GR00T expects absolute actions (next state), then internally converts
    to relative during training based on the ActionConfig.

    action[t] = state[t+1] for t < T-1
    action[T-1] = state[T-1] (last frame: stay in place)
    """
    T = states.shape[0]
    actions = np.zeros((T, 16), dtype=np.float32)
    for t in range(T - 1):
        actions[t] = states[t + 1]
    # Last action = no movement (same as last state)
    actions[T - 1] = states[T - 1]
    return actions


def scan_episodes(input_dir: str):
    """Scan episode directories without loading data. Returns list of (ep_name, meta)."""
    episodes = []
    root = Path(input_dir)
    for ep_dir in sorted(root.glob("episode_*")):
        npz_path = ep_dir / "data.npz"
        meta_path = ep_dir / "meta.json"
        if npz_path.exists():
            meta = {}
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
            episodes.append((ep_dir.name, meta))
    return episodes


def streaming_stats():
    """Accumulate mean/std/min/max/q01/q99 using streaming method."""
    return {
        "n": 0,
        "sum": None,
        "sum_sq": None,
        "min": None,
        "max": None,
        "chunks_p1": [],
        "chunks_p99": [],
    }


def push_streaming(acc, chunk):
    """Push a (N, D) chunk into streaming accumulator."""
    N = chunk.shape[0]
    if acc["n"] == 0:
        acc["sum"] = chunk.sum(axis=0)
        acc["sum_sq"] = (chunk ** 2).sum(axis=0)
        acc["min"] = chunk.min(axis=0)
        acc["max"] = chunk.max(axis=0)
    else:
        acc["sum"] += chunk.sum(axis=0)
        acc["sum_sq"] += (chunk ** 2).sum(axis=0)
        acc["min"] = np.minimum(acc["min"], chunk.min(axis=0))
        acc["max"] = np.maximum(acc["max"], chunk.max(axis=0))
    acc["n"] += N
    # Keep percentile samples (at most 10000 per chunk for memory)
    if N > 0:
        sample_n = min(10000, N)
        idx = np.random.choice(N, size=sample_n, replace=False)
        acc["chunks_p1"].append(np.percentile(chunk[idx], 1, axis=0, keepdims=True))
        acc["chunks_p99"].append(np.percentile(chunk[idx], 99, axis=0, keepdims=True))


def freeze_stats(acc):
    """Produce final stats dict from streaming accumulator."""
    mean = acc["sum"] / acc["n"]
    variance = acc["sum_sq"] / acc["n"] - mean ** 2
    variance = np.clip(variance, 0, None)
    std = np.sqrt(variance)
    p1 = np.concatenate(acc["chunks_p1"], axis=0).mean(axis=0)
    p99 = np.concatenate(acc["chunks_p99"], axis=0).mean(axis=0)
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min": acc["min"].tolist(),
        "max": acc["max"].tolist(),
        "q01": p1.tolist(),
        "q99": p99.tolist(),
    }


def convert_dataset(input_dir: str, output_dir: str, fps: int):
    out = Path(output_dir)
    episodes_meta = scan_episodes(input_dir)

    if not episodes_meta:
        print(f"No episodes found in {input_dir}")
        return

    print(f"Found {len(episodes_meta)} episodes")

    # Build task_index mapping from per-episode meta.json
    task_map = {}
    for _, meta in episodes_meta:
        t = meta.get("task", "unknown task")
        if t not in task_map:
            task_map[t] = len(task_map)
    print(f"Detected {len(task_map)} unique tasks: {task_map}")

    chunks_size = 1000
    episode_meta_list = []

    # Streaming accumulators for stats
    state_acc = streaming_stats()
    action_acc = streaming_stats()
    eef_rel_acc = streaming_stats()
    joint_rel_acc = streaming_stats()
    gripper_abs_acc = streaming_stats()

    total_frames = 0

    # Pre-create directories
    for chunk_idx in range((len(episodes_meta) + chunks_size - 1) // chunks_size):
        (out / "data" / f"chunk-{chunk_idx:03d}").mkdir(parents=True, exist_ok=True)
        video_dir = out / "videos" / f"chunk-{chunk_idx:03d}"
        for cam_key in ["observation.images.hand_view", "observation.images.table_view"]:
            (video_dir / cam_key).mkdir(parents=True, exist_ok=True)

    for ep_idx, (ep_name, meta) in enumerate(episodes_meta):
        npz_path = Path(input_dir) / ep_name / "data.npz"
        print(f"  [{ep_idx+1}/{len(episodes_meta)}] {ep_name}...", end="", flush=True)

        data = dict(np.load(npz_path))
        state = data["state"]  # (T, 16)
        T = state.shape[0]
        if T < 2:
            print(f" SKIP ({T} frames)")
            continue

        action = compute_actions(state)
        task_idx = task_map[meta.get("task", "unknown task")]

        # Streaming stats (T-1 frames)
        push_streaming(state_acc, state[:-1])
        push_streaming(action_acc, action[:-1])
        push_streaming(eef_rel_acc, np.diff(state[:, 0:9], axis=0))
        push_streaming(joint_rel_acc, np.diff(state[:, 9:15], axis=0))
        push_streaming(gripper_abs_acc, state[1:, 15:16])

        # Write parquet
        chunk_idx = ep_idx // chunks_size
        parquet_path = out / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet"

        rows = []
        for t in range(T):
            rows.append({
                "observation.state": state[t].tolist(),
                "action": action[t].tolist(),
                "episode_index": ep_idx,
                "frame_index": t,
                "timestamp": t / fps,
                "task_index": task_idx,
                "annotation.human.task_description": task_idx,
            })

        df = pd.DataFrame(rows)
        pq.write_table(pa.Table.from_pandas(df), str(parquet_path))

        # Write MP4 videos
        video_dir = out / "videos" / f"chunk-{chunk_idx:03d}"
        video_mapping = {
            "observation.images.hand_view": data.get("images_hand"),
            "observation.images.table_view": data.get("images_table"),
        }

        for cam_name, frames in video_mapping.items():
            if frames is not None and len(frames) == T:
                video_path = video_dir / cam_name / f"episode_{ep_idx:06d}.mp4"
                encode_video_mp4(frames, str(video_path), fps)

        episode_meta_list.append({"episode_index": ep_idx, "length": T})
        total_frames += T

        # Free memory
        del data, state, action, df, rows

        print(f" {T} frames (task={task_idx})")

    if not episode_meta_list:
        print("No valid episodes to convert.")
        return

    # ── Write meta files ──
    meta_dir = out / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # info.json
    info = {
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "chunks_size": chunks_size,
        "fps": fps,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [16], "names": None},
            "action": {"dtype": "float32", "shape": [16], "names": None},
            "episode_index": {"dtype": "int64", "shape": []},
            "frame_index": {"dtype": "int64", "shape": []},
            "timestamp": {"dtype": "float32", "shape": []},
            "task_index": {"dtype": "int64", "shape": []},
            "annotation.human.task_description": {"dtype": "int64", "shape": []},
            "observation.images.hand_view": {"dtype": "video", "shape": [240, 320, 3], "names": None},
            "observation.images.table_view": {"dtype": "video", "shape": [240, 320, 3], "names": None},
        },
        "total_episodes": len(episode_meta_list),
        "total_chunks": (len(episode_meta_list) + chunks_size - 1) // chunks_size,
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for em in episode_meta_list:
            f.write(json.dumps(em) + "\n")

    # tasks.jsonl
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for task_desc, task_idx in task_map.items():
            f.write(json.dumps({"task_index": task_idx, "task": task_desc}) + "\n")

    # modality.json
    modality = {
        "state": {
            "eef_9d": {"start": 0, "end": 9, "original_key": "observation.state"},
            "joint_pos": {"start": 9, "end": 15, "original_key": "observation.state"},
            "gripper_pos": {"start": 15, "end": 16, "original_key": "observation.state"},
        },
        "action": {
            "eef_9d": {"start": 0, "end": 9, "original_key": "action"},
            "joint_pos": {"start": 9, "end": 15, "original_key": "action"},
            "gripper_pos": {"start": 15, "end": 16, "original_key": "action"},
        },
        "video": {
            "hand_view": {"original_key": "observation.images.hand_view"},
            "table_view": {"original_key": "observation.images.table_view"},
        },
        "annotation": {
            "human.task_description": {"original_key": "annotation.human.task_description"}
        },
    }
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=2)

    # stats.json
    stats = {
        "observation.state": freeze_stats(state_acc),
        "action": freeze_stats(action_acc),
    }
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # relative_stats.json
    relative_stats = {
        "eef_9d": freeze_stats(eef_rel_acc),
        "joint_pos": freeze_stats(joint_rel_acc),
        "gripper_pos": freeze_stats(gripper_abs_acc),
    }
    with open(meta_dir / "relative_stats.json", "w") as f:
        json.dump(relative_stats, f, indent=2)

    print(f"\nDone! Dataset written to {out}")
    print(f"  Episodes: {len(episode_meta_list)}")
    print(f"  Total frames: {total_frames}")
    print(f"  State dim: {state_acc['sum'].shape[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True,
                        help="Directory of record_demo.py .npz output")
    parser.add_argument("--output-dir", required=True,
                        help="LeRobot v2 dataset output directory")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    convert_dataset(args.input_dir, args.output_dir, args.fps)
