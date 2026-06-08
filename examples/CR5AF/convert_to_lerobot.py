#!/usr/bin/env python3
"""Convert CR5AF .npz recordings to LeRobot v2 format for GR00T finetuning.

Task descriptions are read from each episode's meta.json (written by record_demo.py).
Multiple tasks are automatically detected and assigned unique task_index values.

Usage:
    # Record episodes separately per task:
    python record_demo.py --task "抓取电机外壳放到固定工装上" --grasp-pose grasp_housing ...
    python record_demo.py --task "抓取轴套放到固定工装上" --grasp-pose grasp_shaft ...

    # Convert all at once — tasks are auto-detected:
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
    """Compute single-step actions from consecutive states.

    states: (T, 16)  [eef_9d(9), joint_pos(6), gripper_pos(1)]

    action[t] = concat[
        eef[t+1] - eef[t],      # RELATIVE EEF delta (9D)
        joints[t+1] - joints[t], # RELATIVE joint delta (6D)
        gripper[t+1],            # ABSOLUTE gripper target (1D)
    ]
    """
    T = states.shape[0]
    actions = np.zeros((T, 16), dtype=np.float32)
    for t in range(T - 1):
        actions[t, 0:9] = states[t + 1, 0:9] - states[t, 0:9]
        actions[t, 9:15] = states[t + 1, 9:15] - states[t, 9:15]
        actions[t, 15:16] = states[t + 1, 15:16]
    return actions


def collect_episodes(input_dir: str):
    """Discover all episode directories from record_demo.py output."""
    episodes = []
    root = Path(input_dir)
    for ep_dir in sorted(root.glob("episode_*")):
        npz_path = ep_dir / "data.npz"
        meta_path = ep_dir / "meta.json"
        if npz_path.exists():
            data = dict(np.load(npz_path))
            meta = {}
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
            episodes.append((ep_dir.name, data, meta))
    return episodes


def convert_dataset(input_dir: str, output_dir: str, fps: int):
    out = Path(output_dir)

    episodes = collect_episodes(input_dir)

    if not episodes:
        print(f"No episodes found in {input_dir}")
        return

    print(f"Found {len(episodes)} episodes")

    # Build task_index mapping from per-episode meta.json
    task_map = {}
    for _, _, meta in episodes:
        t = meta.get("task", "unknown task")
        if t not in task_map:
            task_map[t] = len(task_map)
    print(f"Detected {len(task_map)} unique tasks: {task_map}")

    chunks_size = 1000
    episode_meta = []
    all_states_for_stats = []
    all_actions_for_stats = []

    # Accumulators for relative_stats
    eef_rel_chunks = []
    joint_rel_chunks = []
    gripper_abs_chunks = []

    for ep_idx, (ep_name, data, meta) in enumerate(episodes):
        state = data["state"]  # (T, 16)
        T = state.shape[0]
        if T < 2:
            print(f"  Skipping {ep_name}: too short ({T} frames)")
            continue

        action = compute_actions(state)

        # Collect for full stats
        all_states_for_stats.append(state[:-1])
        all_actions_for_stats.append(action[:-1])

        # Collect for per-key relative stats
        eef_rel_chunks.append(np.diff(state[:, 0:9], axis=0))
        joint_rel_chunks.append(np.diff(state[:, 9:15], axis=0))
        gripper_abs_chunks.append(state[1:, 15:16])

        # Write per-episode parquet
        chunk_idx = ep_idx // chunks_size
        parquet_dir = out / "data" / f"chunk-{chunk_idx:03d}"
        parquet_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = parquet_dir / f"episode_{ep_idx:06d}.parquet"

        rows = []
        task_idx = task_map[meta.get("task", "unknown task")]
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
            "hand_view": data.get("images_hand"),
            "table_view": data.get("images_table"),
        }

        for cam_name, frames in video_mapping.items():
            if frames is not None and len(frames) == T:
                original_key = f"observation.images.{cam_name}"
                cam_subdir = video_dir / original_key
                cam_subdir.mkdir(parents=True, exist_ok=True)
                video_path = cam_subdir / f"episode_{ep_idx:06d}.mp4"
                encode_video_mp4(frames, str(video_path), fps)

        episode_meta.append({"episode_index": ep_idx, "length": T})
        print(f"  {ep_name}: {T} frames")

    if not all_states_for_stats:
        print("No valid episodes to convert.")
        return

    # Collect all state for full stats (not just T-1)
    all_states_full = np.concatenate(
        [data["state"] for _, data, _ in episodes if data["state"].shape[0] >= 2], axis=0
    )
    all_actions_full = np.concatenate(all_actions_for_stats, axis=0)
    eef_rel_all = np.concatenate(eef_rel_chunks, axis=0)
    joint_rel_all = np.concatenate(joint_rel_chunks, axis=0)
    gripper_abs_all = np.concatenate(gripper_abs_chunks, axis=0)

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
        "total_episodes": len(episode_meta),
        "total_chunks": (len(episode_meta) + chunks_size - 1) // chunks_size,
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for em in episode_meta:
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
            "human": {
                "task_description": {"original_key": "annotation.human.task_description"}
            }
        },
    }
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=2)

    # stats.json
    def stats_dict(arr):
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "q01": np.percentile(arr, 1, axis=0).tolist(),
            "q99": np.percentile(arr, 99, axis=0).tolist(),
        }

    stats = {
        "observation.state": stats_dict(all_states_full),
        "action": stats_dict(all_actions_full),
    }
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # relative_stats.json
    relative_stats = {
        "eef_9d": stats_dict(eef_rel_all),
        "joint_pos": stats_dict(joint_rel_all),
        "gripper_pos": stats_dict(gripper_abs_all),
    }
    with open(meta_dir / "relative_stats.json", "w") as f:
        json.dump(relative_stats, f, indent=2)

    print(f"\nDone! Dataset written to {out}")
    print(f"  Episodes: {len(episode_meta)}")
    print(f"  Total frames: {len(all_states_full)}")
    print(f"  State dim: {all_states_full.shape[1]}")
    print(f"  Action dim: {all_actions_full.shape[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True,
                        help="Directory of record_demo.py .npz output")
    parser.add_argument("--output-dir", required=True,
                        help="LeRobot v2 dataset output directory")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    convert_dataset(args.input_dir, args.output_dir, args.fps)
