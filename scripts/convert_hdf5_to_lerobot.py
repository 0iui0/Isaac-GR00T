"""Convert IsaacLab HDF5 dataset to LeRobot v2 format for GR00T finetuning.

Usage:
    python scripts/convert_hdf5_to_lerobot.py \
        --hdf5_path /path/to/pick_orange.hdf5 \
        --output_dir /path/to/output \
        --task_description "pick the orange" \
        --fps 30
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def get_successful_episodes(hdf5_path: str) -> list[tuple[str, h5py.Group]]:
    """Get list of (episode_name, episode_group) for successful episodes."""
    f = h5py.File(hdf5_path, "r")
    data = f["data"]
    episodes = []
    for key in sorted(data.keys(), key=lambda x: int(x.split("_")[1])):
        ep = data[key]
        if ep.attrs.get("success", False):
            episodes.append((key, ep))
    f.close()
    return episodes


def encode_video_mp4(frames: np.ndarray, output_path: str, fps: int = 30):
    """Encode a sequence of frames (T, H, W, 3) uint8 to MP4."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    T, H, W, C = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    for i in range(T):
        writer.write(cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
    writer.release()


def convert_dataset(
    hdf5_path: str,
    output_dir: str,
    task_description: str,
    fps: int,
    skip_first_n: int = 5,
):
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading HDF5: {hdf5_path}")
    f = h5py.File(hdf5_path, "r")

    # Collect successful episodes
    data = f["data"]
    ep_keys = sorted(
        [k for k in data.keys() if data[k].attrs.get("success", False)],
        key=lambda x: int(x.split("_")[1]),
    )
    print(f"Found {len(ep_keys)} successful episodes: {ep_keys}")

    all_rows = []
    episode_meta = []

    for ep_idx, ep_key in enumerate(tqdm(ep_keys, desc="Processing episodes")):
        ep = data[ep_key]
        num_frames = int(ep.attrs.get("num_samples", ep["obs/joint_pos"].shape[0]))

        joint_pos = ep["obs/joint_pos"][:]  # (T, 6)
        joint_pos_target = ep["obs/joint_pos_target"][:]  # (T, 6)

        # Encode videos for this episode
        ep_len = num_frames - skip_first_n

        for t in range(skip_first_n, num_frames):
            row = {
                "observation.state": joint_pos[t].tolist(),
                "action": joint_pos_target[t].tolist(),
                "episode_index": ep_idx,
                "frame_index": t - skip_first_n,
                "timestamp": (t - skip_first_n) / fps,
                "task_index": 0,
            }
            all_rows.append(row)

        episode_meta.append({"episode_index": int(ep_idx), "length": int(ep_len)})

        # Write videos
        for cam_name in ["front", "wrist"]:
            video_dir = output_dir / "videos/train-00000-of-00001"
            os.makedirs(video_dir, exist_ok=True)
            video_path = video_dir / f"{ep_idx:06d}_{cam_name}.mp4"
            if not video_path.exists():
                frames = ep[f"obs/{cam_name}"][skip_first_n:num_frames]
                encode_video_mp4(frames, str(video_path), fps)
                print(f"  Wrote {cam_name} video: {video_path} ({frames.shape[0]} frames)")
            else:
                print(f"  Skip existing {cam_name} video: {video_path}")

    f.close()

    # Write parquet
    print("Writing parquet data...")
    df = pd.DataFrame(all_rows)
    data_dir = output_dir / "data"
    os.makedirs(data_dir, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(df),
        data_dir / "train-00000-of-00001.parquet",
    )

    # Write meta/info.json
    info = {
        "data_path": "data/train-{episode_chunk:05d}-of-00001.parquet",
        "video_path": "videos/train-{episode_chunk:05d}-of-00001/{episode_index:06d}_{video_key}.mp4",
        "chunks_size": len(ep_keys),
        "fps": fps,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [6], "names": None},
            "action": {"dtype": "float32", "shape": [6], "names": None},
            "episode_index": {"dtype": "int64", "shape": []},
            "frame_index": {"dtype": "int64", "shape": []},
            "timestamp": {"dtype": "float32", "shape": []},
            "task_index": {"dtype": "int64", "shape": []},
        },
    }

    # Write meta/tasks.jsonl
    meta_dir = output_dir / "meta"
    os.makedirs(meta_dir, exist_ok=True)

    with open(meta_dir / "info.json", "w") as fh:
        json.dump(info, fh, indent=2)

    with open(meta_dir / "tasks.jsonl", "w") as fh:
        fh.write(json.dumps({"task_index": 0, "task": task_description}) + "\n")

    with open(meta_dir / "episodes.jsonl", "w") as fh:
        for em in episode_meta:
            fh.write(json.dumps(em) + "\n")

    # Write modality.json (same as SO100 config)
    modality = {
        "state": {
            "single_arm": {"start": 0, "end": 5},
            "gripper": {"start": 5, "end": 6},
        },
        "action": {
            "single_arm": {"start": 0, "end": 5},
            "gripper": {"start": 5, "end": 6},
        },
        "video": {
            "front": {"original_key": "observation.images.front"},
            "wrist": {"original_key": "observation.images.wrist"},
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
        },
    }
    with open(meta_dir / "modality.json", "w") as fh:
        json.dump(modality, fh, indent=2)

    # Generate stats.json
    print("Computing statistics...")
    states = np.array([r["observation.state"] for r in all_rows], dtype=np.float32)
    actions = np.array([r["action"] for r in all_rows], dtype=np.float32)

    stats = {
        "observation.state": {
            "mean": states.mean(axis=0).tolist(),
            "std": states.std(axis=0).tolist(),
            "min": states.min(axis=0).tolist(),
            "max": states.max(axis=0).tolist(),
        },
        "action": {
            "mean": actions.mean(axis=0).tolist(),
            "std": actions.std(axis=0).tolist(),
            "min": actions.min(axis=0).tolist(),
            "max": actions.max(axis=0).tolist(),
        },
    }
    with open(meta_dir / "stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)

    print(f"\nDone! Dataset written to: {output_dir}")
    print(f"Total episodes: {len(ep_keys)}")
    print(f"Total frames: {len(all_rows)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task_description", default="pick the orange")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--skip_first_n", type=int, default=5)
    args = parser.parse_args()
    convert_dataset(
        args.hdf5_path, args.output_dir, args.task_description, args.fps, args.skip_first_n
    )
