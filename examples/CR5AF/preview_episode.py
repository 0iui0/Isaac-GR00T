#!/usr/bin/env python3
"""Preview recorded episode data (npz + images).

Usage:
    # Single episode
    python preview_episode.py /path/to/episode_000000

    # All episodes in a directory
    python preview_episode.py /path/to/recordings/housing

    # With custom fps
    python preview_episode.py /path/to/recordings/housing --fps 15
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def find_episodes(path: str) -> list[Path]:
    """Find all episode directories under path."""
    p = Path(path)
    # If pointing directly to an episode dir
    if (p / "data.npz").exists():
        return [p]
    # If pointing to a recordings directory, find all episode_* dirs
    episodes = sorted(p.glob("episode_*"))
    episodes = [e for e in episodes if (e / "data.npz").exists()]
    return episodes


def play_episode(ep_dir: Path, fps: int = 30) -> str:
    """Play a single episode. Returns: 'next', 'replay', 'quit'."""
    npz_path = ep_dir / "data.npz"
    meta_path = ep_dir / "meta.json"

    # Load data
    data = np.load(npz_path)
    states = data["state"]
    images_hand = data["images_hand"]
    images_table = data["images_table"]
    gripper_states = data["gripper_states"]

    # Load metadata
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    num_frames = len(states)
    hz = meta.get("hz", 30)
    task = meta.get("task", "N/A")
    grasp = meta.get("grasp_pose", "N/A")

    # Display size
    h = max(images_hand.shape[1], images_table.shape[1])
    w = max(images_hand.shape[2], images_table.shape[2])

    dt = 1.0 / fps
    paused = False
    frame_idx = 0

    while True:
        hand_img = images_hand[frame_idx]
        table_img = images_table[frame_idx]
        gs = gripper_states[frame_idx]

        hand_disp = cv2.resize(hand_img, (w, h))
        table_disp = cv2.resize(table_img, (w, h))

        t_sec = frame_idx / hz
        grip_str = "CLOSE" if gs < 0.5 else "OPEN"
        info = f"frame={frame_idx}/{num_frames} t={t_sec:.1f}s grip={grip_str}"

        cv2.putText(hand_disp, "D405 (hand)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(table_disp, "D455 (table)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        preview = np.hstack([table_disp, hand_disp])

        # Top bar: episode info
        header = f"{ep_dir.name} | {task} | {grasp}"
        cv2.putText(preview, header, (10, h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(preview, info, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if paused:
            cv2.putText(preview, "PAUSED", (w - 100, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Episode Preview", preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            return "quit"
        elif key == ord(" "):
            paused = not paused
        elif key == ord("n"):  # Next episode
            return "next"
        elif key == ord("r"):  # Replay
            return "replay"
        elif key == 81 or key == 2:  # LEFT arrow
            frame_idx = max(0, frame_idx - 30)
            continue
        elif key == 83 or key == 3:  # RIGHT arrow
            frame_idx = min(num_frames - 1, frame_idx + 30)
            continue

        if not paused:
            frame_idx += 1
            if frame_idx >= num_frames:
                return "next"
            time.sleep(dt)


def main():
    parser = argparse.ArgumentParser(description="Preview recorded episodes")
    parser.add_argument("path", help="Episode dir or recordings directory")
    parser.add_argument("--fps", type=int, default=30, help="Playback speed")
    args = parser.parse_args()

    episodes = find_episodes(args.path)
    if not episodes:
        print(f"No episodes found at {args.path}")
        return

    print(f"Found {len(episodes)} episode(s)")
    for i, ep in enumerate(episodes):
        meta_path = ep / "meta.json"
        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        nf = meta.get("num_frames", "?")
        hz = meta.get("hz", 30)
        dur = nf / hz if isinstance(nf, int) else "?"
        print(f"  [{i}] {ep.name}: {nf} frames ({dur:.1f}s)")

    print()
    print("Controls: SPACE=pause, N=next episode, R=replay, Q=quit, LEFT/RIGHT=skip")
    print()

    idx = 0
    while 0 <= idx < len(episodes):
        ep = episodes[idx]
        meta_path = ep / "meta.json"
        nf = "?"
        task = "?"
        if meta_path.exists():
            with open(meta_path) as f:
                m = json.load(f)
            nf = m.get("num_frames", "?")
            task = m.get("task", "?")

        print(f"[{idx + 1}/{len(episodes)}] {ep.name} | {task} | {nf} frames")

        result = play_episode(ep, args.fps)

        if result == "quit":
            break
        elif result == "replay":
            continue  # same idx
        else:  # next
            idx += 1

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
