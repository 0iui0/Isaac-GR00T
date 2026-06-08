#!/usr/bin/env python3
"""CR5AF + TopHand closed-loop deployment with GR00T N1.7 fine-tuned model.

Runs inference on the training machine (2xRTX 5090) or directly on Thor,
streaming actions to CR5AF via ServoP and controlling TopHand for grasping.

Usage (on training machine serving Thor):
    python examples/CR5AF/deploy_cr5af.py \
        --model-path /tmp/cr5af_finetune/checkpoint-10000 \
        --robot-ip 192.168.5.1 \
        --task "pick motor housing and shaft sleeve, place on fixture" \
        --grasp-pose grasp_housing \
        --hz 8

Usage (server mode on training machine, client on Thor):
    # Training machine:
    python examples/CR5AF/deploy_cr5af.py --server --model-path ... --port 5555

    # Thor:
    python examples/CR5AF/deploy_cr5af.py --client --server-ip 192.168.16.XXX --port 5555 \
        --robot-ip 192.168.5.1 --grasp-pose grasp_housing
"""

import argparse
import os
import socket
import struct
import subprocess
import sys
import threading
import time
import json
import glob
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


# ─── Camera ──────────────────────────────────────────────────────────────────

class RealSenseCamera:
    def __init__(self, serial_or_index, width=320, height=240, fps=30):
        import pyrealsense2 as rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        if isinstance(serial_or_index, str) and not serial_or_index.isdigit():
            config.enable_device(serial_or_index)
        else:
            config.enable_device(str(serial_or_index))
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(config)
        for _ in range(30):
            self.pipeline.wait_for_frames()

    def read(self):
        frames = self.pipeline.wait_for_frames()
        color = frames.get_color_frame()
        return np.asanyarray(color.get_data()) if color else None

    def close(self):
        self.pipeline.stop()


# ─── CR5AF Connection ────────────────────────────────────────────────────────

RT_TOOL_VECTOR = 624
RT_FRAME_MAGIC = 0x123456789ABCDEF


class CR5AFConnection:
    def __init__(self, robot_ip, cmd_port=29999, rt_port=30004):
        self.robot_ip = robot_ip
        self.cmd_port = cmd_port
        self.rt_port = rt_port
        self._rt_sock = None
        self._cmd_sock = None
        self._cmd_lock = threading.Lock()
        self._running = True
        self._lock = threading.Lock()
        self._joint_pos = None
        self._tool_vector = None
        self._quaternion = None
        self._digital_outputs = 0

    def connect(self):
        self._connect_rt()
        self._connect_cmd()

    def _connect_rt(self):
        self._rt_sock = self._new_sock((self.robot_ip, self.rt_port))
        t = threading.Thread(target=self._rt_loop, daemon=True)
        t.start()

    def _new_sock(self, addr):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)
        s.connect(addr)
        return s

    def _rt_loop(self):
        while self._running:
            try:
                data = self._rt_sock.recv(1440)
                if len(data) >= 648:
                    magic = struct.unpack_from("<Q", data, 48)[0]
                    if magic == RT_FRAME_MAGIC:
                        with self._lock:
                            self._joint_pos = list(struct.unpack_from("<6d", data, 192))
                            self._tool_vector = list(struct.unpack_from("<6d", data, RT_TOOL_VECTOR))
                            self._quaternion = list(struct.unpack_from("<4d", data, 700))
                            self._digital_outputs = struct.unpack_from("<Q", data, 24)[0]
            except Exception:
                if self._running:
                    try:
                        self._rt_sock.close()
                    except Exception:
                        pass
                    try:
                        self._rt_sock = self._new_sock((self.robot_ip, self.rt_port))
                    except Exception:
                        time.sleep(0.5)

    def _connect_cmd(self):
        self._cmd_sock = self._new_sock((self.robot_ip, self.cmd_port))
        self._cmd_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._cmd_sock.setblocking(False)
        try:
            self._cmd_sock.recv(4096)
        except BlockingIOError:
            pass
        self._cmd_sock.setblocking(True)

    def dashboard_cmd(self, cmd, timeout=3.0):
        with self._cmd_lock:
            self._cmd_sock.settimeout(timeout)
            self._cmd_sock.sendall(cmd.encode())
            resp = bytearray()
            while True:
                c = self._cmd_sock.recv(1)
                if not c or c == b";":
                    break
                resp.extend(c)
            return resp.decode().strip()

    def servop(self, x, y, z, rx, ry, rz):
        cmd = f"ServoP({x:.3f},{y:.3f},{z:.3f},{rx:.3f},{ry:.3f},{rz:.3f})"
        with self._cmd_lock:
            try:
                self._cmd_sock.setblocking(False)
                try:
                    while True:
                        self._cmd_sock.recv(4096)
                except BlockingIOError:
                    pass
                self._cmd_sock.setblocking(True)
                self._cmd_sock.sendall(cmd.encode())
            except Exception as e:
                print(f"\n[ServoP ERROR] {e}", flush=True)
                self._reconnect_cmd()

    def get_state(self):
        with self._lock:
            if self._joint_pos is None:
                return None
            return (
                list(self._joint_pos),
                list(self._tool_vector),
                list(self._quaternion),
                int(self._digital_outputs),
            )

    def _reconnect_cmd(self):
        try:
            self._cmd_sock.close()
        except Exception:
            pass
        try:
            self._connect_cmd()
        except Exception:
            pass

    def close(self):
        self._running = False
        for s in (self._rt_sock, self._cmd_sock):
            if s:
                try:
                    s.close()
                except Exception:
                    pass


# ─── TopHand Wrapper ─────────────────────────────────────────────────────────

class TopHandCLI:
    def __init__(self, tophand_bin="tophand", hand="right"):
        cmd = [tophand_bin, "-r" if hand == "right" else "-l"]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        self._read_until("[OK] TopHand initialized", timeout=15)

    def _read_until(self, expected, timeout=10):
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                break
            if expected in line:
                return line
        return ""

    def send_cmd(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def grasp(self, name, step="pre_grasp"):
        self.send_cmd(f"grasp {name} {step}")

    def home(self):
        self.send_cmd("home")

    def close(self):
        self.send_cmd("quit")
        self.proc.terminate()
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()


# ─── State Conversion ────────────────────────────────────────────────────────

def quat_to_rot6d(q):
    r = R.from_quat(q)
    mat = r.as_matrix()
    rot6d = np.concatenate([mat[:, 0], mat[:, 1]])
    return rot6d.astype(np.float32)


def build_state_dict(joint_pos, tool_vector, quaternion, gripper_pos):
    """Build GR00T state dict from CR5AF RT data."""
    eef_xyz = np.array(tool_vector[:3], dtype=np.float32)
    eef_rot6d = quat_to_rot6d(quaternion)
    eef_9d = np.concatenate([eef_xyz, eef_rot6d])
    return {
        "eef_9d": eef_9d.reshape(1, 1, 9),
        "joint_pos": np.array(joint_pos, dtype=np.float32).reshape(1, 1, 6),
        "gripper_pos": np.array([gripper_pos], dtype=np.float32).reshape(1, 1, 1),
    }


# ─── Inference Server (runs on training machine) ─────────────────────────────

def run_server(model_path: str, port: int):
    """ZMQ REP server that accepts observations and returns actions."""
    import zmq
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    print(f"Loading model from {model_path}...")
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        model_path=model_path,
        device="cuda:0",
    )
    print("Model loaded.")

    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")
    print(f"Inference server listening on port {port}")

    while True:
        try:
            import msgpack
            import msgpack_numpy
            msgpack_numpy.patch()
            raw = socket.recv()
            request = msgpack.unpackb(raw)
            obs = request["observation"]
            action = policy.get_action(obs)
            socket.send(msgpack.packb(action))
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            socket.send(msgpack.packb({"error": str(e)}))


# ─── Safety ───────────────────────────────────────────────────────────────────

class SafetyChecker:
    """Multi-layer safety limits to guard against erratic policy outputs.

    Layers (applied in order):
    1. NaN/inf detection → emergency stop
    2. Per-step delta clipping → translation + rotation independently
    3. Workspace bounding box → hard xyz limits
    4. Consecutive violation counter → auto-stop if too many in a row
    """

    def __init__(
        self,
        max_translation_delta: float = 30.0,   # mm per step (8Hz → 240mm/s max)
        max_rotation_delta: float = 10.0,       # deg per step
        workspace_min_xyz: tuple = (-200.0, -400.0, 0.0),   # mm
        workspace_max_xyz: tuple = (400.0, 200.0, 600.0),   # mm
        max_consecutive_violations: int = 3,
    ):
        self.max_translation_delta = max_translation_delta
        self.max_rotation_delta = max_rotation_delta
        self.workspace_min = np.array(workspace_min_xyz, dtype=np.float32)
        self.workspace_max = np.array(workspace_max_xyz, dtype=np.float32)
        self.max_consecutive_violations = max_consecutive_violations
        self.violation_count = 0
        self.last_target = None

    def check_numerics(self, action: dict) -> bool:
        """Return True if any NaN or inf detected in action arrays."""
        for key, arr in action.items():
            if arr is None:
                return True
            a = np.asarray(arr)
            if np.any(np.isnan(a)) or np.any(np.isinf(a)):
                return True
        return False

    def clip_delta(self, target_xyz_rxyz: list, current_pose: list) -> list:
        """Clip per-step translation and rotation deltas independently."""
        target = np.array(target_xyz_rxyz, dtype=np.float32)
        current = np.array(current_pose, dtype=np.float32)

        # Translation delta
        trans_delta = target[:3] - current[:3]
        trans_norm = np.linalg.norm(trans_delta)
        if trans_norm > self.max_translation_delta:
            trans_delta = trans_delta / trans_norm * self.max_translation_delta

        # Rotation delta (per-axis)
        rot_delta = target[3:6] - current[3:6]
        rot_delta = np.clip(rot_delta, -self.max_rotation_delta, self.max_rotation_delta)

        clipped = np.concatenate([current[:3] + trans_delta, current[3:6] + rot_delta])
        return clipped.tolist()

    def enforce_workspace(self, target_xyz_rxyz: list) -> list:
        """Hard-clamp xyz to the workspace bounding box."""
        result = list(target_xyz_rxyz)
        result[0] = max(self.workspace_min[0], min(self.workspace_max[0], result[0]))
        result[1] = max(self.workspace_min[1], min(self.workspace_max[1], result[1]))
        result[2] = max(self.workspace_min[2], min(self.workspace_max[2], result[2]))
        return result

    def check_and_clip(
        self, target_xyz_rxyz: list, current_pose: list
    ) -> tuple[list, bool]:
        """Apply all safety layers. Returns (safe_target, emergency_stop).
        - emergency_stop=True: target is safe, proceed
        - emergency_stop=False: too many consecutive violations, stop now
        """
        # Layer 2: delta clipping
        target = self.clip_delta(target_xyz_rxyz, current_pose)

        # Layer 3: workspace hard limit
        target = self.enforce_workspace(target)

        # Track violations
        was_clipped = not np.allclose(target, target_xyz_rxyz, rtol=1e-4)
        if was_clipped:
            self.violation_count += 1
            if self.violation_count >= self.max_consecutive_violations:
                return target, False
        else:
            self.violation_count = 0

        return target, True


# ─── Direct Inference (no client/server) ─────────────────────────────────────

class DirectPolicy:
    """Load model locally for direct inference (training machine or Jetson)."""

    def __init__(self, model_path: str, device: str = "cuda:0"):
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        print(f"Loading model from {model_path}...")
        self.policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            model_path=model_path,
            device=device,
        )
        print("Model loaded.")

    def get_action(self, observation):
        return self.policy.get_action(observation)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true",
                        help="Run as inference server on training machine")
    parser.add_argument("--client", action="store_true",
                        help="Run as client sending observations to server")
    parser.add_argument("--server-ip", default="192.168.16.158")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--model-path", required=True,
                        help="Path to fine-tuned model checkpoint")
    parser.add_argument("--robot-ip", default="192.168.5.1")
    parser.add_argument("--task", required=True,
                        help="Task description for the policy")
    parser.add_argument("--grasp-pose", required=True,
                        choices=["grasp_shaft", "grasp_housing"])
    parser.add_argument("--tophand-bin",
                        default="/home/tophand/workspaces/tophand_sdk/build/tophand")
    parser.add_argument("--hz", type=float, default=8.0,
                        help="Control rate (Hz)")
    parser.add_argument("--action-exec-horizon", type=int, default=5,
                        help="Number of action steps to execute before re-inferring")
    parser.add_argument("--speed", type=float, default=50.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-translation-delta", type=float, default=30.0,
                        help="Max EEF translation per step in mm (8Hz: 30mm→240mm/s)")
    parser.add_argument("--max-rotation-delta", type=float, default=10.0,
                        help="Max EEF rotation per step in degrees")
    args = parser.parse_args()

    if args.server:
        run_server(args.model_path, args.port)
        return

    if args.client:
        print("Client mode: connect to inference server on training machine")
        print("(Not yet implemented - use direct mode or run server on same machine)")
        return

    dt = 1.0 / args.hz

    print("=" * 60)
    print("CR5AF + TopHand GR00T Deployment")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Robot: {args.robot_ip}  Task: {args.task}")
    print(f"Grasp: {args.grasp_pose}  Rate: {args.hz} Hz")
    print("=" * 60)

    # ── Load model ──
    print("\n[1/5] Loading GR00T model...")
    policy = DirectPolicy(args.model_path, device=args.device)

    # ── Connect CR5AF ──
    print("\n[2/5] Connecting to CR5AF...")
    conn = CR5AFConnection(args.robot_ip)
    conn.connect()
    print("  EnableRobot:", conn.dashboard_cmd("EnableRobot()"))
    print("  SpeedFactor:", conn.dashboard_cmd(f"SpeedFactor({int(args.speed)})"))
    conn.dashboard_cmd("ResetRobot()")
    time.sleep(0.3)

    for _ in range(20):
        state = conn.get_state()
        if state:
            break
        time.sleep(0.1)
    if not state:
        print("ERROR: No RT data.")
        sys.exit(1)
    jp, tv, quat, do = state
    print(f"  Joints: {[f'{x:.1f}' for x in jp]}")

    # ── Start TopHand ──
    print(f"\n[3/5] Starting TopHand...")
    try:
        hand = TopHandCLI(tophand_bin=args.tophand_bin)
        hand.home()
        time.sleep(2)
        hand.grasp(args.grasp_pose, "pre_grasp")
        time.sleep(2)
        print("  TopHand ready.")
    except Exception as e:
        print(f"  ERROR: {e}")
        hand = None

    # ── Start Cameras ──
    print("\n[4/5] Starting cameras...")
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        cam_serial_hand = None
        cam_serial_table = None
        for dev in ctx.devices:
            name = dev.get_info(rs.camera_info.name)
            sn = dev.get_info(rs.camera_info.serial_number)
            if "455" in name and cam_serial_table is None:
                cam_serial_table = sn
            elif "405" in name and cam_serial_hand is None:
                cam_serial_hand = sn
        if cam_serial_table and cam_serial_hand:
            table_cam = RealSenseCamera(cam_serial_table)
            hand_cam = RealSenseCamera(cam_serial_hand)
            print(f"  D455 (table): serial={cam_serial_table}")
            print(f"  D405 (hand):  serial={cam_serial_hand}")
        else:
            table_cam = RealSenseCamera("0")
            hand_cam = RealSenseCamera("1")
            print("  Using camera indices 0 and 1")
    except ImportError:
        print("  pyrealsense2 not available, using OpenCV fallback...")
        table_cam = cv2.VideoCapture(0)
        hand_cam = cv2.VideoCapture(1)
        table_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        table_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        hand_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        hand_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # ── Inference Loop ──
    print("\n[5/5] Starting inference loop...")
    print("=" * 60)
    print("Controls:")
    print("  Ctrl+C = stop")
    print("=" * 60)

    # Video history buffer: 2 frames for delta_indices=[-20, 0]
    # At 8Hz, -20 steps = -2.5s, 0 = current
    video_history_hand = deque(maxlen=21)  # enough for history window
    video_history_table = deque(maxlen=21)
    gripper_state = 1.0  # start open
    step_count = 0
    action_buffer = None
    action_step = 0
    safety = SafetyChecker(
        max_translation_delta=args.max_translation_delta,
        max_rotation_delta=args.max_rotation_delta,
    )
    print(f"  Safety: max_delta={safety.max_translation_delta}mm/{safety.max_rotation_delta}deg  "
          f"workspace=({safety.workspace_min.tolist()},{safety.workspace_max.tolist()})")

    def read_camera(cam):
        if isinstance(cam, RealSenseCamera):
            return cam.read()
        else:
            ret, frame = cam.read()
            return frame if ret else None

    try:
        while True:
            loop_start = time.monotonic()

            # Read state
            state = conn.get_state()
            if not state:
                time.sleep(dt)
                continue
            jp, tv, quat, do = state

            # Read cameras
            table_frame = read_camera(table_cam)
            hand_frame = read_camera(hand_cam)

            if table_frame is not None:
                video_history_table.append(cv2.resize(table_frame, (320, 240)))
            if hand_frame is not None:
                video_history_hand.append(cv2.resize(hand_frame, (320, 240)))

            # Need enough history
            if len(video_history_hand) < 2 or len(video_history_table) < 2:
                time.sleep(dt)
                continue

            # Build observation
            # Video: 2 frames at delta [-20, 0] → last frame + frame from ~20 steps ago
            # For simplicity at 8Hz, we use the two most recent frames
            hand_frames = np.stack([
                video_history_hand[-2],  # older frame (~125ms ago at 8Hz)
                video_history_hand[-1],  # current frame
            ], axis=0)  # (T=2, H, W, C)
            table_frames = np.stack([
                video_history_table[-2],
                video_history_table[-1],
            ], axis=0)

            state_dict = build_state_dict(jp, tv, quat, gripper_state)

            observation = {
                "video": {
                    "hand_view": hand_frames[None, ...].astype(np.uint8),   # (1, 2, 240, 320, 3)
                    "table_view": table_frames[None, ...].astype(np.uint8),
                },
                "state": state_dict,
                "language": {
                    "annotation.human.task_description": [[args.task]],
                },
            }

            # Run inference every action_exec_horizon steps
            if step_count % args.action_exec_horizon == 0:
                try:
                    action = policy.get_action(observation)
                except Exception as e:
                    print(f"\n[Inference ERROR] {e}")
                    time.sleep(dt)
                    continue

                # Layer 1: NaN/inf check
                if safety.check_numerics(action):
                    print("\n[SAFETY STOP] NaN/inf detected in model output!")
                    break

                action_buffer = {
                    k: v[0] for k, v in action.items()
                }  # (40, D) each
                action_step = 0

            # Execute current action step
            if action_buffer is not None:
                idx = action_step
                if idx < args.action_exec_horizon:
                    eef_delta = action_buffer["eef_9d"][idx]       # (9,) relative delta
                    joint_delta = action_buffer["joint_pos"][idx]   # (6,) relative delta
                    gripper_target = action_buffer["gripper_pos"][idx]  # (1,) absolute

                    # Apply EEF delta in tool frame
                    pose = tv  # current TCP pose [x, y, z, rx, ry, rz]
                    raw_target = [
                        pose[0] + float(eef_delta[0]),
                        pose[1] + float(eef_delta[1]),
                        pose[2] + float(eef_delta[2]),
                        pose[3] + float(eef_delta[3]),
                        pose[4] + float(eef_delta[4]),
                        pose[5] + float(eef_delta[5]),
                    ]

                    # Safety: delta clip + workspace hard limit
                    delta_mag = np.linalg.norm(
                        np.array(raw_target[:3]) - np.array(pose[:3])
                    )
                    if delta_mag > 0.5:  # above dead zone — execute
                        safe_target, ok = safety.check_and_clip(raw_target, pose)
                        if not ok:
                            print(f"\n[SAFETY STOP] {safety.violation_count} consecutive "
                                  f"violations — policy may be erratic!")
                            break
                        if safety.violation_count > 0:
                            print(f"\n[SAFETY CLIP] violation #{safety.violation_count} "
                                  f"pos=({raw_target[0]:.0f},{raw_target[1]:.0f},{raw_target[2]:.0f})→"
                                  f"({safe_target[0]:.0f},{safe_target[1]:.0f},{safe_target[2]:.0f})")
                        conn.servop(*safe_target)

                    # Gripper control
                    if gripper_target[0] < 0.5 and gripper_state > 0.5:
                        if hand:
                            hand.grasp(args.grasp_pose, "grasp")
                        gripper_state = 0.0
                        print("\n[GRIPPER] CLOSE")
                    elif gripper_target[0] > 0.5 and gripper_state < 0.5:
                        if hand:
                            hand.grasp(args.grasp_pose, "pre_grasp")
                        gripper_state = 1.0
                        print("\n[GRIPPER] OPEN")

                    action_step += 1

            step_count += 1
            print(f"\r[step {step_count}] pos=({tv[0]:.1f},{tv[1]:.1f},{tv[2]:.1f}) "
                  f"gripper={gripper_state:.0f}    ", end="", flush=True)

            elapsed = time.monotonic() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\n\nStopping...")

    # Cleanup
    conn.dashboard_cmd("ResetRobot()")
    conn.close()
    if hand:
        hand.close()
    if isinstance(table_cam, RealSenseCamera):
        table_cam.close()
    if isinstance(hand_cam, RealSenseCamera):
        hand_cam.close()
    print("Done.")


if __name__ == "__main__":
    main()
