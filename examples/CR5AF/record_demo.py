#!/usr/bin/env python3
"""CR5AF + TopHand data collection for GR00T N1.7 fine-tuning.

Records episodes with:
- CR5AF RT state (eef_9d + joint_pos + gripper_pos)
- D405 hand camera + D455 table camera (RealSense)
- SpaceMouse teleop via ServoP
- TopHand grasp control via CLI subprocess

Usage (on thor):
    cd ~/workspaces/hil-serl
    .venv/bin/python record_demo.py \
      --robot-ip 192.168.5.1 \
      --task "pick motor housing and place on fixture" \
      --output-dir recordings/cr5af_demos
"""

import argparse
import os
import select
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
    """Intel RealSense D400 series camera with background frame capture."""

    def __init__(self, serial_or_index, width=640, height=480, fps=15,
                 exposure=0, gain=0):
        import pyrealsense2 as rs

        self.pipeline = rs.pipeline()
        config = rs.config()
        if isinstance(serial_or_index, str) and not serial_or_index.isdigit():
            config.enable_device(serial_or_index)
        else:
            config.enable_device(str(serial_or_index))
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        profile = self.pipeline.start(config)

        # Set camera options
        for sensor in profile.get_device().query_sensors():
            try:
                if exposure > 0:
                    sensor.set_option(rs.option.enable_auto_exposure, 0)
                    sensor.set_option(rs.option.exposure, float(exposure))
                if gain > 0:
                    sensor.set_option(rs.option.gain, float(gain))
            except Exception:
                pass

        # Skip auto-exposure settling frames
        for _ in range(30):
            self.pipeline.wait_for_frames()

        # Background thread for non-blocking reads
        self._latest_frame = None
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        while self._running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=2000)
                color = frames.get_color_frame()
                if color:
                    self._latest_frame = np.asanyarray(color.get_data()).copy()
            except Exception:
                pass

    def read(self):
        return self._latest_frame

    def close(self):
        self._running = False
        if hasattr(self, "_thread") and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.pipeline.stop()


# ─── CR5AF Connection (reused from hil-serl teleop_cr5af.py) ─────────────────

RT_TOOL_VECTOR = 624
RT_FRAME_MAGIC = 0x123456789ABCDEF


class CR5AFConnection:
    """Persistent TCP connections to CR5AF."""

    def __init__(self, robot_ip, cmd_port=29999, rt_port=30004):
        self.robot_ip = robot_ip
        self.cmd_port = cmd_port
        self.rt_port = rt_port
        self._rt_sock = None
        self._cmd_sock = None
        self._cmd_lock = threading.Lock()
        self._running = True

        # Cached RT data
        self._lock = threading.Lock()
        self._joint_pos = None       # QActual[6]
        self._tool_vector = None     # xyz + rxyz
        self._quaternion = None      # ActualQuaternion[4]
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
        import socket as _socket

        while self._running:
            try:
                data = self._rt_sock.recv(1440)
                if len(data) >= 648:
                    magic = struct.unpack_from("<Q", data, 48)[0]
                    if magic == RT_FRAME_MAGIC:
                        with self._lock:
                            # Joint positions: QActual[6] at offset 192 (6 doubles)
                            self._joint_pos = list(struct.unpack_from("<6d", data, 192))
                            # Tool vector: xyz+rxyz at offset 624 (6 doubles)
                            self._tool_vector = list(struct.unpack_from("<6d", data, RT_TOOL_VECTOR))
                            # Actual quaternion: at offset 1384 (4 doubles)
                            self._quaternion = list(struct.unpack_from("<4d", data, 1384))
                            # Digital outputs
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
        # Drain stale data
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
                # Drain previous response
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
    """Control TopHand via CLI subprocess."""

    def __init__(self, tophand_bin="tophand", hand="right"):
        cmd = [tophand_bin, "-r" if hand == "right" else "-l"]
        cwd = str(Path(tophand_bin).parent.parent)
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=cwd,
        )
        # Wait for initialization
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
        print(f"  [TopHand CMD] {cmd}", flush=True)
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()
        # Read response
        time.sleep(0.1)
        resp = []
        while True:
            import select as _sel
            if _sel.select([self.proc.stdout], [], [], 0.0)[0]:
                line = self.proc.stdout.readline().strip()
                if line:
                    resp.append(line)
                else:
                    break
            else:
                break
        if resp:
            print(f"  [TopHand RSP] {resp}", flush=True)

    def grasp(self, name, step="pre_grasp"):
        """Apply a saved grasp pose. step = 'pre_grasp' | 'grasp'."""
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
    """Convert quaternion [x, y, z, w] to rotation_6d (first two columns of rotation matrix)."""
    r = R.from_quat(q)  # scipy expects [x, y, z, w]
    mat = r.as_matrix()
    rot6d = np.concatenate([mat[:, 0], mat[:, 1]])  # first two columns = 6 values
    return rot6d.astype(np.float32)


# ─── SpaceMouse (Hidraw) ────────────────────────────────────────────────────


class HidrawSpaceMouse:
    """Reads SpaceMouse via hidraw (easyhid) — works on Bluetooth and USB.

    Background thread polls hidraw for HID reports, caches latest state.
    action = [tx, ty, tz, roll, pitch, yaw] (normalized ~[-1, 1])
    buttons = [BTN_0, BTN_1]
    """

    _SUPPORTED_IDS = [(0x256F, 0xC63A), (0x256F, 0xC62E)]

    def __init__(self, device_path=""):
        self._device = None
        self._axes = [0.0] * 6
        self._buttons = [0, 0]
        self._running = True

        from easyhid import Enumeration

        hid = Enumeration()
        found_dev = None
        for d in hid.find():
            for vid, pid in self._SUPPORTED_IDS:
                if d.vendor_id == vid and d.product_id == pid:
                    found_dev = d
                    break
            if found_dev:
                break

        if found_dev is None:
            raise RuntimeError("No SpaceMouse found via hidraw")

        found_dev.open()
        found_dev.set_nonblocking(True)
        self._device = found_dev
        self._bytes_to_read = 13
        self._thread = threading.Thread(target=self._hidraw_loop, daemon=True)
        self._thread.start()

    def _hidraw_loop(self):
        def _to_int16(lo, hi):
            val = lo | (hi << 8)
            return val - 65536 if val >= 32768 else val

        while self._running:
            try:
                data = self._device.read(self._bytes_to_read)
                if not data:
                    data = self._device.read(self._bytes_to_read, timeout_ms=50)
                if data and len(data) >= 3:
                    channel = data[0]
                    if channel == 1 and len(data) >= 13:
                        self._axes[0] = _to_int16(data[1], data[2]) / 350.0
                        self._axes[1] = _to_int16(data[3], data[4]) / -350.0
                        self._axes[2] = _to_int16(data[5], data[6]) / 350.0
                        self._axes[3] = _to_int16(data[7], data[8]) / -350.0
                        self._axes[4] = _to_int16(data[9], data[10]) / -350.0
                        self._axes[5] = _to_int16(data[11], data[12]) / 350.0
                    elif channel == 3 and len(data) >= 2:
                        btn_byte = data[1]
                        self._buttons[0] = 1 if (btn_byte & 0x01) else 0
                        self._buttons[1] = 1 if (btn_byte & 0x02) else 0
            except Exception:
                pass
            time.sleep(0.001)

    def get_action(self):
        a = self._axes
        action = np.array([a[1], -a[0], a[2], a[4], a[3], a[5]], dtype=np.float32)
        return action, self._buttons[:]

    def close(self):
        self._running = False
        if self._device:
            try:
                self._device.close()
            except Exception:
                pass


def build_state(joint_pos, tool_vector, quaternion, gripper_pos):
    """Build the GR00T state vector: eef_9d(9) + joint_pos(6) + gripper_pos(1) = 16D."""
    eef_xyz = np.array(tool_vector[:3], dtype=np.float32)       # mm
    eef_rot6d = quat_to_rot6d(quaternion)
    eef_9d = np.concatenate([eef_xyz, eef_rot6d])              # 9D
    jp = np.array(joint_pos, dtype=np.float32)                  # 6D
    gp = np.array([gripper_pos], dtype=np.float32)              # 1D
    return np.concatenate([eef_9d, jp, gp])                     # 16D


# ─── SpaceMouse ──────────────────────────────────────────────────────────────


# ─── Main Recording Script ───────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-ip", default="192.168.5.1")
    parser.add_argument("--task", required=True,
                        help="Task description, e.g. 'pick motor housing and place on fixture'")
    parser.add_argument("--grasp-pose", required=True,
                        help="TopHand grasp pose name (e.g. grasp_shaft, grasp_housing, grasp_outer_housing)")
    parser.add_argument("--output-dir", default="recordings/cr5af_demos")
    parser.add_argument("--speed", type=float, default=50.0)
    parser.add_argument("--action-scale", type=float, default=8.0,
                        help="mm per SpaceMouse unit per step")
    parser.add_argument("--rot-scale", type=float, default=3.0,
                        help="deg per SpaceMouse rotation unit per step")
    parser.add_argument("--translation-only", action="store_true",
                        help="Only control XYZ position, ignore rotation")
    parser.add_argument("--hz", type=float, default=30.0,
                        help="Control/recording rate (Hz)")
    parser.add_argument("--dead-zone", type=float, default=0.15)
    parser.add_argument("--delta-threshold", type=float, default=0.3)
    parser.add_argument("--tophand-bin",
                        default="/home/tophand/workspaces/tophand_sdk/build/tophand",
                        help="Path to tophand binary")
    parser.add_argument("--tophand-hand", default="left", choices=["left", "right"],
                        help="TopHand hand side (left or right)")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=15)
    parser.add_argument("--hand-exposure", type=int, default=0,
                        help="D405 hand camera exposure (0=auto, e.g. 50000)")
    parser.add_argument("--hand-gain", type=int, default=0,
                        help="D405 hand camera gain (0=default, e.g. 64)")
    parser.add_argument("--table-exposure", type=int, default=0,
                        help="D455 table camera exposure (0=auto)")
    parser.add_argument("--table-gain", type=int, default=0,
                        help="D455 table camera gain (0=default)")
    parser.add_argument("--preview", action="store_true",
                        help="Show camera preview window (cv2)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print debug info every frame")
    args = parser.parse_args()

    dt = 1.0 / args.hz

    print("=" * 60)
    print("CR5AF + TopHand Data Collection for GR00T N1.7")
    print("=" * 60)
    print(f"Robot: {args.robot_ip}  Task: {args.task}")
    print(f"Grasp: {args.grasp_pose}  Rate: {args.hz} Hz")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Connect CR5AF ──
    print("\n[1/4] Connecting to CR5AF...")
    conn = CR5AFConnection(args.robot_ip)
    conn.connect()
    print("  EnableRobot:", conn.dashboard_cmd("EnableRobot()"))
    print("  SpeedFactor:", conn.dashboard_cmd(f"SpeedFactor({int(args.speed)})"))
    conn.dashboard_cmd("ResetRobot()")
    time.sleep(0.3)

    # Wait for RT data
    for _ in range(20):
        state = conn.get_state()
        if state:
            break
        time.sleep(0.1)
    if not state:
        print("ERROR: No RT data. Check port 30004.")
        sys.exit(1)
    jp, tv, quat, do = state
    print(f"  Joints: {[f'{x:.1f}' for x in jp]}")
    print(f"  TCP: x={tv[0]:.1f} y={tv[1]:.1f} z={tv[2]:.1f}")

    # ── Start TopHand ──
    print(f"\n[2/4] Starting TopHand ({args.tophand_bin})...")
    try:
        hand = TopHandCLI(tophand_bin=args.tophand_bin, hand=args.tophand_hand)
        hand.home()
        time.sleep(2)
        print("  TopHand ready.")
        # Apply pre_grasp to open hand for approach
        hand.grasp(args.grasp_pose, "pre_grasp")
        time.sleep(2)
        print(f"  Applied pre_grasp of '{args.grasp_pose}'.")
    except Exception as e:
        print(f"  ERROR starting TopHand: {e}")
        print("  Continuing without gripper control...")
        hand = None

    # ── Start Cameras ──
    print("\n[3/4] Starting cameras...")
    try:
        # D455 = table view (first detected), D405 = hand view (wrist)
        cam_serial_hand = None
        cam_serial_table = None
        import pyrealsense2 as rs
        ctx = rs.context()
        for dev in ctx.devices:
            name = dev.get_info(rs.camera_info.name)
            sn = dev.get_info(rs.camera_info.serial_number)
            if "455" in name and cam_serial_table is None:
                cam_serial_table = sn
            elif "405" in name and cam_serial_hand is None:
                cam_serial_hand = sn

        if cam_serial_table and cam_serial_hand:
            table_cam = RealSenseCamera(cam_serial_table, args.camera_width, args.camera_height, args.camera_fps, args.table_exposure, args.table_gain)
            hand_cam = RealSenseCamera(cam_serial_hand, args.camera_width, args.camera_height, args.camera_fps, args.hand_exposure, args.hand_gain)
            print(f"  D455 (table): serial={cam_serial_table}")
            print(f"  D405 (hand):  serial={cam_serial_hand}")
        else:
            # Fallback: use indices
            table_cam = RealSenseCamera("0", args.camera_width, args.camera_height, args.camera_fps, args.table_exposure, args.table_gain)
            hand_cam = RealSenseCamera("1", args.camera_width, args.camera_height, args.camera_fps, args.hand_exposure, args.hand_gain)
            print("  Using camera indices 0 and 1")
    except ImportError:
        print("  pyrealsense2 not available, using OpenCV fallback...")
        table_cam = cv2.VideoCapture(0)
        hand_cam = cv2.VideoCapture(1)
        table_cam.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        table_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
        hand_cam.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        hand_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)

    # ── Start SpaceMouse ──
    print("\n[4/4] Starting SpaceMouse...")
    try:
        sm = HidrawSpaceMouse()
    except Exception:
        print("  ERROR: SpaceMouse not found.")
        sys.exit(1)

    # Calibrate zero
    print("  Calibrating zero (keep SpaceMouse still)...")
    zero_samples = []
    for _ in range(int(1.0 / dt)):
        a, _ = sm.get_action()
        zero_samples.append(np.array(a[:6], dtype=np.float32))
        time.sleep(dt)
    zero_offset = np.median(zero_samples, axis=0)
    print(f"  Zero offset: {zero_offset}")

    # ── Recording State Machine ──
    gripper_state = 1.0  # start open
    prev_left_btn = False

    episodes = []
    current_episode = None
    frame_idx = 0
    last_sent = time.monotonic()

    print("\n" + "=" * 60)
    print("READY. Controls:")
    print("  Left button  = toggle recording on/off")
    print("  Right button = hold to move (deadman for ServoP)")
    print("  G key        = toggle gripper open/close")
    print("  Q key        = quit")
    print("  Ctrl+C       = exit")
    print(f"  Task: {args.task}")
    print("=" * 60)

    def read_camera(cam):
        if isinstance(cam, RealSenseCamera):
            return cam.read()
        else:
            ret, frame = cam.read()
            return frame if ret else None

    try:
        while True:
            loop_start = time.monotonic()
            action, buttons = sm.get_action()
            action = np.array(action, dtype=np.float32)
            action[:6] -= zero_offset

            frame_idx += 1

            # Read state
            state = conn.get_state()
            if not state:
                time.sleep(dt)
                continue
            jp, tv, quat, do = state

            # Read cameras
            table_frame = read_camera(table_cam)
            hand_frame = read_camera(hand_cam)

            # Camera preview (always on when --preview)
            if args.preview and table_frame is not None and hand_frame is not None:
                th = min(table_frame.shape[0], hand_frame.shape[0])
                tw = min(table_frame.shape[1], hand_frame.shape[1])
                preview = np.hstack([
                    cv2.resize(table_frame, (tw, th)),
                    cv2.resize(hand_frame, (tw, th)),
                ])
                ep_len = len(current_episode["state"]) if current_episode else 0
                label = f"ep={len(episodes)} len={ep_len} grip={'CLOSE' if gripper_state < 0.5 else 'OPEN'}"
                cv2.putText(preview, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("D455(table) | D405(hand)", preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("g"):
                    if gripper_state > 0.5:
                        if hand:
                            hand.grasp(args.grasp_pose, "grasp")
                        gripper_state = 0.0
                        print("\n[GRIPPER] CLOSED")
                    else:
                        if hand:
                            hand.grasp(args.grasp_pose, "pre_grasp")
                        gripper_state = 1.0
                        print("\n[GRIPPER] OPENED")

            # Build state vector
            state_vec = build_state(jp, tv, quat, gripper_state)

            # Left button = toggle recording on/off (edge-triggered)
            if buttons[0] and not prev_left_btn:
                if current_episode is None:
                    print(f"\n[EPISODE {len(episodes)}] recording...")
                    current_episode = {
                        "state": [],
                        "images_hand": [],
                        "images_table": [],
                        "timestamps": [],
                        "gripper_states": [],
                    }
                else:
                    print(f"\n[EPISODE {len(episodes)}] stopped ({len(current_episode['state'])} frames)")
                    episodes.append(current_episode)
                    current_episode = None
            prev_left_btn = buttons[0]

            # Record frame if recording
            if current_episode is not None:
                current_episode["state"].append(state_vec)
                if hand_frame is not None:
                    current_episode["images_hand"].append(cv2.resize(hand_frame, (320, 240)))
                if table_frame is not None:
                    current_episode["images_table"].append(cv2.resize(table_frame, (320, 240)))
                current_episode["timestamps"].append(time.time())
                current_episode["gripper_states"].append(gripper_state)

            # Right button = deadman for ServoP motion (hold to move)
            if not buttons[1]:
                if args.verbose and frame_idx % 10 == 0:
                    ep_len = len(current_episode["state"]) if current_episode else 0
                    print(f"\r[released] btn={buttons} ep={len(episodes)} len={ep_len}    ", end="", flush=True)
                time.sleep(dt)
                continue

            # Gripper toggle while deadman held: left was already handled above

            # Dead zone
            if np.max(np.abs(action[:6])) < args.dead_zone:
                time.sleep(dt)
                continue

            # SpaceMouse → delta (same mapping as cr5af_server)
            tx, ty, tz, pitch, roll, yaw = action[:6]
            if args.translation_only:
                rot_delta = [0.0, 0.0, 0.0]
            else:
                rot_delta = [
                    -roll * args.rot_scale,
                    pitch * args.rot_scale,
                    -yaw * args.rot_scale,
                ]
            delta = np.array([
                tx * args.action_scale,
                ty * args.action_scale,
                -tz * args.action_scale,
            ] + rot_delta)

            # Record frame
            if current_episode is not None:
                current_episode["state"].append(state_vec)
                if hand_frame is not None:
                    current_episode["images_hand"].append(cv2.resize(hand_frame, (320, 240)))
                if table_frame is not None:
                    current_episode["images_table"].append(cv2.resize(table_frame, (320, 240)))
                current_episode["timestamps"].append(time.time())
                current_episode["gripper_states"].append(gripper_state)

            # ServoP
            if np.max(np.abs(delta)) < args.delta_threshold:
                time.sleep(dt)
                continue

            pose = tv  # current TCP pose
            target = [
                pose[0] + delta[0], pose[1] + delta[1], pose[2] + delta[2],
                pose[3] + delta[3], pose[4] + delta[4], pose[5] + delta[5],
            ]
            conn.servop(*target)

            elapsed_ms = (time.monotonic() - last_sent) * 1000
            last_sent = time.monotonic()

            ep_len = len(current_episode["state"]) if current_episode else 0
            print(f"\r  x={target[0]:.1f} y={target[1]:.1f} z={target[2]:.1f} "
                  f"ep={ep_len} dt={elapsed_ms:.0f}ms    ", end="", flush=True)

            # Maintain Hz
            elapsed = time.monotonic() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\n\nExiting...")

    # Save last episode
    if current_episode is not None and len(current_episode["state"]) > 10:
        episodes.append(current_episode)
        print(f"[EPISODE {len(episodes)-1}] saved ({len(current_episode['state'])} frames)")

    # ── Save to disk ──
    print(f"\nSaving {len(episodes)} episodes to {args.output_dir}...")
    for i, ep in enumerate(episodes):
        ep_dir = Path(args.output_dir) / f"episode_{i:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        state_arr = np.array(ep["state"], dtype=np.float32)          # (T, 16)
        images_hand_arr = np.array(ep["images_hand"], dtype=np.uint8)  # (T, 240, 320, 3)
        images_table_arr = np.array(ep["images_table"], dtype=np.uint8)

        np.savez_compressed(
            ep_dir / "data.npz",
            state=state_arr,
            images_hand=images_hand_arr,
            images_table=images_table_arr,
            timestamps=np.array(ep["timestamps"]),
            gripper_states=np.array(ep["gripper_states"], dtype=np.float32),
        )
        # Save metadata
        with open(ep_dir / "meta.json", "w") as f:
            json.dump({
                "task": args.task,
                "grasp_pose": args.grasp_pose,
                "num_frames": len(ep["state"]),
                "hz": args.hz,
            }, f, indent=2)

        print(f"  episode_{i:06d}: {len(ep['state'])} frames, "
              f"state={state_arr.shape}, "
              f"hand={images_hand_arr.shape}, "
              f"table={images_table_arr.shape}")

    # Cleanup
    sm.close()
    cv2.destroyAllWindows()
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
