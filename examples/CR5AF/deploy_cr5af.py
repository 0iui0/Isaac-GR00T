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
    def __init__(self, serial_or_index, width=320, height=240, fps=30,
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

        # Set exposure/gain (matching record_demo.py)
        if exposure > 0 or gain > 0:
            for sensor in profile.get_device().query_sensors():
                try:
                    if exposure > 0:
                        sensor.set_option(rs.option.enable_auto_exposure, 0)
                        sensor.set_option(rs.option.exposure, float(exposure))
                    if gain > 0:
                        sensor.set_option(rs.option.gain, float(gain))
                except Exception:
                    pass

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
                            self._quaternion = list(struct.unpack_from("<4d", data, 1384))
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

    def enable_impedance(self):
        """Set low stiffness/damping for smooth ServoP (matching hil-serl cr5af_server)."""
        with self._cmd_lock:
            try:
                self._cmd_sock.sendall(b"FCSetStiffness(100,100,500,50,50,50)")
                time.sleep(0.05)
                self._cmd_sock.sendall(b"FCSetDamping(500,500,800,500,500,500)")
            except Exception:
                pass

    def servop(self, x, y, z, rx, ry, rz, gain=300):
        cmd = f"ServoP({x:.3f},{y:.3f},{z:.3f},{rx:.3f},{ry:.3f},{rz:.3f},gain={gain})"
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
        cwd = str(Path(tophand_bin).parent.parent)  # tophand_sdk root
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=cwd,
        )
        self._read_until("[OK] TopHand initialized", timeout=15)

    def _read_until(self, expected, timeout=10):
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                time.sleep(0.1)
                continue
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

def rxyz_to_rot6d(rxyz_deg):
    """Convert TCP axis-angle [rx, ry, rz] degrees → rotation_6d."""
    r = R.from_rotvec(rxyz_deg, degrees=True)
    mat = r.as_matrix()
    rot6d = np.concatenate([mat[:, 0], mat[:, 1]])
    return rot6d.astype(np.float32)


def rot6d_to_matrix(rot6d):
    """Convert 6D rotation representation back to 3x3 rotation matrix."""
    a1 = rot6d[:3]
    a2 = rot6d[3:6]
    # Orthogonalize via Gram-Schmidt
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def rxyz_to_matrix(rxyz_deg):
    """Convert TCP axis-angle [rx, ry, rz] degrees → 3x3 rotation matrix."""
    return R.from_rotvec(rxyz_deg, degrees=True).as_matrix()


def build_state_dict(joint_pos, tool_vector, gripper_pos):
    """Build GR00T state dict from CR5AF RT data. Uses TCP axis-angle directly."""
    eef_xyz = np.array(tool_vector[:3], dtype=np.float32)
    eef_rot6d = rxyz_to_rot6d(tool_vector[3:6])
    eef_9d = np.concatenate([eef_xyz, eef_rot6d])
    return {
        "eef_9d": eef_9d.reshape(1, 1, 9),
        "joint_pos": np.array(joint_pos, dtype=np.float32).reshape(1, 1, 6),
        "gripper_pos": np.array([gripper_pos], dtype=np.float32).reshape(1, 1, 1),
    }


# ─── Inference Server (runs on training machine) ─────────────────────────────

def run_server(model_path: str, port: int, modality_config_path: str = ""):
    """ZMQ REP server that accepts observations and returns actions."""
    import zmq
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    # Register custom modality config for NEW_EMBODIMENT
    if modality_config_path:
        import importlib.util
        spec = importlib.util.spec_from_file_location("cr5af_config", modality_config_path)
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)
        print(f"Registered modality config from {modality_config_path}")

    print(f"Loading model from {model_path}...")
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        model_path=model_path,
        device="cuda:0",
    )
    print("Model loaded.")

    # Load TensorRT engines for acceleration (if available)
    trt_engine_path = os.path.join(model_path.rstrip("/"), "trt_engines", "engines")
    if os.path.isdir(trt_engine_path) and any(f.endswith(".engine") for f in os.listdir(trt_engine_path)):
        try:
            repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)
            from scripts.deployment.trt_model_forward import setup_tensorrt_engines
            setup_tensorrt_engines(policy, trt_engine_path, mode="n17_full_pipeline")
            print(f"TRT engines loaded from {trt_engine_path}")
        except Exception as e:
            print(f"TRT loading failed (falling back to PyTorch): {e}")
    else:
        print(f"No TRT engines found at {trt_engine_path}, using PyTorch")

    # Keep 4 denoising steps for accuracy. TRT single sample + EMA handles smoothness.

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
            # Average multiple inferences to reduce diffusion noise
            t0 = time.monotonic()
            num_samples = 2  # TRT双样本~350ms, 精度优先
            actions = []
            for _ in range(num_samples):
                a, _ = policy.get_action(obs)
                actions.append({k: np.asarray(v) for k, v in a.items()})
            # Average across samples
            action = {}
            for k in actions[0]:
                stacked = np.stack([a[k] for a in actions], axis=0)
                action[k] = stacked.mean(axis=0)
            dt_infer = time.monotonic() - t0
            # Debug: print all timesteps for eef_9d to check trajectory
            print(f"  [infer {dt_infer*1000:.0f}ms, {num_samples} samples]", flush=True)
            for k, v in action.items():
                arr = np.asarray(v)
                if k == "eef_9d" and arr.ndim == 3:
                    # Print xyz for each timestep
                    for t in range(arr.shape[1]):
                        xyz = arr[0, t, :3]
                        print(f"  [eef_9d] t={t}: xyz=[{xyz[0]:.1f} {xyz[1]:.1f} {xyz[2]:.1f}]", flush=True)
                else:
                    flat = arr.flatten()[:min(9, arr.size)]
                    print(f"  [{k}] shape={arr.shape} first={flat.round(3)}", flush=True)
            # Convert numpy arrays to lists for msgpack serialization
            action_serializable = {
                k: v.tolist() if hasattr(v, 'tolist') else v
                for k, v in action.items()
            }
            socket.send(msgpack.packb(action_serializable))
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            socket.send(msgpack.packb({"error": str(e)}))


def run_client(server_ip: str, port: int, robot_ip: str,
               task: str, grasp_pose: str, tophand_bin: str,
               tophand_hand: str,
               hand_exposure: int, hand_gain: int,
               table_exposure: int, table_gain: int,
               hz: float, action_exec_horizon: int, speed: float,
               dry_run: bool, translation_only: bool,
               max_translation_delta: float, max_rotation_delta: float):
    """ZMQ REQ client that captures obs, sends to server, executes actions."""
    import zmq
    import msgpack
    import msgpack_numpy
    msgpack_numpy.patch()

    dt = 1.0 / hz

    # ── Connect to server ──
    print(f"\n[1/4] Connecting to inference server {server_ip}:{port}...")
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REQ)
    socket.connect(f"tcp://{server_ip}:{port}")
    socket.setsockopt(zmq.RCVTIMEO, 5000)
    socket.setsockopt(zmq.SNDTIMEO, 1000)
    print("  Connected.")

    # ── Connect CR5AF ──
    # Always connect for state reading; only send commands when not dry-run
    print(f"\n[2/4] Connecting to CR5AF ({robot_ip})...")
    conn = CR5AFConnection(robot_ip)
    conn.connect()
    if not dry_run:
        print("  EnableRobot:", conn.dashboard_cmd("EnableRobot()"))
        print("  SpeedFactor:", conn.dashboard_cmd(f"SpeedFactor({int(speed)})"))
        conn.dashboard_cmd("ResetRobot()")
        conn.enable_impedance()
    else:
        print("  DRY-RUN: reading state only, no commands sent")
    time.sleep(0.3)

    state = None
    for _ in range(20):
        state = conn.get_state()
        if state:
            break
        time.sleep(0.1)
    if not state:
        print("ERROR: No RT data from robot.")
        sys.exit(1)
    jp, tv, quat, do = state
    print(f"  TCP: x={tv[0]:.1f} y={tv[1]:.1f} z={tv[2]:.1f}")

    # ── Start TopHand ──
    hand = None
    if not dry_run:
        print(f"\n[3/4] Starting TopHand...")
        try:
            hand = TopHandCLI(tophand_bin=tophand_bin, hand=tophand_hand)
            hand.home()
            time.sleep(2)
            hand.grasp(grasp_pose, "pre_grasp")
            time.sleep(2)
            print("  TopHand ready.")
        except Exception as e:
            print(f"  ERROR: {e}")
            hand = None
    else:
        print("\n[3/4] DRY-RUN: skipping TopHand")

    # ── Start cameras ──
    print("\n[4/4] Starting cameras...")
    cam_serial_hand = None
    cam_serial_table = None
    try:
        import pyrealsense2 as rs
        ctx_rs = rs.context()
        for dev in ctx_rs.devices:
            name = dev.get_info(rs.camera_info.name)
            sn = dev.get_info(rs.camera_info.serial_number)
            if "455" in name and cam_serial_table is None:
                cam_serial_table = sn
            elif "405" in name and cam_serial_hand is None:
                cam_serial_hand = sn
        if cam_serial_table and cam_serial_hand:
            table_cam = RealSenseCamera(cam_serial_table, 640, 480, 30,
                                            exposure=table_exposure, gain=table_gain)
            hand_cam = RealSenseCamera(cam_serial_hand, 640, 480, 30,
                                       exposure=hand_exposure, gain=hand_gain)
            print(f"  D455 (table): serial={cam_serial_table}")
            print(f"  D405 (hand):  serial={cam_serial_hand}")
    except ImportError:
        print("  pyrealsense2 not available, using OpenCV...")
        table_cam = cv2.VideoCapture(0)
        hand_cam = cv2.VideoCapture(1)

    # ── Safety checker ──
    safety = SafetyChecker(
        max_translation_delta=max_translation_delta,
        max_rotation_delta=max_rotation_delta,
    )

    # ── Inference loop ──
    print("\n" + "=" * 60)
    print("READY. Press Ctrl+C to stop.")
    print("=" * 60)

    # Video history for temporal dimension
    video_history_hand = deque(maxlen=21)
    video_history_table = deque(maxlen=21)
    gripper_state = 1.0  # start open
    ema_target = None  # EMA-smoothed absolute target for smooth motion
    action_buffer = np.zeros((0, 16), dtype=np.float32)
    action_step = 0

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
            table_frame = table_cam.read() if isinstance(table_cam, RealSenseCamera) else table_cam.read()[1]
            hand_frame = hand_cam.read() if isinstance(hand_cam, RealSenseCamera) else hand_cam.read()[1]
            if table_frame is None or hand_frame is None:
                time.sleep(dt)
                continue

            # Resize to 320x240 and store in history
            table_small = cv2.resize(table_frame, (320, 240))
            hand_small = cv2.resize(hand_frame, (320, 240))
            video_history_table.append(table_small)
            video_history_hand.append(hand_small)

            # Need at least 2 frames for temporal dim
            if len(video_history_hand) < 2 or len(video_history_table) < 2:
                time.sleep(dt)
                continue

            # Build state vector: TCP axis-angle → rot6d (consistent with training data)
            eef_rot6d = rxyz_to_rot6d(tv[3:6])
            eef_9d = np.concatenate([np.array(tv[:3], dtype=np.float32), eef_rot6d])

            # Build video: (B=1, T=2, H=240, W=320, C=3) — current + previous frame
            hand_frames = np.stack([video_history_hand[-2], video_history_hand[-1]], axis=0)
            table_frames = np.stack([video_history_table[-2], video_history_table[-1]], axis=0)

            # Build observation dict — double expand_dims for (B, T) dims
            observation = {
                "video": {
                    "hand_view": hand_frames[None, ...].astype(np.uint8),    # (1, 2, 240, 320, 3)
                    "table_view": table_frames[None, ...].astype(np.uint8),  # (1, 2, 240, 320, 3)
                },
                "state": {
                    "eef_9d": eef_9d[None, None, ...],          # (1, 1, 9)
                    "joint_pos": np.array(jp, dtype=np.float32)[None, None, ...],  # (1, 1, 6)
                    "gripper_pos": np.array([gripper_state], dtype=np.float32)[None, None, ...],  # (1, 1, 1)
                },
                "language": {"annotation.human.task_description": [[task]]},
            }

            # Send to server, get action
            try:
                request = msgpack.packb({"observation": observation})
                socket.send(request)
                raw = socket.recv()
                response = msgpack.unpackb(raw)
                if isinstance(response, dict) and "error" in response:
                    print(f"\nServer error: {response['error']}")
                    time.sleep(dt)
                    continue
                action = response
            except (zmq.error.Again, zmq.error.ZMQError):
                print("\nServer timeout/connection error, reconnecting...")
                # REQ socket must recv before next send — rebuild after timeout
                socket.close()
                socket = ctx.socket(zmq.REQ)
                socket.connect(f"tcp://{server_ip}:{port}")
                socket.setsockopt(zmq.RCVTIMEO, 5000)
                socket.setsockopt(zmq.SNDTIMEO, 1000)
                time.sleep(dt)
                continue

            # Process action dict from server
            # Server sends: {"eef_9d": (1, T, 9), "joint_pos": (1, T, 6), "gripper_pos": (1, T, 1)}
            # msgpack converts numpy → nested lists, we convert back and squeeze batch dim
            action_dict = {
                k: np.array(v, dtype=np.float32).squeeze(0) for k, v in action.items()
            }

            # Use first timestep (most accurate prediction, closest to current state)
            abs_eef = action_dict["eef_9d"][0]       # (9,) absolute target xyz+rot6d
            abs_joints = action_dict["joint_pos"][0]  # (6,) absolute target joint angles
            gripper_target = float(action_dict["gripper_pos"][0][0])  # absolute: 0=close, 1=open

            # Compute deltas from current pose to absolute target
            delta_xyz = abs_eef[:3] - np.array(tv[:3], dtype=np.float32)   # mm
            # Rotation delta: compute via rotation matrices
            if translation_only:
                delta_rxyz = np.zeros(3, dtype=np.float32)  # no rotation change
            else:
                target_mat = rot6d_to_matrix(abs_eef[3:9])
                current_mat = rxyz_to_matrix(tv[3:6])
                delta_mat = target_mat @ current_mat.T
                delta_rxyz = R.from_matrix(delta_mat).as_rotvec(degrees=True)

            # Safety: clip deltas, then apply to current pose
            safe_delta_xyz, safe_delta_rxyz, ok = safety.check_and_clip(
                delta_xyz, delta_rxyz, np.array(tv[:3]),
            )

            if not ok:
                print("\n\nSAFETY STOP! Too many violations.")
                break

            # Camera visualization (like record_demo: D455 left | D405 right)
            if table_frame is not None and hand_frame is not None:
                th = min(table_frame.shape[0], hand_frame.shape[0])
                tw = min(table_frame.shape[1], hand_frame.shape[1])
                preview = np.hstack([
                    cv2.resize(table_frame, (tw, th)),
                    cv2.resize(hand_frame, (tw, th)),
                ])
                label = (f"D455(table) | D405(hand)"
                         f"  delta=[{safe_delta_xyz[0]:.1f} {safe_delta_xyz[1]:.1f} {safe_delta_xyz[2]:.1f}]mm"
                         f"  grip={'CLOSE' if gripper_state < 0.5 else 'OPEN'}")
                cv2.putText(preview, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("D455(table) | D405(hand)", preview)
                cv2.waitKey(1)

            # ServoP: EMA-smoothed absolute target to reduce diffusion noise jitter
            alpha = 0.6  # higher = faster tracking, lower = smoother
            new_target = np.array([abs_eef[0], abs_eef[1], abs_eef[2]], dtype=np.float32)
            if ema_target is None:
                ema_target = new_target.copy()
            else:
                ema_target = alpha * new_target + (1 - alpha) * ema_target

            if not dry_run:
                target_x, target_y, target_z = ema_target[0], ema_target[1], ema_target[2]
                # Dead zone: skip ServoP if delta < 1.5mm to suppress jitter at target
                delta = np.linalg.norm(np.array([target_x, target_y, target_z]) - np.array(tv[:3]))
                if delta < 1.5:
                    time.sleep(dt)
                    continue
                # Safety: only enforce workspace limits (not delta limits)
                if not safety.check_workspace(np.array([target_x, target_y, target_z])):
                    print("\n\nSAFETY: model target outside workspace!")
                    break
                if translation_only:
                    conn.servop(target_x, target_y, target_z, tv[3], tv[4], tv[5])
                else:
                    target_mat = rot6d_to_matrix(abs_eef[3:9])
                    target_rxyz = R.from_matrix(target_mat).as_rotvec(degrees=True)
                    conn.servop(target_x, target_y, target_z,
                                float(target_rxyz[0]), float(target_rxyz[1]), float(target_rxyz[2]))

            # Gripper
            if not dry_run and abs(gripper_target - gripper_state) > 0.5:
                if gripper_target < 0.5 and hand:
                    hand.grasp(grasp_pose, "grasp")
                elif gripper_target > 0.5 and hand:
                    hand.grasp(grasp_pose, "pre_grasp")
                gripper_state = gripper_target

            # Maintain Hz
            elapsed = time.monotonic() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\n\nStopping...")

    # Cleanup
    if not dry_run:
        conn.dashboard_cmd("ResetRobot()")
    conn.close()
    if hand:
        hand.close()
    if isinstance(table_cam, RealSenseCamera):
        table_cam.close()
    if isinstance(hand_cam, RealSenseCamera):
        hand_cam.close()
    socket.close()
    ctx.term()
    print("Done.")


# ─── Safety ───────────────────────────────────────────────────────────────────

class SafetyChecker:
    """Clip per-step deltas and enforce workspace limits.

    Works entirely on deltas (delta_xyz_mm, delta_rxyz_deg) to avoid
    confusion between absolute positions and incremental movements.
    """

    def __init__(
        self,
        max_translation_delta: float = 10.0,   # mm per step
        max_rotation_delta: float = 5.0,        # deg per step
        workspace_min_xyz: tuple = (200.0, -400.0, 0.0),
        workspace_max_xyz: tuple = (800.0, 200.0, 600.0),
        max_consecutive_violations: int = 10,
    ):
        self.max_translation_delta = max_translation_delta
        self.max_rotation_delta = max_rotation_delta
        self.workspace_min = np.array(workspace_min_xyz, dtype=np.float32)
        self.workspace_max = np.array(workspace_max_xyz, dtype=np.float32)
        self.max_consecutive_violations = max_consecutive_violations
        self.violation_count = 0

    def check_numerics(self, arr: np.ndarray) -> bool:
        """Return True if NaN or inf detected."""
        return np.any(np.isnan(arr)) or np.any(np.isinf(arr))

    def clip_delta(
        self, delta_xyz: np.ndarray, delta_rxyz: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Clip translation and rotation deltas independently."""
        # Translation: clip magnitude
        trans_norm = np.linalg.norm(delta_xyz)
        if trans_norm > self.max_translation_delta:
            delta_xyz = delta_xyz / trans_norm * self.max_translation_delta
        # Rotation: clip per-axis
        delta_rxyz = np.clip(delta_rxyz, -self.max_rotation_delta, self.max_rotation_delta)
        return delta_xyz, delta_rxyz

    def check_workspace(self, target_xyz: np.ndarray) -> bool:
        """Return True if target xyz is within workspace."""
        return np.all(target_xyz >= self.workspace_min) and \
               np.all(target_xyz <= self.workspace_max)

    def check_and_clip(
        self,
        delta_xyz: np.ndarray,
        delta_rxyz: np.ndarray,
        current_xyz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Apply all safety layers. Returns (clipped_delta_xyz, clipped_delta_rxyz, ok).
        ok=False means too many consecutive violations → emergency stop.
        """
        if self.check_numerics(delta_xyz) or self.check_numerics(delta_rxyz):
            self.violation_count += 1
            return delta_xyz * 0, delta_rxyz * 0, False

        delta_xyz, delta_rxyz = self.clip_delta(delta_xyz, delta_rxyz)

        # Workspace: check if current + delta is within bounds
        target_xyz = current_xyz + delta_xyz
        if not self.check_workspace(target_xyz):
            # Clamp delta so target stays in workspace
            clamped = np.clip(target_xyz, self.workspace_min, self.workspace_max)
            delta_xyz = clamped - current_xyz
            self.violation_count += 1
        else:
            self.violation_count = 0

        if self.violation_count >= self.max_consecutive_violations:
            return delta_xyz, delta_rxyz, False

        return delta_xyz, delta_rxyz, True


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
    parser.add_argument("--modality-config-path",
                        default="examples/CR5AF/cr5af_config.py",
                        help="Path to custom modality config for NEW_EMBODIMENT")
    parser.add_argument("--server-ip", default="192.168.16.158")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--model-path", default="",
                        help="Path to fine-tuned model checkpoint (required for server/direct)")
    parser.add_argument("--robot-ip", default="192.168.5.1")
    parser.add_argument("--task", default="",
                        help="Task description for the policy (required for client/direct)")
    parser.add_argument("--grasp-pose", default="",
                        choices=["grasp_shaft", "grasp_housing"],
                        help="Grasp pose name (required for client/direct)")
    parser.add_argument("--translation-only", action="store_true",
                        help="Only use xyz from model output, keep rotation fixed")
    parser.add_argument("--tophand-bin",
                        default="/home/tophand/workspaces/tophand_sdk/build/tophand")
    parser.add_argument("--tophand-hand", default="left", choices=["left", "right"],
                        help="TopHand hand side")
    parser.add_argument("--hand-exposure", type=int, default=0,
                        help="D405 hand camera exposure (0=auto)")
    parser.add_argument("--hand-gain", type=int, default=0,
                        help="D405 hand camera gain (0=default)")
    parser.add_argument("--table-exposure", type=int, default=0,
                        help="D455 table camera exposure (0=auto)")
    parser.add_argument("--table-gain", type=int, default=0,
                        help="D455 table camera gain (0=default)")
    parser.add_argument("--hz", type=float, default=8.0,
                        help="Control rate (Hz)")
    parser.add_argument("--action-exec-horizon", type=int, default=5,
                        help="Number of action steps to execute before re-inferring")
    parser.add_argument("--speed", type=float, default=50.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print actions without executing ServoP (for testing)")
    parser.add_argument("--max-translation-delta", type=float, default=10.0,
                        help="Max EEF translation per step in mm (start small!)")
    parser.add_argument("--max-rotation-delta", type=float, default=5.0,
                        help="Max EEF rotation per step in degrees")
    args = parser.parse_args()

    # Validate: model-path required for server/direct, task+grasp for non-server
    if not args.client and not args.model_path:
        parser.error("--model-path is required for server/direct mode")
    if not args.server:
        if not args.task:
            parser.error("--task is required for client/direct mode")
        if not args.grasp_pose:
            parser.error("--grasp-pose is required for client/direct mode")

    if args.server:
        run_server(args.model_path, args.port,
                   modality_config_path=args.modality_config_path)
        return

    if args.client:
        speed = min(args.speed, 100)  # SpeedFactor valid range is 0-100
        run_client(args.server_ip, args.port, args.robot_ip,
                   task=args.task, grasp_pose=args.grasp_pose,
                   tophand_bin=args.tophand_bin, tophand_hand=args.tophand_hand,
                   hand_exposure=args.hand_exposure, hand_gain=args.hand_gain,
                   table_exposure=args.table_exposure, table_gain=args.table_gain,
                   hz=args.hz, action_exec_horizon=args.action_exec_horizon,
                   speed=speed, dry_run=args.dry_run,
                   translation_only=args.translation_only,
                   max_translation_delta=args.max_translation_delta,
                   max_rotation_delta=args.max_rotation_delta)
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
    conn.enable_impedance()
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

            state_dict = build_state_dict(jp, tv, gripper_state)

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
