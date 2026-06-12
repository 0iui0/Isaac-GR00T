# CR5AF GR00T N1.7 Fine-tuning & Deployment

Doosan CR5AF 6-axis arm + TopHand dexterous hand fine-tuning pipeline for GR00T N1.7.

## Quick Start

```bash
# 1. Record demos (Thor)
.venv/bin/python record_demo.py \
  --task "Grasp the motor shaft with sleeve..." \
  --grasp-pose grasp_housing \
  --output-dir recordings/cr5af_demos \
  --translation-only --preview

# 2. Convert to LeRobot v2
.venv/bin/python convert_to_lerobot.py \
  --input-dir recordings/cr5af_demos \
  --output-dir datasets/cr5af_v2 \
  --lookahead 50

# 3. Fine-tune
bash finetune_l50.sh

# 4. Deploy
# Training machine:
CUDA_VISIBLE_DEVICES=1 .venv/bin/python deploy_cr5af.py \
  --server --model-path <checkpoint> --port 5555

# Thor:
cd ~/workspaces/hil-serl && .venv/bin/python deploy_cr5af.py \
  --client --server-ip <IP> --port 5555 \
  --robot-ip 192.168.5.1 --task "..." \
  --grasp-pose grasp_housing --tophand-hand left \
  --translation-only --speed 100
```

---

## Lessons Learned

### 1. Rotation: TCP axis-angle, NOT RT quaternion

**问题**: 推理时机械臂往 -Y 方向漂移 60-70°，完全无法控制。

**根因**: Doosan RT 数据中 offset 1384 的 quaternion **不代表 TCP 姿态**。对比验证：`quat → rot6d` 和 `tool_vector[3:6] → rot6d` 误差达 120°。

**修复**: 始终用 `tool_vector[3:6]`（TCP axis-angle in degrees）计算 rot6d。

```python
# ✅ 正确
def rxyz_to_rot6d(rxyz_deg):
    r = R.from_rotvec(rxyz_deg, degrees=True)
    mat = r.as_matrix()
    return np.concatenate([mat[:, 0], mat[:, 1]]).astype(np.float32)

# ❌ 错误
# def quat_to_rot6d(q):
#     r = R.from_quat(q)  # 这个 quaternion 不代表 TCP！
#     ...
```

### 2. Action Lookahead: state[t+1] 信号太弱

**问题**: 模型输出始终停在训练数据平均值，不响应当前状态。

**根因**: `action[t] = state[t+1]` 产生每步仅 ~0.03mm 的 delta。模型学到的状态条件信号 SNR 太低，collapse 到"预测平均位置"。

| Lookahead | Per-step delta | 数据统计 max/step | 模型行为 |
|-----------|---------------|------------------|---------|
| 1 (默认) | 0.05mm | 0.3mm | 固定输出，不动 |
| 50 | **4.4mm** | 15mm | 能跟踪方向 ✅ |

**修复**: `convert_to_lerobot.py` 加 `--lookahead 50`。

```python
def compute_actions(states, lookahead=50):
    actions[t] = states[t + lookahead]  # 放大信号 50 倍
```

### 3. Frozen VLM 不足以进行视觉条件控制

**问题**: 即使 103 episodes，frozen VLM + DiT head 输出仍无视当前图像。

**根因**: 1.09B DiT head 参数无法单独学会视觉条件策略。VLM backbone 被冻住，不产生 task-specific 特征。

| 训练配置 | 可训练参数 | Loss | 行为 |
|---------|-----------|------|------|
| DiT only | 1.09B (34.7%) | 0.138 | 固定输出 ✅ pipeline 验证 |
| +tune-visual | 1.50B (47.7%) | 0.059 | 有轨迹方向 ✅ |
| +tune-visual+top2LLM | 1.50B (47.7%) | 0.054 | =tune-visual 无改善 |
| +lookahead=50 | 1.50B (47.7%) | 0.621 | 能跟踪目标 ✅ 精度不够 ❌ |

**结论**: RTX 5090 32GB 无法全调 LLM（需要 80GB+）。如需更高精度需：
- A100/H100 80GB 开 `--tune-llm`
- 或 500+ episodes 补偿 frozen backbone

### 4. Camera FPS 不匹配导致时序混乱

**问题**: 推理时模型输出杂乱，时序特征不对。

**根因**: 训练数据转换用 `--fps 30`，但 deploy 相机跑在 15fps。`delta_indices=[-20, 0]` 对应的时间窗口：
- 训练: 30fps × 20 帧 = **667ms** 历史
- 推理: 15fps × 20 帧 = **1333ms** 历史（2 倍！）

**修复**: deploy 相机 fps 改为 30。

### 5. D405 曝光设置不一致

**问题**: 推理时 D405 图像过曝或偏暗，模型输出异常。

**根因**: Deploy `RealSenseCamera` 未设置曝光参数，自动曝光在不同场景下会产生不同图像。

**修复**: deploy 加 `--hand-exposure`/`--hand-gain`/`--table-exposure`/`--table-gain` 参数，录制和推理用**相同手动曝光值**。

### 6. ZMQ REQ 协议状态错误

**问题**: Server 超时后 client 无法再次发送，报 `Operation cannot be accomplished in current state`。

**根因**: ZMQ REQ socket 必须严格遵循 send→recv 顺序。超时后 socket 状态损坏。

**修复**: 超时异常处理中重建 socket。

### 7. 推理延迟优化历程

| 方案 | 延迟 (2 samples) | 说明 |
|------|-----------------|------|
| PyTorch 3 samples | 600ms | 精度最高，延迟太大 |
| Torch.compile | 600ms | DiT+Qwen 黑盒，compile 无效 |
| **TRT full pipeline (batch=1)** | **350ms** | 7 engines, ViT+LLM+Action Head |
| TRT 1 sample + EMA | 195ms | 精度差，抖动 |
| 双 GPU 并行推理 | 580ms | CUDA 上下文切换抵消收益 ❌ |
| TRT batch_size=2 构建 | ❌ 失败 | Myelin 编译器 bug on Thor |

**结论**: TRT 2 samples = 350ms 是当前最优解。Thor 128GB 显存受限于 TRT 编译器，无法用 batch_size=2。

### 8. ServoP 阻抗控制

**问题**: 放置阶段机械臂微抖。

**根因**: 扩散模型预测噪声 + ServoP 高频更新 = 微抖。

**修复**:
- `set_impedance()` 是无效命令（Doosan 没有此 API）
- ✅ `FCSetStiffness + FCSetDamping` 才是正确的阻抗控制
- ✅ `ServoP(..., gain=300)` 降低 servo 增益让运动更柔顺

### 9. TopHand pre_grasp 不生效

**问题**: 部署时 TopHand 不进入 pre_grasp 状态。

**根因**: 
1. `subprocess.Popen` 缺少 `cwd` 参数 → 找不到 `assets/grasp_poses.json`
2. `send_cmd` 缺少 `time.sleep(0.1)` 读响应

**修复**: 对齐 `record_demo.py` 的 TopHandCLI 实现。

### 10. Doosan API 注意事项

| API | 行为 | 用于 |
|-----|------|------|
| `ServoP(x,y,z,rx,ry,rz,gain=X)` | 非阻塞，中断可 | 实时控制 ✅ |
| `MovL(...)` | **阻塞**，完成后才返回 | 归位、点到点移动 |
| `set_impedance()` | ❌ **不存在** | — |
| `FCSetStiffness + FCSetDamping` | 设置柔顺参数 | 阻抗控制 ✅ |
| `SpeedFactor(X)` | X 范围 0-100 | 速度限制 ✅ |

---

## Training History

### v1 (24 episodes, frozen VLM)
- Pipeline 验证：数据采集→转换→训练→部署全链路跑通
- Loss: - | 结果: 固定输出

### v2 (103 episodes, frozen VLM, 20k steps)
- Loss: 0.138 | 结果: 固定输出（与 v1 相同）
- 教训: Frozen VLM 不够

### v3 (103 episodes, tune-visual, 20k steps)
- Loss: 0.059 | 结果: 有轨迹方向
- 问题: -Y 方向漂移（root cause: action delta 0.05mm）

### v4 (103 episodes, tune-visual + top2 LLM, 10k steps)
- Loss: 0.054 | 结果: =v3 无明显改善
- 教训: 2 层 LLM 不够，RNN 瓶颈在视觉编码

### v5 (103 eps success + 30 fail, tune-visual + top2 LLM, lookahead=50)
- Loss: 0.621 | 结果: 能跟踪目标，精度 ~5mm
- **最佳方案** ✅ 但精度还不够任务要求

### v6 (109 eps success only, tune-visual + top2 LLM, lookahead=50)
- 仅使用成功数据训练（VLA=BC，失败数据不可用于行为克隆）
- 正在训练中... 🔄

### IQL Critic v2 (174 eps, 100k steps)
- Q(s,a) 范围: [-0.05, 1.08] — 成功/失败区分清晰 ✅
- 失败数据: 仅用于 Critic + Success Classifier，不用于 GR00T ✅
- **瓶颈**: 推理时 GR00T 的 (s, a) 偏离训练分布 → Critic 输出 flat

---

## Pipeline 数据闭环计划

### 当前状态 (v6)
```
数据 → GR00T BC 训练 → 部署推理 → 成功轨迹回流GR00T → 持续改进
       仅成功数据      实时控制     +失败数据→QGF Critic
                                   +Classifier自动化标注
```

### Success Classifier (下一步)
训练一个轻量 CNN，输入 D405/D455 图像，输出 success/fail 概率:
- 部署时自动标注 reward，无需人工按 F 键
- 参考 hil-serl 的成功检测方案
- 训练数据: 现有 174 eps + 部署中新增

### 人类接管检测 (Human Takeover Detection)
部署时检测"SpaceMouse 激活 + ServoP 停止"= 人类接管:
- 接管前 N 步标记为 fail → 加入 Critic 训练
- 接管后的操作 → 成功则加入 GR00T 训练（分布外探索）
- 实现数据自动回流闭环

### 数据闭环流程
```
GR00T 部署
  → 成功轨迹 → 加入GR00T训练集 → 重训GR00T (BC) ✅
  → 失败轨迹 → QGF Critic + Success Classifier ✅
  → 人类接管 → 接管前→Critic / 接管后→GR00T ✅
  → 自动标注 → Success Classifier 替代人工按键
```

### 6维力觉集成 (规划中)
CR5AF RT 数据 (offset 1304) 包含 6 维力传感器，可扩展 state 到 22D:
- 接触检测: 力值突变 → 判断是否接触
- 对齐纠正: x/y 侧向力 → 判断是否对中
- 插入深度: Z方向力变化 → 判断插到位
- 密集 reward: 力值变化作为 QGF 的连续 reward（比 0/1 更高效）

### 多任务扩展 (规划中)
GR00T N1.7 通过 language conditioning 区分任务，共享 VLM backbone:
- 单模型支持多个接触型任务
- 需要当前任务精度达标后再扩展

---

## Files

| File | Purpose |
|------|---------|
| `cr5af_config.py` | Modality config for NEW_EMBODIMENT |
| `record_demo.py` | Data collection on Thor |
| `convert_to_lerobot.py` | LeRobot v2 conversion with `--lookahead` |
| `deploy_cr5af.py` | Server/client deployment (TRT + PyTorch) |
| `finetune_tune_visual.sh` | Training v3 |
| `finetune_tune_llm.sh` | Training v4 |
| `finetune_l50.sh` | Training v5 (recommended) |
| `train_iql_critic.py` | IQL Critic + Value training for QGF RL post-training |
| `preview_episode.py` | Episode viewer |
| `README.md` | This file |

## Model Checkpoints

| Version | Path | Size |
|---------|------|------|
| v2 | `/tmp/cr5af_finetune_v2/grasp-housing-v2-103eps` | 9GB |
| v3 | `/tmp/cr5af_finetune_v3/grasp-housing-v3-tunevisual` | 9GB |
| v4 | `/tmp/cr5af_finetune_v4/grasp-housing-v4-tunellm` | 9GB |
| v5 | `/tmp/cr5af_finetune_v5/grasp-housing-v5-l50` | 9GB |
| IQL Critic | `/tmp/cr5af_iql_critic/critic.pt` + `value.pt` | 1MB |
| RL Dataset | `/datasets/cr5af_grasp_housing_qgf.npz` | 100MB |
| RL Data (LeRobot) | `/datasets/cr5af_grasp_housing_l50_rl` | 135GB |
