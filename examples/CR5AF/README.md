# CR5AF GR00T N1.7 Fine-tuning

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
.venv/bin/python deploy_cr5af.py \
  --client --server-ip <IP> --port 5555 \
  --robot-ip 192.168.5.1 --task "..." \
  --grasp-pose grasp_housing --tophand-hand left \
  --translation-only
```

## Key Design Decisions

### Rotation: TCP axis-angle, NOT RT quaternion

Doosan RT data at offset 1384 contains a quaternion that **does NOT match TCP orientation**. Always use `tool_vector[3:6]` (axis-angle degrees) directly:

```python
def rxyz_to_rot6d(rxyz_deg):
    r = R.from_rotvec(rxyz_deg, degrees=True)
    mat = r.as_matrix()
    return np.concatenate([mat[:, 0], mat[:, 1]]).astype(np.float32)
```

### Action Lookahead

Default action format (`action[t] = state[t+1]`) produces per-step deltas of ~0.03mm — far too small for the model to learn meaningful state conditioning. **Use `--lookahead 50`** in `convert_to_lerobot.py` to set `action[t] = state[t+50]`, increasing delta to ~1.5mm (50x larger signal).

Without this, the model collapses to outputting the dataset's average position regardless of the current robot state.

### Language: English

The Qwen3-VL backbone was trained on English. Always use English task descriptions.

### Action Representation: RELATIVE EEF + RELATIVE Joints + ABSOLUTE Gripper

EEF and joint actions are represented as relative deltas (better generalization). Gripper is absolute (binary open/close).

## Training History

### v1 (24 episodes, frozen VLM)
- First attempt with minimal data
- Model output: fixed single position ✅ (pipeline verified)

### v2 (103 episodes, frozen VLM, 20k steps)
- 103 episodes, different start/target positions
- Loss: 0.138
- Model output: fixed single position near dataset average
- **Lesson**: Frozen VLM cannot condition on visual state

### v3 (103 episodes, tune-visual, 20k steps)
- Unfroze visual encoder (~407M extra params)
- Loss: 0.059 (down from 0.138)
- Model output: varies with observation, tracks trajectory direction
- **Issue**: Model drifts in -Y direction regardless of target position
- **Root cause**: action delta only 0.03mm — state conditioning signal too weak

### v4 (103 episodes, tune-visual + top2 LLM layers, 10k steps)
- Added top 2 LLM layers (~144M extra params)
- Batch size: 2, Learning rate: 1e-6
- (in progress)

### v5 (103 episodes, tune-visual + top2 LLM layers, lookahead=50)
- Action delta amplified from 0.05mm to 4.4mm (88x)
- (pending)

## Files

| File | Purpose |
|------|---------|
| `cr5af_config.py` | Modality config for NEW_EMBODIMENT |
| `record_demo.py` | Data collection (Thor) |
| `convert_to_lerobot.py` | LeRobot v2 conversion with `--lookahead` |
| `deploy_cr5af.py` | Server/client inference deployment |
| `finetune_tune_visual.sh` | Training v3 (tune-visual) |
| `finetune_tune_llm.sh` | Training v4 (tune-visual + LLM layers) |
| `finetune_l50.sh` | Training v5 (tune-visual + LLM layers + lookahead=50) |
| `preview_episode.py` | Episode viewer |
