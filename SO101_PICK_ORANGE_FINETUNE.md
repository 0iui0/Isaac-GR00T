# SO101 PickOrange 微调 GR00T N1.7 操作指南

## 前置条件

```bash
export HF_HOME=<your_hf_cache_path>   # 例如: ~/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0         # 或 0,1 使用多卡
```

## 已完成的步骤

### 1. 数据转换 (已完成)
- 源文件: Leisaac datasets `pick_orange.hdf5` (42GB, 15 episodes, 5 successful)
- 输出目录: `/datasets/so101_pick_orange_lerobot/`
- 格式: LeRobot v2 (parquet + MP4 video)
- 转换脚本: `scripts/convert_hdf5_to_lerobot.py`

### 2. 代码修改 (已完成)

| 文件 | 修改 | 原因 |
|------|------|------|
| `gr00t/model/gr00t_n1d7/gr00t_n1d7.py` | `get_backbone_cls()` 增加 `os.path.isdir()` 支持 | 本地路径 backbone |
| `gr00t/model/gr00t_n1d7/gr00t_n1d7.py` | `device`/`dtype` 属性改用 `for p` 循环代替 `next(iter())` | DataParallel 下 StopIteration |
| `gr00t/experiment/launch_finetune.py` | 使用 `$HF_HOME` 环境变量定位 backbone | 去掉硬编码路径 |
| `gr00t/experiment/launch_finetune.py` | 添加 gradient checkpointing + 冻结 VLLN/projector | 降低显存需求 |
| `gr00t/experiment/experiment.py` | `report_to` 改为同时记录 tensorboard 和 wandb | 实验可视化 |
| `gr00t/eval/run_gr00t_server.py` | （已回退 default_language，由客户端传递） | 客户端通过 `--policy_language_instruction` 传 |
| leisaac `policy/base.py` | `call_endpoint` 增加 error dict 检查 | 服务器错误时不再 KeyError |

### 3. 基础模型下载 (已完成)
- GR00T-N1.7-3B: `$HF_HOME/nv-community/GR00T-N1.7-3B/`
- Cosmos-Reason2-2B: `$HF_HOME/Cosmos-Reason2-2B-git/`
- config.json 和 processor_config.json 已指向本地路径

## 显存优化

### 已应用的优化 (在 launch_finetune.py 中)

```python
# 冻结 VLLN 和 projector，只微调 diffusion model
config.model.tune_vlln = False
config.model.tune_projector = False

# Gradient checkpointing 减少激活值显存
config.training.gradient_checkpointing = True
```

- `tune_vlln = False`: 冻结 VLLN 模块参数 (节省梯度 + 优化器状态)
- `tune_projector = False`: 冻结 projector 层
- `gradient_checkpointing = True`: 用重计算换显存，激活值显存从 ~6GB 降至 ~1.5GB

### 如果仍然 OOM：利用 GPU 1 显存

如果单卡 32GB 仍然不够，可以使用 DeepSpeed ZeRO-2 将优化器状态分片到 GPU 0 和 GPU 1：

```bash
# 1. 先确认 GPU 1 有足够空闲显存
nvidia-smi

# 2. 设置使用两张卡
export CUDA_VISIBLE_DEVICES=0,1

# 3. 添加 --num-gpus 2 参数到训练命令
```

如果 DeepSpeed 不可用，备选方案是 PyTorch 的 `device_map="auto"` 自动分片。

---

## 显存需求总结

### 参数分布

| 模块 | 参数量 | 训练策略 |
|------|--------|----------|
| Backbone (Qwen3-VL 2B) | 2.05B | 冻结 |
| VLLN + Self-Attention | 0.20B | 冻结 |
| Projector (State/Action Encoder/Decoder) | 0.31B | 冻结 |
| **DiT (Diffusion Transformer)** | **1.09B** | **训练** |

### 各场景显存估算

| 配置 | 可训练 | 权重 | 梯度+优化器 | 激活值 | 总计 | 说明 |
|------|--------|------|------------|--------|------|------|
| **当前最优** (grad ckpt + 冻结 VLLN/projector) | 1.09B | ~8 GB | ~13 GB | ~1.5 GB | **~26 GB** | ✅ 单卡 32GB 能跑 |
| 原始配置 (冻结 backbone, 无 grad ckpt) | 1.60B | ~8 GB | ~19 GB | ~6 GB | **~36 GB** | ❌ 32GB OOM |
| 放开 tune_projector (激活 projector, grad ckpt) | 1.40B | ~8 GB | ~17 GB | ~1.5 GB | **~31 GB** | ⚠️ 32GB 勉强 |
| 全参微调 (3B 全部训练) | 3.14B | ~12 GB | ~37 GB | ~6 GB | **~55 GB** | ❌ 需要 A100-80G |

> 权重 ~8 GB 的构成：3.14B fp32 = 12.6 GB，但大部分冻结参数保持 bf16 即 ~6.3 GB，加上可训练部分的 fp32 copy，总和约 8 GB。
> 梯度+优化器 = 可训练参数量 × 4 bytes × 3 (fp32 weights copy + momentum + variance)。

### GPU 选型建议

| GPU | 显存 | 当前配置 | 推荐用途 |
|-----|------|----------|----------|
| RTX 4090 | 24 GB | ❌ 不够 | 需上 LoRA/QLoRA |
| **RTX 5090** | **32 GB** | **✅ 够但紧凑** | 冻结 backbone + grad ckpt，batch=4 |
| A100-40GB / RTX 6000 Ada | 40 GB | ✅ 充裕 | 可放开 tune_projector，batch=8 |
| A100-80GB / H100 | 80 GB | ✅ 非常充裕 | 全参数微调，大 batch |
| 2× RTX 5090 (ZeRO-2) | 64 GB | ✅ 充裕 | 可放开 tune_projector，分摊优化器状态 |

### 当前配置的显存使用明细

```
┌────────────────────────────────┐
│  CUDA context + misc  ~1.0 GB │
│  冻结参数 (bf16)      ~4.1 GB │  backbone + VLLN + projector
│  可训练参数 (bf16)    ~2.2 GB │  DiT
│  可训练 fp32 copy     ~4.4 GB │  DiT 参数的 fp32 副本
│  梯度 (fp32)          ~4.4 GB │
│  Adam exp_avg (fp32)  ~4.4 GB │
│  Adam exp_avg_sq (fp32)~4.4 GB│
│  激活值 (grad ckpt)   ~1.5 GB │
│  ──────────────────────────── │
│  总计                 ~26 GB  │  < 32 GB ✅
└────────────────────────────────┘
```

---

## 步骤 4：执行微调

```bash
export HF_HOME=<your_hf_cache_path>
CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python gr00t/experiment/launch_finetune.py \
    --base-model-path $HF_HOME/nv-community/GR00T-N1.7-3B \
    --dataset-path /datasets/so101_pick_orange_lerobot \
    --modality-config-path examples/SO101_pick_orange/so101_config.py \
    --embodiment-tag NEW_EMBODIMENT \
    --output-dir /datasets/so101_pick_orange_finetune \
    --max-steps 2000 \
    --save-steps 500 \
    --save-total-limit 5 \
    --warmup-ratio 0.05 \
    --weight-decay 1e-5 \
    --learning-rate 1e-4 \
    --global-batch-size 4 \
    --gradient-accumulation-steps 4 \
    --num-gpus 1 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 2 \
    --shard-size 1024 \
    --num-shards-per-epoch 100000 \
    --episode-sampling-rate 0.1 \
    --use-wandb true \
    --wandb-project so101_pick_orange
```

> 首次使用 wandb 需要先登录：`wandb login`（API key: https://wandb.ai/authorize）。
> 如果 OOM，将 `CUDA_VISIBLE_DEVICES=0` 改为 `CUDA_VISIBLE_DEVICES=0,1`，`--num-gpus 1` 改为 `--num-gpus 2`。

## 步骤 5：启动推理服务器

微调完成后：

```bash
export HF_HOME=<your_hf_cache_path>
source .venv/bin/activate

python gr00t/eval/run_gr00t_server.py \
    --model-path /datasets/so101_pick_orange_finetune/checkpoint-2000 \
    --embodiment-tag new_embodiment \
    --port 5555
```

## 步骤 6：在 tmux 中运行推理

```bash
# 进入 tmux
tmux attach -t leisaac-docker

# 在容器内执行
cd /workspace/isaaclab/isaac-lab-scripts/leisaac
python scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --eval_rounds=10 \
    --policy_type=gr00tn1.5 \
    --policy_host=localhost \
    --policy_port=5555 \
    --policy_timeout_ms=5000 \
    --policy_action_horizon=16 \
    --policy_language_instruction="Pick up the orange and place it on the plate" \
    --device=cuda \
    --enable_cameras
```

> 注意：容器使用 host 网络，`localhost:5555` 直接访问主机的推理服务器。
> `--policy_language_instruction` 将任务描述传递给策略客户端。

---

## 文件清单

| 路径 | 说明 |
|------|------|
| `/datasets/so101_pick_orange_lerobot/` | LeRobot v2 格式数据集 |
| `/datasets/so101_pick_orange_lerobot/meta/` | info.json, episodes.jsonl, tasks.jsonl, modality.json, stats.json |
| `/datasets/so101_pick_orange_lerobot/data/train-00000-of-00001/data.parquet` | 状态+动作数据 |
| `/datasets/so101_pick_orange_lerobot/videos/` | MP4 视频 (front + wrist) |
| `$HF_HOME/nv-community/GR00T-N1.7-3B/` | 基础模型权重 |
| `$HF_HOME/Cosmos-Reason2-2B-git/` | Cosmos-Reason2-2B backbone |
| `examples/SO101_pick_orange/so101_config.py` | SO101 modality 配置 |
| `scripts/convert_hdf5_to_lerobot.py` | HDF5 转 LeRobot 脚本 |

---

## 踩坑记录

### 训练阶段

| 坑 | 现象 | 修复 |
|---|---|---|
| StopIteration | DataParallel 下 `next(iter(parameters()))` 空 | `gr00t_n1d7.py` 改为 `for p` 循环 |
| tyro bool 解析 | `--use-wandb true` 报错 | 去掉值，只用 `--use-wandb` |
| wandb 国内超时 | `wandb.init()` 90s timeout | 设代理 `https_proxy` |
| wandb 401 | .netrc 里旧 key 被撤销 | `wandb login --relogin` |
| setuptools | v82 删除 `pkg_resources` | `pip install "setuptools<70"` |
| 显存 OOM | 3B 模型 32GB 不够 | 冻结 VLLN/projector + gradient checkpointing |

### 推理阶段

| 坑 | 现象 | 修复 |
|---|---|---|
| language 为 None | 仿真不发 task description，服务器 `check_observation` 断言失败 | 客户端通过 `--policy_language_instruction` 传 |
| 服务器错误 → KeyError: 0 | 服务器返回 `{"error": "..."}` dict，leisaac 客户端 `call_endpoint` 未检查 error | `base.py` 增加 error dict 检查 |
| 端口占用 | 旧服务器未关，新实例 `Address already in use` | 先 kill 旧进程 |
