# Wan14B 相机控制模型（RealCam-Vid）训练与推理代码梳理

本文基于当前仓库代码，面向目标：

- 使用 `RealCam-Vid` 数据集，进行 `PAI/Wan2.1-Fun-V1.1-14B-Control-Camera` 的微调。
- 覆盖两条训练路线：全量微调（full）与 LoRA。
- 给出推理/验证代码入口，以及关键接口的输入输出。


## 1. 相机控制相关代码总览

### 1.1 Wan 相机控制模型脚本（全）

- 推理：
  - `examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-14B-Control-Camera.py`
  - `examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-1.3B-Control-Camera.py`
  - `examples/wanvideo/model_inference/Wan2.2-Fun-A14B-Control-Camera.py`
- 低显存推理：
  - `examples/wanvideo/model_inference_low_vram/Wan2.1-Fun-V1.1-14B-Control-Camera.py`
  - `examples/wanvideo/model_inference_low_vram/Wan2.1-Fun-V1.1-1.3B-Control-Camera.py`
  - `examples/wanvideo/model_inference_low_vram/Wan2.2-Fun-A14B-Control-Camera.py`
- 全量训练脚本：
  - `examples/wanvideo/model_training/full/Wan2.1-Fun-V1.1-14B-Control-Camera.sh`
  - `examples/wanvideo/model_training/full/Wan2.1-Fun-V1.1-1.3B-Control-Camera.sh`
  - `examples/wanvideo/model_training/full/Wan2.2-Fun-A14B-Control-Camera.sh`
- LoRA 训练脚本：
  - `examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-14B-Control-Camera.sh`
  - `examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-14B-Control-Camera-RealPose.sh`
  - `examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-14B-Control-Camera-RealPose.sbatch`
  - `examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-1.3B-Control-Camera.sh`
  - `examples/wanvideo/model_training/lora/Wan2.2-Fun-A14B-Control-Camera.sh`
- 训练后验证：
  - full：`examples/wanvideo/model_training/validate_full/Wan2.1-Fun-V1.1-14B-Control-Camera.py`
  - lora：`examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-V1.1-14B-Control-Camera.py`

### 1.2 核心实现文件

- 统一训练入口：`examples/wanvideo/model_training/train.py`
- Wan Pipeline：`diffsynth/pipelines/wan_video.py`
- 相机位姿处理：`diffsynth/models/wan_video_camera_controller.py`
- 训练框架（LoRA 注入/冻结策略）：`diffsynth/diffusion/training_module.py`
- 通用 LoRA 加载接口：`diffsynth/diffusion/base_pipeline.py`
- 通用数据集：`diffsynth/core/data/unified_dataset.py`
- 参数解析：`diffsynth/diffusion/parsers.py`


## 2. 端到端流程（按你关注的深度学习主线）

### 2.1 数据加载

#### 入口
`examples/wanvideo/model_training/train.py`

训练时构造 `UnifiedDataset`：

```python
# train.py
dataset = UnifiedDataset(
    base_path=args.dataset_base_path,
    metadata_path=args.dataset_metadata_path,
    data_file_keys=args.data_file_keys.split(","),
    main_data_operator=UnifiedDataset.default_video_operator(...),
    special_operator_map={
        "camera_control_pose_file": ToAbsolutePath(args.dataset_base_path),
        "pose_file_aligned": ToAbsolutePath(args.dataset_base_path),
        "pose_file_raw": ToAbsolutePath(args.dataset_base_path),
    }
)
```

#### 关键接口与 I/O

1. `UnifiedDataset(metadata_path=...)`
- 输入：`csv/json/jsonl` 元数据。
- 输出：每次 `__getitem__` 返回一条样本 `dict`。

2. `data_file_keys`
- 只会对这些 key 执行文件加载/路径处理。
- 默认值在 `parsers.py` 是 `image,video`，对于 RealCam-Vid 必须覆盖。

3. `WanTrainingModule` 内字段读取逻辑（`train.py`）
- 视频字段：优先 `video`，否则 `video_path`
- 文本字段：优先 `prompt`，否则 `caption`

#### RealCam-Vid 推荐字段

基于 `Wan2.1-Fun-V1.1-14B-Control-Camera-RealPose.sh` 与 `validate_lora` 脚本，推荐元数据包含：

- `video_path`
- `caption`
- `pose_file_aligned`（或 `pose_file_raw`）
- `width`
- `height`
- 可选：`video_id`

训练命令里对应：

```bash
--data_file_keys "video_path,pose_file_aligned"
--extra_inputs "input_image,camera_control_pose_file,camera_control_pose_width,camera_control_pose_height"
```

`camera_control_pose_file` 在代码中可自动从 `pose_file_aligned` / `pose_file_raw` 回退映射，不要求你必须把元数据列改名。


### 2.2 模型权重加载

#### 入口

- 训练侧：`WanTrainingModule.__init__ -> parse_model_configs -> WanVideoPipeline.from_pretrained`
- 推理侧：各 `model_inference/*.py` 直接调用 `WanVideoPipeline.from_pretrained`

#### 关键接口

1. `ModelConfig`
- 输入：`path` 或 `model_id + origin_file_pattern`
- 输出：可被 `download_and_load_models` 解析的配置对象

2. `--model_id_with_origin_paths`
- 典型 14B camera：
  - `diffusion_pytorch_model*.safetensors`（DiT）
  - `models_t5_umt5-xxl-enc-bf16.pth`（T5）
  - `Wan2.1_VAE.pth`（VAE）
  - `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`（CLIP）

3. 公共文件重定向（`wan_video.py`）
- `models_t5...pth`、`Wan2.1_VAE.pth`、`models_clip...pth` 会被重定向到 `DiffSynth-Studio/Wan-Series-Converted-Safetensors`。


### 2.3 模型初始化

#### 入口
`WanVideoPipeline.from_pretrained(...)`

#### 关键初始化结果

- `pipe.dit`（Wan DiT）
- `pipe.text_encoder`
- `pipe.vae`
- `pipe.image_encoder`
- 其他可选模块（motion_controller/vace/vap/...）

`Wan2.2-Fun-A14B-Control-Camera` 会加载两套 DiT：
- `pipe.dit`（high_noise）
- `pipe.dit2`（low_noise）

推理中按 `switch_DiT_boundary` 切换。

#### 训练态切换
`DiffusionTrainingModule.switch_pipe_to_training_mode(...)`

- `pipe.freeze_except(...)`：冻结非训练模块。
- full 训练：`--trainable_models "dit"`。
- LoRA 训练：`--lora_base_model "dit"`（无需 `trainable_models` 也可）。


### 2.4 Camera Control 条件注入

核心类：`WanVideoUnit_FunCameraControl`（`wan_video.py`）

它支持两种控制源：

1. 方向+速度：
- `camera_control_direction`
- `camera_control_speed`

2. 真实位姿（推荐 RealCam-Vid）：
- `camera_control_pose_file` 或 `camera_control_poses`
- `camera_control_pose_width`
- `camera_control_pose_height`

关键过程：

```python
pose_entries = self._load_pose_entries(...)
if pose_entries is not None:
    camera_control_plucker_embedding = process_pose_file(...)
elif camera_control_direction is not None:
    camera_control_plucker_embedding = pipe.dit.control_adapter.process_camera_coordinates(...)

x = dit.patchify(x, control_camera_latents_input)
```

#### I/O（关键）

`WanVideoUnit_FunCameraControl.process(...)` 输出：

- `control_camera_latents_input`: 相机控制 latent（给 `dit.patchify`）
- `y`: 首帧条件 latent（保持 I2V 条件）

`model_fn_wan_video(...)` 输入包含：

- `latents`
- `context`
- `y`
- `control_camera_latents_input`

输出：

- `noise_pred`（用于扩散损失）


### 2.5 损失与训练输出

`FlowMatchSFTLoss(pipe, **inputs)`：

- 输入：`input_latents` + 条件（文本、相机控制等）
- 输出：`torch.Tensor` 标量 loss

模型保存：

- `ModelLogger` 保存 `epoch-{k}.safetensors` 或 `step-{k}.safetensors`
- `--remove_prefix_in_ckpt "pipe.dit."` 后，导出键可直接用于 `pipe.dit.load_state_dict(...)` 或 `pipe.load_lora(...)`


## 3. Full 微调 vs LoRA 微调（相机控制模型）

### 3.1 Full（14B camera）

脚本：`examples/wanvideo/model_training/full/Wan2.1-Fun-V1.1-14B-Control-Camera.sh`

关键参数：

- `--trainable_models "dit"`
- `--remove_prefix_in_ckpt "pipe.dit."`
- `--extra_inputs "input_image,camera_control_direction,camera_control_speed"`（示例数据）

对 RealCam-Vid（pose）建议改为：

- `--data_file_keys "video_path,pose_file_aligned"`
- `--extra_inputs "input_image,camera_control_pose_file,camera_control_pose_width,camera_control_pose_height"`

### 3.2 LoRA（14B camera）

基础脚本：`examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-14B-Control-Camera.sh`

RealCam-Vid 脚本：`examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-14B-Control-Camera-RealPose.sh`

关键参数：

- `--lora_base_model "dit"`
- `--lora_target_modules "q,k,v,o,ffn.0,ffn.2"`
- `--lora_rank 32`
- `--extra_inputs "input_image,camera_control_pose_file,camera_control_pose_width,camera_control_pose_height"`

### 3.3 LoRA 注入与加载细节

训练注入：`training_module.py`

```python
lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
model = inject_adapter_in_model(lora_config, model)
```

训练恢复（继续训）支持 `--lora_checkpoint`，内部会做 key 映射：
- `lora_A.weight -> lora_A.default.weight`
- `lora_B.weight -> lora_B.default.weight`

推理加载：`BasePipeline.load_lora`

```python
pipe.load_lora(pipe.dit, ".../epoch-4.safetensors", alpha=1)
```

Wan2.2 双 DiT 时需要分别加载 `dit` 与 `dit2`。


## 4. 推理与验证代码（相机控制）

### 4.1 基础推理（方向+速度）

- `examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-14B-Control-Camera.py`

核心调用：

```python
video = pipe(
    prompt=...,
    input_image=input_image,
    camera_control_direction="Left",
    camera_control_speed=0.01,
    seed=0,
    tiled=True,
)
```

### 4.2 用 RealCam-Vid 验证（位姿文件）

- `examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-V1.1-14B-Control-Camera.py`

该脚本已实现：

- 从 `annotations/validation.json` 读样本
- 自动读取 `pose_file_aligned` 或 `pose_file_raw`
- 若位姿缺失，回退 `camera_control_direction + speed`
- 脚本中 `pipe.load_lora(...)` 默认是注释状态，验证 LoRA 时需要手动取消注释并改成你的 checkpoint 路径。

这就是最接近你目标的数据驱动验证脚本。

### 4.3 full / lora 权重加载方式

- full 验证：
  - `state_dict = load_state_dict(".../epoch-k.safetensors")`
  - `pipe.dit.load_state_dict(state_dict)`
- lora 验证：
  - `pipe.load_lora(pipe.dit, ".../epoch-k.safetensors", alpha=1)`


## 5. 面向 RealCam-Vid 训练 WAN14B Camera LoRA 的落地建议

### 5.1 数据准备

你可以直接用 JSON，不必先转 CSV。推荐元数据样例（单条）：

```json
{
  "video_path": "RealEstate10K/train/xxx/yyy.mp4",
  "caption": "a camera moving left in a living room",
  "pose_file_aligned": "poses/train/xxx/yyy.txt",
  "width": 1280,
  "height": 720,
  "video_id": "yyy"
}
```

如果要转 CSV，可用：
- `examples/wanvideo/model_training/convert_realestate10k_metadata.py`

### 5.2 LoRA 训练推荐命令（14B + pose）

可直接用现成脚本：
- `examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-14B-Control-Camera-RealPose.sh`

等价核心参数：

```bash
accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path /path/to/RealCam-Vid \
  --dataset_metadata_path /path/to/RealCam-Vid/annotations/train.json \
  --data_file_keys "video_path,pose_file_aligned" \
  --height 480 --width 832 \
  --model_id_with_origin_paths "PAI/Wan2.1-Fun-V1.1-14B-Control-Camera:diffusion_pytorch_model*.safetensors,PAI/Wan2.1-Fun-V1.1-14B-Control-Camera:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.1-Fun-V1.1-14B-Control-Camera:Wan2.1_VAE.pth,PAI/Wan2.1-Fun-V1.1-14B-Control-Camera:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image,camera_control_pose_file,camera_control_pose_width,camera_control_pose_height"
```

### 5.3 Full 训练命令（14B + pose）

仓库没有现成 `RealPose full` 脚本，但可由 LoRA 脚本直接改：

- 删除 LoRA 参数
- 增加 `--trainable_models "dit"`


## 6. 关键接口清单（输入输出速查）

### 6.1 `WanVideoPipeline.from_pretrained(...)`

- 输入：`model_configs`, `tokenizer_config`, `audio_processor_config`, `torch_dtype`, `device`
- 输出：初始化完的 `WanVideoPipeline`

### 6.2 `WanVideoPipeline.__call__(...)`

和相机控制相关的关键输入：

- `prompt: str`
- `input_image: PIL.Image`
- 方向控制：`camera_control_direction`, `camera_control_speed`
- 位姿控制：`camera_control_pose_file | camera_control_poses`, `camera_control_pose_width`, `camera_control_pose_height`
- 其他：`height`, `width`, `num_frames`, `seed`, `tiled`

输出：

- `list[PIL.Image]` 或量化视频帧（由 `output_type` 决定）

### 6.3 `WanTrainingModule.get_pipeline_inputs(data)`

- 输入：单条样本 `dict`
- 输出：`(inputs_shared, inputs_posi, inputs_nega)`

其中 `inputs_shared` 至少含：

- `input_video`, `height`, `width`, `num_frames`
- 由 `extra_inputs` 注入的相机控制字段

### 6.4 `BasePipeline.load_lora(module, lora_config_or_path, alpha=1)`

- 输入：目标模块（常用 `pipe.dit`）、LoRA 文件、缩放系数
- 输出：无显式返回；模块被打补丁/融合


## 7. 当前与你日志相关的阻塞点（必须注意）

你的日志 `logs/wan21_14b_cam_lora_11952375.err` 显示：

- 报错位置：`process_pose_file -> ray_condition`
- 错误：`Expected all tensors to be on the same device, but found cuda and cpu`

触发原因（代码层面）：

- `process_pose_file` 中 `K`、`c2ws` 默认在 CPU 上构造；
- `ray_condition` 内部网格在 `device`（GPU）上构造；
- 运算时发生 CPU/GPU 混用。

在正式大规模训练前，建议先做一次小样本 dry-run 并确认该问题是否已在你当前分支修复。


## 8. 一句话结论

针对 `RealCam-Vid -> Wan2.1-Fun-V1.1-14B-Control-Camera`，仓库已经具备完整链路：

- 训练：`train.py` + `RealPose` LoRA 脚本（可直接用）
- 验证：`validate_lora`（已支持 `pose_file_aligned/raw`）
- 推理：`WanVideoPipeline`（支持方向速度和真实位姿两种相机控制）

你下一步主要是：统一数据字段 + 先跑通单卡小样本 + 再上多卡正式训练。
