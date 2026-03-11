# Wan2.1 SpeedControl 训练代码分析

本文基于仓库现有代码，梳理 `Wan2.1-1.3b-speedcontrol-v1` 的训练数据要求与训练流程。重点回答两个问题：

1. 训练时数据应当长什么样。
2. `motion_bucket_id` 是如何真正影响模型的。


## 1. 入口脚本与两种训练方式

### 1.1 LoRA 训练脚本

脚本：`examples/wanvideo/model_training/lora/Wan2.1-1.3b-speedcontrol-v1.sh`

关键参数：

* `--lora_base_model "dit"`：在 DiT 上打 LoRA。
* `--remove_prefix_in_ckpt "pipe.dit."`：导出时保存 DiT 相关权重（LoRA）。
* `--extra_inputs "motion_bucket_id"`：告诉训练流程从数据里额外读取 `motion_bucket_id`。

结论：LoRA 路线主要训练 DiT（LoRA 参数），速度控制条件依然参与前向。


### 1.2 Full 训练脚本

脚本：`examples/wanvideo/model_training/full/Wan2.1-1.3b-speedcontrol-v1.sh`

关键参数：

* `--trainable_models "motion_controller"`：只训练运动控制器。
* `--remove_prefix_in_ckpt "pipe.motion_controller."`：导出时仅保存运动控制器权重。
* `--extra_inputs "motion_bucket_id"`：同样需要运动桶标签。

结论：full 路线是“只训 motion controller”的专门训练。

### 1.3 Full 与 LoRA 的关系

它们默认是两条**独立**路线，不是脚本内自动串联的两阶段流程：

1. `full`：训练 `motion_controller`。
2. `lora`：训练 `dit` 的 LoRA（速度条件仍参与前向，但 `motion_controller` 默认冻结）。

如果你希望做“两阶段”（先 full 再 LoRA），可以手动实现，但需要在第二阶段显式加载第一阶段得到的 `motion_controller` 权重；脚本不会自动接续。


## 2. 训练数据格式（由代码推导）

训练入口统一是 `examples/wanvideo/model_training/train.py`。该脚本对字段的读取逻辑如下：

* 视频字段：优先 `video`，否则 `video_path`。
* 文本字段：优先 `prompt`，否则 `caption`。
* 额外字段：当 `--extra_inputs` 含 `motion_bucket_id` 时，样本中必须有该键。

因此，SpeedControl 最小可用样本应包含：

* `video` 或 `video_path`
* `prompt` 或 `caption`
* `motion_bucket_id`


### 2.1 与示例脚本默认值相关的注意事项

`diffsynth/diffusion/parsers.py` 中 `--data_file_keys` 默认是 `image,video`。  
SpeedControl 示例脚本没有显式覆盖 `--data_file_keys`，这意味着它默认假设元数据列名是 `video`。

如果你的元数据用的是 `video_path`，建议显式加上：

```bash
--data_file_keys "video_path"
```

并保证 `prompt/caption` 与 `motion_bucket_id` 在元数据里存在。


### 2.2 推荐的 metadata 结构示例

CSV 示例（推荐）：

```csv
video,prompt,motion_bucket_id
clips/a.mp4,"a dog running on grass",10
clips/b.mp4,"a city timelapse at sunset",80
```

JSON 示例（可用）：

```json
[
  {
    "video_path": "clips/a.mp4",
    "caption": "a dog running on grass",
    "motion_bucket_id": 10
  }
]
```


## 3. 训练流程（代码路径）

### 3.1 数据加载

`train.py` 构造 `UnifiedDataset`：

* 用 `metadata_path` 读 CSV/JSON。
* 对 `data_file_keys` 中的视频路径执行 `default_video_operator`，解码并采样帧。
* 非文件字段（如 `prompt/caption`、`motion_bucket_id`）保留为普通值。


### 3.2 组装 Pipeline 输入

`WanTrainingModule.get_pipeline_inputs` 做三件事：

1. 取视频和文本（`video/video_path`、`prompt/caption`）。
2. 写入公共输入：`input_video/height/width/num_frames/cfg_scale/...`。
3. 调用 `parse_extra_inputs`，把 `motion_bucket_id` 放进 `inputs_shared`。


### 3.3 Unit 前处理

`WanVideoPipeline` 的 units 中包含 `WanVideoUnit_SpeedControl`，它会把标量 `motion_bucket_id` 转为张量，供后续 `model_fn` 使用。


### 3.4 损失与反向传播

`FlowMatchSFTLoss` 中：

1. 随机采样一个时间步 `t`。
2. 对视频 latent 加噪，得到训练输入。
3. 调 `model_fn_wan_video(...)` 预测噪声。
4. 与目标噪声做 MSE，回传梯度。


## 4. `motion_bucket_id` 如何影响生成

核心注入点在 `model_fn_wan_video`：

* 先由时间步构造 `t_mod`。
* 若 `motion_bucket_id` 与 `motion_controller` 都存在，则执行：
  * `t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))`

这表示 `motion_bucket_id` 不是后处理参数，而是直接进入 DiT 的时间调制分支，影响去噪过程。

`WanMotionControllerModel` 内部将 `motion_bucket_id * 10` 做正弦嵌入，再通过 MLP 输出调制向量。代码中没有显式 clamp，因此标签范围由你的数据设计决定；示例推理常用 `0/50/100` 作为慢/中/快参考。


## 5. 对自定义数据集的落地建议

1. 先保证 metadata 至少有：`video(或video_path)`、`prompt(或caption)`、`motion_bucket_id`。
2. 若你用 `video_path` 列名，训练命令里显式设置 `--data_file_keys "video_path"`。
3. `motion_bucket_id` 建议用离散整数，先从小范围开始（例如 `0~100`）。
4. 如果目的是训练“纯速度控制器”，用 full 脚本；如果想在现有速度控制模型上适配新域内容，用 LoRA 脚本。


## 6. 一句话总结

SpeedControl 训练本质是：给每个视频样本附一个 `motion_bucket_id` 标签，训练时将该标签通过 `motion_controller` 注入 DiT 的时间调制，从而学习“运动幅度/速度”条件控制。


## 7. Motion Controller 结构（代码级）

`WanMotionControllerModel` 定义在 `diffsynth/models/wan_video_motion_controller.py`，结构非常轻量：

1. 输入：`motion_bucket_id`（标量或 batch 标量）。
2. 先做正弦位置编码：`sinusoidal_embedding_1d(freq_dim, motion_bucket_id * 10)`。
3. 再过一个 3 层 MLP：
   * `Linear(freq_dim, dim)` + `SiLU`
   * `Linear(dim, dim)` + `SiLU`
   * `Linear(dim, dim * 6)`
4. 输出：长度为 `6 * dim` 的向量。

对于 `Wan2.1-T2V-1.3B`，`dim=1536`，所以 motion controller 输出维度是 `9216`，随后 reshape 成 `(6, 1536)`。

为什么是 `6`：因为它对齐了 DiT block 内的 6 组调制量：

* `shift_msa`
* `scale_msa`
* `gate_msa`
* `shift_mlp`
* `scale_mlp`
* `gate_mlp`


## 8. 它如何注入到 Wan 模型

注入发生在 `model_fn_wan_video`：

1. 先由 diffusion timestep 生成基础 `t_mod`（形状 `(B, 6, dim)`）。
2. 再把 motion controller 输出加到 `t_mod` 上：
   * `t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))`
3. `t_mod` 传给每一个 DiT block。
4. 每个 block 内用 `t_mod` 生成 `shift/scale/gate`，调制 self-attention 和 FFN。

这意味着 speedcontrol 不是在最后阶段“调一点强度”，而是贯穿整个去噪网络的层内调制。

### 8.1 不是 token 维拼接，而是调制向量相加

常见误解是“把 speed control 和 timestep 沿 token 维拼接”。当前实现并不是这样：

1. 先得到时间条件 `t_mod_time`。
2. 再得到速度条件 `t_mod_speed`。
3. 直接执行加法：`t_mod = t_mod_time + t_mod_speed`。

因此 speed control 走的是与 timestep 同一条 AdaLN 调制通路，但融合方式是**加法**，不是 token 拼接。

### 8.2 从标量到张量的维度流（1.3B）

以 batch=1 为例，`motion_bucket_id` 的形状变化：

1. 标量 `motion_bucket_id`（来自 metadata）。
2. `WanVideoUnit_SpeedControl` 转成 tensor：`(1,)`。
3. 正弦编码后：`(1, 256)`（`freq_dim=256`）。
4. 经过 motion controller MLP：`(1, 6*1536) = (1, 9216)`。
5. `unflatten` 后：`(1, 6, 1536)`，与 `t_mod_time` 对齐并相加。

补充一点：`dit.head` 使用的是 `t`（时间嵌入），不是 `t_mod`；但由于前面所有 transformer block 都被 `t_mod` 调制，最终输出仍会显著受 `motion_bucket_id` 影响。


## 9. speedcontrol 信号是如何“产生”的

从当前仓库代码看，`motion_bucket_id` 是一个**外部提供**的监督标签：

* 训练代码不会在线从视频计算它。
* `WanVideoUnit_SpeedControl` 只做类型转换（标量 -> tensor）。
* 数据集中必须提前写好 `motion_bucket_id` 列（CSV/JSON 字段）。

仓库里没有 `motion_bucket_id` 自动生成脚本（至少当前代码树未提供），所以其生成流程应是离线预处理完成的。

### 9.1 从代码可确定的语义

可以确定的是：

* `motion_bucket_id` 控制的是“运动强度/速度幅度”的全局条件（文档中也写了值越大运动幅度越大）。
* 它不直接提供运动方向、轨迹或对象级控制。
* 它更像一个“全局运动等级”条件，作用于全视频去噪过程。
* 它不直接改变输出视频的 `fps` 或 `num_frames`；这些由推理参数/导出参数决定。`motion_bucket_id` 控制的是“相邻生成帧之间的运动幅度”。


## 10. 若你要自建 `motion_bucket_id`（工程建议）

虽然仓库未给出官方生成器，但可按下面策略构造，和当前架构是匹配的：

1. 对每段训练视频计算帧间运动强度（如光流幅值均值、或帧间差分统计）。
2. 对全数据做归一化（例如按分位数）。
3. 量化成离散 bucket（如 `0~100`）。
4. 写入 metadata 的 `motion_bucket_id` 列。

这套做法的关键是保持标签与“可感知运动幅度”单调相关，因为 motion controller 学到的是一个连续调制函数，而不是离散类别 one-hot。


## 11. 常见问题（来自讨论）

### 11.1 这里有没有给 speedcontrol 训练一个 adapter？

没有单独的 adapter。  
speedcontrol 的条件编码器就是 `WanMotionControllerModel`（小型 MLP），并通过 `t_mod` 注入 DiT。

### 11.2 为什么是 6 组调制参数？

因为每个 DiT block 里有两条被条件调制的主分支，每条分支各需要 `shift/scale/gate` 三组参数：

1. self-attention 分支：`shift_msa`, `scale_msa`, `gate_msa`
2. FFN 分支：`shift_mlp`, `scale_mlp`, `gate_mlp`

总计 `2 x 3 = 6` 组。  
另外，当前实现中的 cross-attention 分支不使用这 6 组 `t_mod` 参数。

### 11.3 这些 6 组是 self-attn 和 FFN 的 LN 调制参数吗？

是。  
它们用于 `norm1`（self-attn 前）和 `norm2`（FFN 前）的 AdaLN 风格调制，同时用 gate 控制对应残差分支注入强度。

### 11.4 仓库有没有推理版本？

有，示例在：

1. `examples/wanvideo/model_inference/`
2. `examples/wanvideo/model_inference_low_vram/`

speedcontrol 对应脚本是 `examples/wanvideo/model_inference/Wan2.1-1.3b-speedcontrol-v1.py`。
