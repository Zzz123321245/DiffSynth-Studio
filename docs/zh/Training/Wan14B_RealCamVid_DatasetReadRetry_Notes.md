# Wan14B RealCamVid 训练数据读取失败重试说明

更新时间：2026-03-20

## 背景

在多卡训练 `Wan2.1-Fun-V1.1-14B-Control-Camera-RealPose` 时，训练已经可以正常进入前向、反向和验证阶段，但会在少量样本上偶发触发 `imageio_ffmpeg` 的读取失败，典型报错如下：

- `OSError: Could not load meta information`
- 报错位置在 `DataLoader worker process` 内部
- 失败样本并不一定是稳定损坏的视频，可能是 NFS 抖动或 `imageio_ffmpeg` 的偶发初始化失败

原始行为是：只要某个样本在读取阶段抛异常，整个训练作业直接退出。

## 这次修改的目标

将训练集的数据读取改成“容错模式”：

- 某个样本读取失败时，不立刻终止训练
- 打印一条告警日志，记录失败样本索引和摘要
- 随机换一个样本继续尝试
- 如果连续多次失败，才最终抛错退出

这样可以把“单个偶发坏读”从致命错误降级为局部跳过。

## 修改文件

### 1. `diffsynth/core/data/unified_dataset.py`

新增了两个构造参数：

- `dataset_read_retry_count`
- `dataset_read_retry_sleep_seconds`

新增了三个内部方法：

- `_get_data_source_size()`
- `_get_sample_summary()`
- `_load_item_once()`

并重写了 `__getitem__()` 的读取流程：

1. 第一次先按原索引读取
2. 若失败，记录异常
3. 后续尝试从当前数据集里随机抽一个新索引重试
4. 每次失败打印 `[WARN] Failed to load dataset sample ...`
5. 超过最大重试次数后，抛出 `RuntimeError`，并附带 `attempted_indices`

注意：

- `load_from_cache=True` 的路径没有启用这个随机换样本逻辑
- 重试只用于训练数据集，不影响缓存数据读取

### 2. `diffsynth/diffusion/parsers.py`

在 dataset 基础参数里新增了两个 CLI 参数：

- `--dataset_read_retry_count`
- `--dataset_read_retry_sleep_seconds`

默认值：

- `dataset_read_retry_count = 8`
- `dataset_read_retry_sleep_seconds = 0.0`

这意味着如果训练脚本不显式传这两个参数，训练集默认会有 8 次替换样本重试能力。

### 3. `examples/wanvideo/model_training/train.py`

训练集构造时，新增传参：

- `dataset_read_retry_count=args.dataset_read_retry_count`
- `dataset_read_retry_sleep_seconds=args.dataset_read_retry_sleep_seconds`

验证集构造时，显式固定为：

- `dataset_read_retry_count=0`
- `dataset_read_retry_sleep_seconds=0.0`

这样设计的原因是：

- 训练阶段更关注鲁棒性，允许跳过个别读失败样本
- 验证阶段更关注可重复性，不希望验证样本被静默替换

## 当前行为总结

当前默认行为如下：

- 训练集：单样本读取失败后，最多额外尝试 8 次，且每次可换一个新样本
- 验证集：任何样本读取失败仍然直接报错

因此，训练中遇到偶发 `imageio` / `ffmpeg` 读失败时，通常不会再立刻把整个作业打死。

## 已知局限

### 1. 这不是“修复坏样本”，而是“绕过单点失败”

如果某一批时间段内底层存储整体不稳定，或者大量样本都读失败，训练仍然会在重试耗尽后退出。

### 2. 训练步数语义不会变化

跳过失败样本后，训练不会补回这个样本，只是继续往下训练。因此从数据分布角度看，属于“容错跳过”，不是“严格等价重读”。

### 3. 验证仍然是严格模式

如果验证集样本本身读失败，当前仍会直接报错。这是有意保留的行为，避免验证 quietly 换样本。

## 推荐的后续观察点

下次训练时，重点看日志里是否出现如下告警：

- `[WARN] Failed to load dataset sample idx=...`

如果有：

- 说明容错逻辑生效了
- 训练应该继续往后跑，而不是立即退出

如果最终仍退出：

- 查看 `attempted_indices=[...]`
- 说明一次失败不是单点偶发，而是连续多个样本或底层读取链路都不稳定

## 如果后续还要继续增强

可以考虑的方向：

1. 在验证集也加入“可选重试但不换样本”的模式
2. 将失败样本写入单独 blacklist 文件，供后续 metadata 过滤
3. 将读取失败计数同步到 wandb，长期观察数据稳定性
4. 在 `LoadVideo.get_reader()` 层面增加针对 `imageio.get_reader()` 的短暂重试

## 本次改动对应的关键位置

- `diffsynth/core/data/unified_dataset.py`
- `diffsynth/diffusion/parsers.py`
- `examples/wanvideo/model_training/train.py`

如果后续要继续接着改，建议优先从 `UnifiedDataset.__getitem__()` 看起，因为这次容错逻辑的核心就在这里。
