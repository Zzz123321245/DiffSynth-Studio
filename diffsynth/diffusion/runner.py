"""
该代码定义了两个函数 `launch_training_task` 和 `launch_data_process_task`，分别用于启动模型训练任务和数据处理任务。
这些函数使用了 `Accelerator` 来加速训练和数据处理过程，并且支持分布式训练。训练过程中会使用 `AdamW` 优化器和一个恒定学习率的调度器。
数据处理任务会将处理后的数据保存到指定的输出路径中。

这里主要完成的是trian_step和data_process_step的功能，分别对应训练和数据处理的步骤。
"""
import os, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


def _single_item_collate(batch):
    return batch[0]


def _build_dataloader(dataset, shuffle, num_workers):
    kwargs = {
        "dataset": dataset,
        "shuffle": shuffle,
        "collate_fn": _single_item_collate,
        "num_workers": num_workers,
        "persistent_workers": False,
    }
    if num_workers > 0:
        # Avoid forking CUDA context into workers, which can leave stale GPU handles.
        kwargs["multiprocessing_context"] = "spawn"
    return torch.utils.data.DataLoader(**kwargs)


def _shutdown_dataloader_workers(dataloader):
    if dataloader is None:
        return
    iterator = getattr(dataloader, "_iterator", None)
    if iterator is None:
        return
    shutdown_workers = getattr(iterator, "_shutdown_workers", None)
    if shutdown_workers is not None:
        try:
            shutdown_workers()
        except Exception:
            pass


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,  # 训练模块，包含模型和前向计算逻辑,是已经实例化的model
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
    on_log_step = None,
    on_checkpoint_saved = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    log_steps = max(getattr(args, "wandb_log_steps", 1), 1) if args is not None else 1

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    raw_dataloader = _build_dataloader(dataset, shuffle=True, num_workers=num_workers)
    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, raw_dataloader, scheduler)

    try:
        for epoch_id in range(num_epochs):
            for data in tqdm(dataloader):
                with accelerator.accumulate(model):
                    optimizer.zero_grad()
                    if dataset.load_from_cache:
                        loss = model({}, inputs=data)
                    else:
                        loss = model(data)
                    accelerator.backward(loss)
                    optimizer.step()
                    checkpoint_path = model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                    scheduler.step()
                    global_step = model_logger.num_steps
                    if on_log_step is not None and global_step % log_steps == 0:
                        loss_for_log = accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                        on_log_step(
                            step=global_step,
                            metrics={
                                "train/loss": loss_for_log,
                                "train/lr": scheduler.get_last_lr()[0],
                            },
                        )
                    if checkpoint_path is not None and on_checkpoint_saved is not None:
                        accelerator.wait_for_everyone()
                        on_checkpoint_saved(
                            step=global_step,
                            checkpoint_path=checkpoint_path,
                            accelerator=accelerator,
                            model=model,
                        )
                        accelerator.wait_for_everyone()
            if save_steps is None:
                model_logger.on_epoch_end(accelerator, model, epoch_id)
        model_logger.on_training_end(accelerator, model, save_steps)
    finally:
        _shutdown_dataloader_workers(raw_dataloader)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers

    raw_dataloader = _build_dataloader(dataset, shuffle=False, num_workers=num_workers)
    model.to(device=accelerator.device)
    model, dataloader = accelerator.prepare(model, raw_dataloader)

    try:
        for data_id, data in enumerate(tqdm(dataloader)):
            with accelerator.accumulate(model):
                with torch.no_grad():
                    folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                    os.makedirs(folder, exist_ok=True)
                    save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                    data = model(data)
                    torch.save(data, save_path)
    finally:
        _shutdown_dataloader_workers(raw_dataloader)
