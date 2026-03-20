import torch, os, argparse, accelerate, warnings, json, csv, atexit, signal, multiprocessing, gc
from tqdm.auto import tqdm
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
from diffsynth.utils.data import save_video
from datetime import timedelta

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _cleanup_runtime():
    for child in multiprocessing.active_children():
        try:
            child.terminate()
        except Exception:
            pass
    for child in multiprocessing.active_children():
        try:
            child.join(timeout=1)
        except Exception:
            pass

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            torch.distributed.destroy_process_group()
        except Exception:
            pass

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    gc.collect()


def _signal_handler(signum, frame):
    raise SystemExit(f"Received signal {signum}")


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        super().__init__()
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True
        
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        audio_processor_config = self.parse_path_or_model_id(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        
    @staticmethod
    def fetch_video_from_data(data):
        video = data.get("video")
        if video is None:
            video = data.get("video_path")
        if video is None:
            raise KeyError("Cannot find video data. Expected key `video` or `video_path`.")
        return video

    @staticmethod
    def fetch_prompt_from_data(data):
        prompt = data.get("prompt")
        if prompt is None:
            prompt = data.get("caption")
        if prompt is None:
            raise KeyError("Cannot find text prompt. Expected key `prompt` or `caption`.")
        return prompt

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = self.fetch_video_from_data(data)[0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = self.fetch_video_from_data(data)[-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            elif extra_input == "camera_control_pose_file":
                if "camera_control_pose_file" in data:
                    inputs_shared["camera_control_pose_file"] = data["camera_control_pose_file"]
                elif "pose_file_aligned" in data:
                    inputs_shared["camera_control_pose_file"] = data["pose_file_aligned"]
                elif "pose_file_raw" in data:
                    inputs_shared["camera_control_pose_file"] = data["pose_file_raw"]
            elif extra_input == "camera_control_pose_width":
                if "camera_control_pose_width" in data:
                    inputs_shared["camera_control_pose_width"] = data["camera_control_pose_width"]
                elif "width" in data:
                    inputs_shared["camera_control_pose_width"] = data["width"]
            elif extra_input == "camera_control_pose_height":
                if "camera_control_pose_height" in data:
                    inputs_shared["camera_control_pose_height"] = data["camera_control_pose_height"]
                elif "height" in data:
                    inputs_shared["camera_control_pose_height"] = data["height"]
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        video = self.fetch_video_from_data(data)
        prompt = self.fetch_prompt_from_data(data)
        inputs_posi = {"prompt": prompt}
        inputs_nega = {}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": video,
            "height": video[0].size[1],
            "width": video[0].size[0],
            "num_frames": len(video),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _load_metadata_rows(metadata_path):
    if metadata_path is None:
        return []
    if metadata_path.endswith(".json"):
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    if metadata_path.endswith(".jsonl"):
        rows = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(metadata_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def _infer_pose_key(data_file_keys: str):
    keys = [k.strip() for k in data_file_keys.split(",") if k.strip()]
    for key in ("camera_control_pose_file", "pose_file_aligned", "pose_file_raw"):
        if key in keys:
            return key
    return "pose_file_aligned"


class WanValidationRunner:
    def __init__(self, args):
        self.args = args
        self.enabled = args.validation_metadata_path is not None and args.validation_num_samples > 0
        self.pose_key = args.validation_pose_key or _infer_pose_key(args.data_file_keys)
        self.rows = _load_metadata_rows(args.validation_metadata_path) if self.enabled else []
        self.sample_indices = list(range(min(len(self.rows), args.validation_num_samples))) if self.enabled else []
        self.dataset = None
        if self.enabled:
            self.dataset = UnifiedDataset(
                base_path=args.dataset_base_path,
                metadata_path=args.validation_metadata_path,
                repeat=1,
                data_file_keys=args.data_file_keys.split(","),
                main_data_operator=UnifiedDataset.default_video_operator(
                    base_path=args.dataset_base_path,
                    max_pixels=args.max_pixels,
                    height=args.height,
                    width=args.width,
                    height_division_factor=16,
                    width_division_factor=16,
                    num_frames=args.num_frames,
                    time_division_factor=4,
                    time_division_remainder=1,
                ),
                special_operator_map={
                    "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
                    "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
                    "camera_control_pose_file": ToAbsolutePath(args.dataset_base_path),
                    "pose_file_aligned": ToAbsolutePath(args.dataset_base_path),
                    "pose_file_raw": ToAbsolutePath(args.dataset_base_path),
                },
                dataset_read_retry_count=0,
                dataset_read_retry_sleep_seconds=0.0,
            )

    def run(self, pipe: WanVideoPipeline, step: int):
        if not self.enabled or self.dataset is None:
            return []

        output_dir = os.path.join(self.args.output_path, self.args.validation_output_subdir, f"step-{step}")
        os.makedirs(output_dir, exist_ok=True)

        results = []
        for sample_id, sample_idx in enumerate(self.sample_indices):
            data = self.dataset[sample_idx]
            try:
                prompt = data.get(self.args.validation_prompt_key, None)
                if prompt is None:
                    prompt = data.get("caption", data.get("prompt", ""))
                if prompt is None or str(prompt).strip() == "":
                    prompt = " "
                video = data.get("video", data.get("video_path"))
                if video is None or len(video) == 0:
                    raise ValueError("Validation sample contains empty video.")
                input_image = video[0]
                num_frames = len(video)
                pose_file = data.get("camera_control_pose_file")
                if pose_file is None:
                    pose_file = data.get(self.pose_key, data.get("pose_file_aligned", data.get("pose_file_raw")))
                pose_width = int(float(data.get("width", self.args.width or 1280)))
                pose_height = int(float(data.get("height", self.args.height or 720)))
                with torch.inference_mode():
                    generated = pipe(
                        prompt=prompt,
                        negative_prompt=self.args.validation_negative_prompt,
                        input_image=input_image,
                        camera_control_pose_file=pose_file,
                        camera_control_pose_width=pose_width,
                        camera_control_pose_height=pose_height,
                        height=self.args.height,
                        width=self.args.width,
                        num_frames=num_frames,
                        seed=self.args.validation_seed + sample_id,
                        num_inference_steps=self.args.validation_inference_steps,
                        tiled=self.args.validation_tiled,
                        progress_bar_cmd=tqdm,
                    )
                save_path = os.path.join(output_dir, f"sample-{sample_id}.mp4")
                save_video(generated, save_path, fps=self.args.validation_fps, quality=5)
                video_rel_path = str(self.rows[sample_idx].get("video_path", ""))
                results.append({
                    "sample_id": sample_id,
                    "video_path": save_path,
                    "video_rel_path": video_rel_path,
                    "prompt": prompt,
                })
            except Exception as e:
                results.append({
                    "sample_id": sample_id,
                    "error": str(e),
                })
        return results


class WandbLogger:
    def __init__(self, args):
        self.enabled = args.wandb_project is not None and args.wandb_project != ""
        self.wandb = None
        if not self.enabled:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError("`wandb` is required when --wandb_project is set. Please run `pip install wandb`.") from e
        self.wandb = wandb
        init_kwargs = {
            "project": args.wandb_project,
            "name": args.wandb_run_name,
            "entity": args.wandb_entity,
            "mode": args.wandb_mode,
            "config": vars(args),
        }
        if args.wandb_tags is not None and args.wandb_tags.strip() != "":
            init_kwargs["tags"] = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
        self.wandb.init(**init_kwargs)

    def log(self, data, step=None):
        if self.enabled:
            self.wandb.log(data, step=step)

    def video(self, path, fps=15):
        if not self.enabled:
            return None
        return self.wandb.Video(path, fps=fps, format="mp4")

    def finish(self):
        if self.enabled:
            self.wandb.finish()


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--wandb_project", type=str, default=None, help="Weights & Biases project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity/team.")
    parser.add_argument("--wandb_mode", type=str, default="online", help="Weights & Biases mode: online/offline/disabled.")
    parser.add_argument("--wandb_tags", type=str, default=None, help="Comma-separated tags for Weights & Biases.")
    parser.add_argument("--wandb_log_steps", type=int, default=1, help="Log scalar metrics every N steps.")
    parser.add_argument("--validation_metadata_path", type=str, default=None, help="Validation metadata path. Inference runs at each save step.")
    parser.add_argument("--validation_num_samples", type=int, default=1, help="Number of validation samples to run per validation trigger.")
    parser.add_argument("--validation_inference_steps", type=int, default=30, help="Inference steps used during validation generation.")
    parser.add_argument("--validation_seed", type=int, default=0, help="Base seed used for validation generation.")
    parser.add_argument("--validation_fps", type=int, default=15, help="FPS used when saving validation videos.")
    parser.add_argument("--validation_negative_prompt", type=str, default="", help="Negative prompt used for validation generation.")
    parser.add_argument("--validation_prompt_key", type=str, default="caption", help="Metadata key to use as prompt on validation samples.")
    parser.add_argument("--validation_pose_key", type=str, default=None, help="Metadata key for pose file on validation samples. Default inferred from data_file_keys.")
    parser.add_argument("--validation_output_subdir", type=str, default="validation", help="Subfolder under output_path for validation videos.")
    parser.add_argument("--validation_tiled", type=_str2bool, default=True, help="Enable tiled VAE during validation inference.")
    parser.add_argument("--dist_timeout_minutes", type=int, default=30, help="Timeout in minutes for distributed synchronization operations. Increase this if you encounter timeout errors during distributed training.")

    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    atexit.register(_cleanup_runtime)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    ddp_kwargs = accelerate.DistributedDataParallelKwargs(
        find_unused_parameters=args.find_unused_parameters
    )
    pg_kwargs = accelerate.InitProcessGroupKwargs(
        timeout=timedelta(minutes=args.dist_timeout_minutes)
    )   
    # accelerator = accelerate.Accelerator(
    #     gradient_accumulation_steps=args.gradient_accumulation_steps,
    #     kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    # )
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs, pg_kwargs],
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
            "camera_control_pose_file": ToAbsolutePath(args.dataset_base_path),
            "pose_file_aligned": ToAbsolutePath(args.dataset_base_path),
            "pose_file_raw": ToAbsolutePath(args.dataset_base_path),
        },
        dataset_min_num_frames=args.dataset_min_num_frames,
        dataset_num_frames_key=args.dataset_num_frames_key,
        dataset_drop_missing_num_frames=args.dataset_drop_missing_num_frames,
        dataset_read_retry_count=args.dataset_read_retry_count,
        dataset_read_retry_sleep_seconds=args.dataset_read_retry_sleep_seconds,
    )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    wandb_logger = WandbLogger(args) if accelerator.is_main_process else None
    validation_runner = WanValidationRunner(args) if accelerator.is_main_process else None

    def _on_log_step(step, metrics):
        if not accelerator.is_main_process or wandb_logger is None:
            return
        wandb_logger.log(metrics, step=step)

    def _on_checkpoint_saved(step, checkpoint_path, accelerator, model):
        if not accelerator.is_main_process or validation_runner is None:
            return
        if not validation_runner.enabled:
            return
        unwrapped = accelerator.unwrap_model(model)
        try:
            results = validation_runner.run(unwrapped.pipe, step=step)
        finally:
            # Validation uses inference path and flips scheduler.training=False.
            # Restore training mode before next optimization step.
            unwrapped.pipe.scheduler.set_timesteps(1000, training=True)
        if wandb_logger is not None:
            logs = {
                "train/checkpoint_step": step,
                "train/checkpoint_path": checkpoint_path,
                "validation/num_samples": len(results),
                "validation/num_success": sum(1 for item in results if "video_path" in item),
            }
            for item in results:
                sample_id = item["sample_id"]
                if "video_path" in item:
                    logs[f"validation/video_{sample_id}"] = wandb_logger.video(item["video_path"], fps=args.validation_fps)
                    logs[f"validation/prompt_{sample_id}"] = item["prompt"]
                    logs[f"validation/source_{sample_id}"] = item["video_rel_path"]
                else:
                    logs[f"validation/error_{sample_id}"] = item["error"]
            wandb_logger.log(logs, step=step)

    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    try:
        if args.task.endswith(":data_process"):
            launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
        else:
            launcher_map[args.task](
                accelerator, dataset, model, model_logger, args=args,
                on_log_step=_on_log_step,
                on_checkpoint_saved=_on_checkpoint_saved,
            )
    finally:
        if accelerator.is_main_process and wandb_logger is not None:
            wandb_logger.finish()
        _cleanup_runtime()
