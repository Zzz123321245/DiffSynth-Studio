import torch
import json
from pathlib import Path
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


DATASET_ROOT = Path("/scratch/p64096rw/ac3d/RealCam-Vid/")
ANNOTATION_FILE = DATASET_ROOT / "annotations" / "validation.json"
SAMPLE_INDEX = 0
NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-Control-Camera", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-Control-Camera", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-Control-Camera", origin_file_pattern="Wan2.1_VAE.pth"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-Control-Camera", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
)

with ANNOTATION_FILE.open("r", encoding="utf-8") as f:
    validation_samples = json.load(f)
if len(validation_samples) <= SAMPLE_INDEX:
    raise IndexError(f"Sample index {SAMPLE_INDEX} is out of range for {ANNOTATION_FILE} (size={len(validation_samples)}).")

sample = validation_samples[SAMPLE_INDEX]
video_path = DATASET_ROOT / sample["video_path"]
if not video_path.is_file():
    raise FileNotFoundError(f"Video file not found: {video_path}")

video_data = VideoData(video_file=str(video_path))
input_image = video_data[0]
prompt = sample["caption"]

camera_control_kwargs = {}
pose_file = None
# pose_file = sample.get("pose_file_aligned") or sample.get("pose_file_raw")
if pose_file is not None:
    pose_path = DATASET_ROOT / pose_file
    if pose_path.is_file():
        camera_control_kwargs["camera_control_pose_file"] = str(pose_path)
        camera_control_kwargs["camera_control_pose_width"] = int(sample.get("width", 1280))
        camera_control_kwargs["camera_control_pose_height"] = int(sample.get("height", 720))

if not camera_control_kwargs:
    camera_control_kwargs = {"camera_control_direction": "Left", "camera_control_speed": 0.01}

video = pipe(
    prompt=prompt,
    negative_prompt=NEGATIVE_PROMPT,
    seed=0, tiled=True,
    input_image=input_image,
    **camera_control_kwargs,
)
output_name = f"video_validation_{SAMPLE_INDEX}_{sample.get('video_id', 'sample')}_Wan2.1-Fun-V1.1-14B-Control-Camera.mp4"
save_video(video, output_name, fps=15, quality=5)

# video = pipe(
#     prompt="一艘小船正勇敢地乘风破浪前行。蔚蓝的大海波涛汹涌，白色的浪花拍打着船身，但小船毫不畏惧，坚定地驶向远方。阳光洒在水面上，闪烁着金色的光芒，为这壮丽的场景增添了一抹温暖。镜头拉近，可以看到船上的旗帜迎风飘扬，象征着不屈的精神与冒险的勇气。这段画面充满力量，激励人心，展现了面对挑战时的无畏与执着。",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     input_image=input_image,
#     camera_control_direction="Up", camera_control_speed=0.01,
# )
# save_video(video, "video_up_Wan2.1-Fun-V1.1-14B-Control-Camera.mp4", fps=15, quality=5)
