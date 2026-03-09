import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-Control-Camera", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-Control-Camera", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-Control-Camera", origin_file_pattern="Wan2.1_VAE.pth"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-Control-Camera", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ],
)
pipe.load_lora(pipe.dit, "models/train/Wan2.1-Fun-V1.1-14B-Control-Camera_lora/epoch-4.safetensors", alpha=1)

video = VideoData("/scratch/p64096rw/ac3d/RealCam-Vid/RealEstate10K/test/gBsz_o7HY80/b13191b0346bee85.mp4", height=480, width=832)

# First and last frame to video
video = pipe(
    prompt="A serene residential backyard with a beige two-story house, stone accents, and a dark brown wooden deck. initially, the patio is adorned with a black metal dining set and four chairs, surrounded by a flagstone design, a small fire pit, and a variety of shrubs and young trees, all under a clear sky. two seconds later, the scene includes a playset, suggesting a family-oriented space. the natural stone tiles and the wooden deck with a staircase remain consistent, with the addition of a swing set and a well-manicured lawn, indicating meticulous care.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    input_image=video[0],
    camera_control_direction="Left", camera_control_speed=0.0,
    seed=0, tiled=True
)
save_video(video, "video_Wan2.1-Fun-V1.1-14B-Control-Camera.mp4", fps=15, quality=5)
