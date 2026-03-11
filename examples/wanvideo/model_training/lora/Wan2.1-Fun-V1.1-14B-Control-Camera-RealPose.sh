#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/scratch/p64096rw/ac3d/RealCam-Vid}"
SPLIT="${SPLIT:-train}"
POSE_KEY="${POSE_KEY:-pose_file_aligned}" # or pose_file_raw
OUTPUT_PATH="${OUTPUT_PATH:-./models/train/Wan2.1-Fun-V1.1-14B-Control-Camera_lora_realpose_${SPLIT}_${POSE_KEY}}"

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path "${DATASET_ROOT}" \
  --dataset_metadata_path "${DATASET_ROOT}/annotations/${SPLIT}.json" \
  --data_file_keys "video_path,${POSE_KEY}" \
  --height 480 \
  --width 832 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "PAI/Wan2.1-Fun-V1.1-14B-Control-Camera:diffusion_pytorch_model*.safetensors,PAI/Wan2.1-Fun-V1.1-14B-Control-Camera:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.1-Fun-V1.1-14B-Control-Camera:Wan2.1_VAE.pth,PAI/Wan2.1-Fun-V1.1-14B-Control-Camera:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-5 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image,camera_control_pose_file,camera_control_pose_width,camera_control_pose_height"
