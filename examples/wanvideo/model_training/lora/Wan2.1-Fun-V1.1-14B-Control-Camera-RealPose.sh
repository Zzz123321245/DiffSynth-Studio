#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/scratch/p64096rw/ac3d/RealCam-Vid}"
SPLIT="${SPLIT:-train}"
POSE_KEY="${POSE_KEY:-pose_file_aligned}" # or pose_file_raw
OUTPUT_PATH="${OUTPUT_PATH:-./models/train/Wan2.1-Fun-V1.1-14B-Control-Camera_lora_realpose_${SPLIT}_${POSE_KEY}}"

TRAIN_METADATA_PATH="${TRAIN_METADATA_PATH:-models/train/Wan2.1-Fun-V1.1-14B-Control-Camera_lora_realpose_train_pose_file_aligned/metadata_train_filtered.json}"
SAVE_STEPS="${SAVE_STEPS:-500}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"

VALIDATION_METADATA_PATH="${VALIDATION_METADATA_PATH:-${DATASET_ROOT}/annotations/validation.json}"
VALIDATION_NUM_SAMPLES="${VALIDATION_NUM_SAMPLES:-1}"
VALIDATION_INFERENCE_STEPS="${VALIDATION_INFERENCE_STEPS:-30}"
VALIDATION_SEED="${VALIDATION_SEED:-0}"
VALIDATION_FPS="${VALIDATION_FPS:-15}"
VALIDATION_OUTPUT_SUBDIR="${VALIDATION_OUTPUT_SUBDIR:-validation}"

WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-Wan2.1-Fun-V1.1-14B-Control-Camera-RealPose}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_LOG_STEPS="${WANDB_LOG_STEPS:-1}"
# If you want to avoid exporting key each run, put your key below once.
# Example: WANDB_API_KEY_IN_SCRIPT="wandb_xxx"
WANDB_API_KEY_IN_SCRIPT="${WANDB_API_KEY_IN_SCRIPT:-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}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-}"

NUM_PROCESSES="${NUM_PROCESSES:-2}"
NUM_MACHINES="${NUM_MACHINES:-1}"

export WANDB_BASE_URL

if [[ -n "${WANDB_PROJECT}" ]]; then
  if [[ -z "${WANDB_API_KEY}" ]] && [[ -n "${WANDB_API_KEY_IN_SCRIPT}" ]]; then
    WANDB_API_KEY="${WANDB_API_KEY_IN_SCRIPT}"
  fi
  if [[ -z "${WANDB_API_KEY}" ]] && [[ -n "${WANDB_API_KEY_FILE}" ]] && [[ -f "${WANDB_API_KEY_FILE}" ]]; then
    WANDB_API_KEY="$(head -n 1 "${WANDB_API_KEY_FILE}" | tr -d '\r\n')"
  fi
  if [[ -n "${WANDB_API_KEY}" ]]; then
    export WANDB_API_KEY
  elif [[ "${WANDB_MODE}" == "online" ]]; then
    echo "[WARN] WANDB_PROJECT is set but WANDB_API_KEY is empty."
    echo "[WARN] Please run 'wandb login', or set WANDB_API_KEY / WANDB_API_KEY_FILE."
  fi
fi

ACCELERATE_ARGS=(
  --num_processes "${NUM_PROCESSES}"
  --num_machines "${NUM_MACHINES}"
)

TRAIN_ARGS=(
  --dataset_base_path "${DATASET_ROOT}"
  --dataset_metadata_path "${TRAIN_METADATA_PATH}"
  --data_file_keys "video_path,${POSE_KEY}"
  --height 480
  --width 832
  --dataset_repeat 1
  --save_steps "${SAVE_STEPS}"
  --model_id_with_origin_paths "PAI/Wan2.1-Fun-V1.1-14B-Control-Camera:diffusion_pytorch_model*.safetensors,PAI/Wan2.1-Fun-V1.1-14B-Control-Camera:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.1-Fun-V1.1-14B-Control-Camera:Wan2.1_VAE.pth,PAI/Wan2.1-Fun-V1.1-14B-Control-Camera:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
  --learning_rate 1e-5
  --num_epochs "${NUM_EPOCHS}"
  --remove_prefix_in_ckpt "pipe.dit."
  --output_path "${OUTPUT_PATH}"
  --lora_base_model "dit"
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2"
  --lora_rank 32
  --extra_inputs "input_image,camera_control_pose_file,camera_control_pose_width,camera_control_pose_height"
  --validation_metadata_path "${VALIDATION_METADATA_PATH}"
  --validation_num_samples "${VALIDATION_NUM_SAMPLES}"
  --validation_inference_steps "${VALIDATION_INFERENCE_STEPS}"
  --validation_seed "${VALIDATION_SEED}"
  --validation_fps "${VALIDATION_FPS}"
  --validation_output_subdir "${VALIDATION_OUTPUT_SUBDIR}"
  --validation_pose_key "${POSE_KEY}"
  --wandb_log_steps "${WANDB_LOG_STEPS}"
)

if [[ -n "${WANDB_PROJECT}" ]]; then
  TRAIN_ARGS+=(--wandb_project "${WANDB_PROJECT}")
  TRAIN_ARGS+=(--wandb_run_name "${WANDB_RUN_NAME}")
  TRAIN_ARGS+=(--wandb_mode "${WANDB_MODE}")
  if [[ -n "${WANDB_ENTITY}" ]]; then
    TRAIN_ARGS+=(--wandb_entity "${WANDB_ENTITY}")
  fi
fi

accelerate launch "${ACCELERATE_ARGS[@]}" examples/wanvideo/model_training/train.py "${TRAIN_ARGS[@]}"
