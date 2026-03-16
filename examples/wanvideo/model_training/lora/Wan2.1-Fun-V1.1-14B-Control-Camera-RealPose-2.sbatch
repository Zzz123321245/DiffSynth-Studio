#!/bin/bash --login
#SBATCH -J wan21_14b_cam_lora_2
#SBATCH -p gpuA
#SBATCH -G 2
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --mem 128G
#SBATCH -t 4-0
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module purge
module load libs/cuda/12.8.1
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/scratch/p64096rw/ac3d/RealCam-Vid}"
SPLIT="${SPLIT:-train}"
POSE_KEY="${POSE_KEY:-pose_file_aligned}" # or pose_file_raw
OUTPUT_PATH="${OUTPUT_PATH:-./models/train/Wan2.1-Fun-V1.1-14B-Control-Camera_lora_realpose_${SPLIT}_${POSE_KEY}}"

TRAIN_METADATA_PATH="${TRAIN_METADATA_PATH:-/mnt/iusers01/eee01/p64096rw/scratch/ac3d/realestate10k2/annotations/train.json}"
SAVE_STEPS="${SAVE_STEPS:-200}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"

# Dataset frame-count filter (disabled by default).
# Set ENABLE_MIN_FRAMES_FILTER=1 to keep only samples with num_frames >= MIN_FRAMES_THRESHOLD.
ENABLE_MIN_FRAMES_FILTER="${ENABLE_MIN_FRAMES_FILTER:-1}"
MIN_FRAMES_THRESHOLD="${MIN_FRAMES_THRESHOLD:-81}"
DATASET_NUM_FRAMES_KEY="${DATASET_NUM_FRAMES_KEY:-num_frames}"
DROP_MISSING_NUM_FRAMES="${DROP_MISSING_NUM_FRAMES:-1}"

VALIDATION_METADATA_PATH="${VALIDATION_METADATA_PATH:-${DATASET_ROOT}/annotations/validation.json}"
VALIDATION_NUM_SAMPLES="${VALIDATION_NUM_SAMPLES:-2}"
VALIDATION_INFERENCE_STEPS="${VALIDATION_INFERENCE_STEPS:-30}"
VALIDATION_SEED="${VALIDATION_SEED:-0}"
VALIDATION_FPS="${VALIDATION_FPS:-15}"
VALIDATION_OUTPUT_SUBDIR="${VALIDATION_OUTPUT_SUBDIR:-validation}"

# We use Weights & Biases (wandb) for experiment tracking and visualization.
WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
WANDB_PROJECT="${WANDB_PROJECT:- WAN14BControlCameraRealPose}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-Wan2.1-Fun-V1.1-14B-Control-Camera-RealPose}"
WANDB_ENTITY="${WANDB_ENTITY:-runzewang2023-university-of-manchester}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_LOG_STEPS="${WANDB_LOG_STEPS:-1}"
# If you want to avoid exporting key each run, put your key below once.
# Example: WANDB_API_KEY_IN_SCRIPT="wandb_xxx"
WANDB_API_KEY_IN_SCRIPT="${WANDB_API_KEY_IN_SCRIPT:-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}"
WANDB_API_KEY="${WANDB_API_KEY:-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}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-}"

NUM_PROCESSES="${NUM_PROCESSES:-}"
NUM_MACHINES="${NUM_MACHINES:-1}"

# Distributed debug logs (enabled by default).
# Set DEBUG_LOGS=0 to disable.
DEBUG_LOGS="${DEBUG_LOGS:-0}"
TORCH_DISTRIBUTED_DEBUG_LEVEL="${TORCH_DISTRIBUTED_DEBUG_LEVEL:-DETAIL}"
NCCL_DEBUG_LEVEL="${NCCL_DEBUG_LEVEL:-WARN}"
NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,COLL}"

if [[ "${DEBUG_LOGS}" == "1" ]]; then
  export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG_LEVEL}"
  export NCCL_DEBUG="${NCCL_DEBUG_LEVEL}"
  # export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS}"
  export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"
  export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
fi

export WANDB_BASE_URL

# Triton autotune cache on NFS can cause slowdown/hang at exit.
# Prefer node-local storage when available.
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SLURM_TMPDIR}/triton_autotune}"
else
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER}/triton_autotune}"
fi
mkdir -p "${TRITON_CACHE_DIR}"

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
  --dataset_repeat 5
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
  --dataset_num_workers 1
  --lora_checkpoint "models/train/Wan2.1-Fun-V1.1-14B-Control-Camera_lora_realpose_train_pose_file_aligned/step-600.safetensors"
  --dist_timeout_minutes 30
  --gradient_accumulation_steps 2

)

if [[ "${ENABLE_MIN_FRAMES_FILTER}" == "1" ]]; then
  TRAIN_ARGS+=(--dataset_min_num_frames "${MIN_FRAMES_THRESHOLD}")
  TRAIN_ARGS+=(--dataset_num_frames_key "${DATASET_NUM_FRAMES_KEY}")
  if [[ "${DROP_MISSING_NUM_FRAMES}" == "1" ]]; then
    TRAIN_ARGS+=(--dataset_drop_missing_num_frames)
  fi
fi

if [[ -n "${WANDB_PROJECT}" ]]; then
  TRAIN_ARGS+=(--wandb_project "${WANDB_PROJECT}")
  TRAIN_ARGS+=(--wandb_run_name "${WANDB_RUN_NAME}")
  TRAIN_ARGS+=(--wandb_mode "${WANDB_MODE}")
  if [[ -n "${WANDB_ENTITY}" ]]; then
    TRAIN_ARGS+=(--wandb_entity "${WANDB_ENTITY}")
  fi
fi

accelerate launch "${ACCELERATE_ARGS[@]}" examples/wanvideo/model_training/train.py "${TRAIN_ARGS[@]}"
