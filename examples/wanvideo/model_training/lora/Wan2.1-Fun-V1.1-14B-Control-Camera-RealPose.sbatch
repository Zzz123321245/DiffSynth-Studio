#!/bin/bash --login
#SBATCH -J wan21_14b_cam_lora
#SBATCH -p gpuA
#SBATCH -G 2
#SBATCH -n 8
#SBATCH --mem 256G
#SBATCH -t 2-0
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

echo "SubmitDir=$SLURM_SUBMIT_DIR"
echo "WorkDir(pwd)=$(pwd)"

module purge
module load libs/cuda/12.8.1

# Activate conda env
source /mnt/iusers01/eee01/p64096rw/scratch/software/miniconda3/bin/activate
conda activate /mnt/iusers01/eee01/p64096rw/scratch/software/miniconda3/envs/diffsynth

echo "JobID=$SLURM_JOB_ID GPUs=$SLURM_GPUS CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} CPUs=$SLURM_NTASKS"

# -----------------------------
# Runtime config (override with: sbatch --export=ALL,VAR=... script.sbatch)
# -----------------------------
DATASET_ROOT="${DATASET_ROOT:-/scratch/p64096rw/ac3d/RealCam-Vid}"
SPLIT="${SPLIT:-train}"
POSE_KEY="${POSE_KEY:-pose_file_aligned}"   # or pose_file_raw
OUTPUT_PATH="${OUTPUT_PATH:-./models/train/Wan2.1-Fun-V1.1-14B-Control-Camera_lora_realpose_${SPLIT}_${POSE_KEY}_job${SLURM_JOB_ID}}"
DS_CONFIG="${DS_CONFIG:-examples/wanvideo/model_training/full/accelerate_config_14B.yaml}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Parse GPU count from slurm env
NUM_GPUS="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-1}}"
if [[ "$NUM_GPUS" =~ ^gpu:([0-9]+)$ ]]; then
  NUM_GPUS="${BASH_REMATCH[1]}"
fi
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]]; then
  NUM_GPUS=1
fi

ACCELERATE_ARGS=(--config_file "$DS_CONFIG" --num_processes "$NUM_GPUS")

accelerate launch "${ACCELERATE_ARGS[@]}" \
  examples/wanvideo/model_training/train.py \
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
