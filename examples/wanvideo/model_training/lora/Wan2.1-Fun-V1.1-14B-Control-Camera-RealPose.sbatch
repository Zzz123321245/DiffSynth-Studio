#!/bin/bash --login
#SBATCH -J wan21_14b_cam_lora_1
#SBATCH -p gpuA
#SBATCH -G 2
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem 128G
#SBATCH -t 1-0
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

mkdir -p "$SLURM_SUBMIT_DIR/logs"

cd "$SLURM_SUBMIT_DIR"

REPO_ROOT="$SLURM_SUBMIT_DIR"
SCRIPT_DIR="${REPO_ROOT}/examples/wanvideo/model_training/lora"
SCRIPT_PATH="${SCRIPT_DIR}/Wan2.1-Fun-V1.1-14B-Control-Camera-RealPose.sh"

echo "SubmitDir=$SLURM_SUBMIT_DIR"
echo "ScriptDir=$SCRIPT_DIR"
echo "RepoRoot=$REPO_ROOT"
echo "WorkDir(pwd)=$(pwd)"

echo "JobID=$SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_GPUS=${SLURM_GPUS:-unset} SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset}"

source /mnt/iusers01/eee01/p64096rw/scratch/software/miniconda3/bin/activate
conda activate /mnt/iusers01/eee01/p64096rw/scratch/software/miniconda3/envs/diffsynth

NUM_GPUS="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-2}}"
if [[ "$NUM_GPUS" =~ ^gpu:([0-9]+)$ ]]; then
  NUM_GPUS="${BASH_REMATCH[1]}"
fi
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]]; then
  NUM_GPUS=2
fi

export NUM_PROCESSES="${NUM_PROCESSES:-$NUM_GPUS}"
export NUM_MACHINES="${NUM_MACHINES:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export GRADIENT_ACCUMULATION_STEPS=4

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "[ERROR] Training script not found: $SCRIPT_PATH" >&2
  exit 1
fi

echo "Launching via $SCRIPT_PATH"
echo "NUM_PROCESSES=$NUM_PROCESSES NUM_MACHINES=$NUM_MACHINES"

bash "$SCRIPT_PATH"
