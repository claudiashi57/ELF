#!/bin/bash
#SBATCH --job-name=pew-ft
#SBATCH --output=/storage/home/claudiashi/ELF/logs/%x-%j.out
#SBATCH --error=/storage/home/claudiashi/ELF/logs/%x-%j.err
#SBATCH --partition=h100
#SBATCH --account=ai_society
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=12:00:00

set -euo pipefail

mkdir -p /storage/home/claudiashi/ELF/logs
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY
export HF_HOME=/checkpoint/fsx-ai-society/claudiashi/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
unset WANDB_DISABLED
export WANDB_BASE_URL=https://meta-fair.wandb.io
export WANDB_ENTITY=claudiashi
if [ -f /storage/home/claudiashi/ELF/.env ]; then
  set -a
  . /storage/home/claudiashi/ELF/.env
  set +a
fi
export WANDB_BASE_URL=https://meta-fair.wandb.io

cd /storage/home/claudiashi/ELF/src
PYTHON_BIN="/home/claudiashi/.conda/envs/jax/bin/python"

echo "==== Job info ===="
echo "HOSTNAME: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "=================="

$PYTHON_BIN --version
$PYTHON_BIN - <<'PY'
import sys, jax
print('python_executable:', sys.executable)
print('jax:', jax.__version__)
print('backend:', jax.default_backend())
print('devices:', jax.devices())
PY

srun $PYTHON_BIN -u train.py \
  --config configs/finetuning_configs/finetune_pew_w152_ELF-B.yml \
  --config_override wandb_entity=claudiashi \
  ${WANDB_RUN_ID:+--config_override wandb_id=$WANDB_RUN_ID}
