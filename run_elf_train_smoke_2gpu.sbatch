#!/bin/bash
#SBATCH --job-name=elf-train-2g
#SBATCH --output=/storage/home/claudiashi/ELF/logs/%x-%j.out
#SBATCH --error=/storage/home/claudiashi/ELF/logs/%x-%j.err
#SBATCH --partition=h100
#SBATCH --account=ai_society
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00

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
export WANDB_DISABLED=true

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
  --config configs/training_configs/train_owt_ELF-B.yml \
  --config_override output_dir=outputs/elf_b_train_smoke_2gpu \
  --config_override data_path=/checkpoint/fsx-ai-society/claudiashi/huggingface/openwebtext-t5 \
  --config_override global_batch_size=2 \
  --config_override epochs=1 \
  --config_override use_wandb=false \
  --config_override eval_freq=999999 \
  --config_override save_freq=1 \
  --config_override log_freq=10
