#!/bin/bash
#SBATCH --job-name=elf-100
#SBATCH --output=/storage/home/claudiashi/ELF/logs/%x-%j.out
#SBATCH --error=/storage/home/claudiashi/ELF/logs/%x-%j.err
#SBATCH --partition=h100
#SBATCH --account=ai_society
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

mkdir -p /storage/home/claudiashi/ELF/logs
mkdir -p /storage/home/claudiashi/ELF/.job_env


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

echo "==== Job info ===="
echo "HOSTNAME: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "=================="

CONDA_PY="/home/claudiashi/.conda/envs/jax/bin/python"

$CONDA_PY --version

$CONDA_PY - <<'PY'
import sys
print('python_executable:', sys.executable)
mods = ['jax', 'flax', 'datasets', 'transformers', 'torch']
for name in mods:
    try:
        m = __import__(name)
        print(f'{name}:', getattr(m, '__version__', 'ok'))
    except Exception as e:
        print(f'{name}: IMPORT_FAILED: {e!r}')
PY

srun $CONDA_PY -u eval.py \
  --config configs/training_configs/train_owt_ELF-B.yml \
  --checkpoint_path embedded-language-flows/ELF-B-owt \
  --config_override output_dir=outputs/elf_b_100_gpu \
  --config_override use_wandb=false \
  --config_override num_samples=100 \
  --config_override global_batch_size=1 \
  --config_override online_eval=false
