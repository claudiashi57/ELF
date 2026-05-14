#!/bin/bash
#SBATCH --job-name=elf-dl-owt
#SBATCH --output=/storage/home/claudiashi/ELF/logs/%x-%j.out
#SBATCH --error=/storage/home/claudiashi/ELF/logs/%x-%j.err
#SBATCH --partition=h100
#SBATCH --account=ai_society
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail
mkdir -p /storage/home/claudiashi/ELF/logs
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY
export HF_HOME=/checkpoint/fsx-ai-society/claudiashi/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
TARGET=/checkpoint/fsx-ai-society/claudiashi/huggingface/openwebtext-t5
mkdir -p /checkpoint/fsx-ai-society/claudiashi/huggingface
PYTHON_BIN=/home/claudiashi/.conda/envs/jax/bin/python

$PYTHON_BIN - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='embedded-language-flows/openwebtext-t5',
    repo_type='dataset',
    local_dir='/checkpoint/fsx-ai-society/claudiashi/huggingface/openwebtext-t5',
    local_dir_use_symlinks=False,
    resume_download=True,
)
print('SNAPSHOT_PATH', path)
PY
