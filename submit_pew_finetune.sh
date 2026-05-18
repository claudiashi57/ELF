#!/bin/bash
set -euo pipefail

cd /storage/home/claudiashi/ELF
mkdir -p logs

ts="$(date -u +%Y%m%d_%H%M%S)"

sbatch \
  --output="/storage/home/claudiashi/ELF/logs/pew-ft-${ts}-%j.out" \
  --error="/storage/home/claudiashi/ELF/logs/pew-ft-${ts}-%j.err" \
  run_pew_finetune.sbatch
