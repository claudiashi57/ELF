# Repository Guidelines

## Project Goal
The immediate contributor goal for this repository is to verify that this TPU-first JAX codebase also runs correctly on GPU. Prioritize changes that improve GPU compatibility, smoke testing, and reproducible local or Slurm-based validation without breaking the original TPU workflow.

## Storage Guidance
For large files and generated artifacts, always write to `/checkpoint/fsx-ai-society/claudiashi` instead of the repository tree or home directory. Use that path for datasets, Hugging Face caches, checkpoints, and other bulky outputs to avoid filling local disk and to match the existing cluster scripts.

## Project Structure & Module Organization
Core code lives in `src/`. Use `src/train.py` for training, `src/eval.py` for checkpoint evaluation, and `src/generation.py`/`src/train_step.py` for sampling and step logic. Model definitions are in `src/modules/`, shared helpers in `src/utils/`, and YAML configs in `src/configs/{training_configs,sampling_configs}/`. Visual assets for the paper live in `assets/`. Cluster job scripts are at the repo root (`run_*.sbatch`). Treat `logs/` and `src/outputs/` as generated artifacts; do not commit new runs unless explicitly needed.

## Build, Test, and Development Commands
Create an environment and install dependencies with `pip install -r requirements.txt`.
Run training from the repo root with `python src/train.py --config src/configs/training_configs/train_owt_ELF-B.yml`.
Run evaluation with `python src/eval.py --config src/configs/training_configs/train_owt_ELF-B.yml --checkpoint_path embedded-language-flows/ELF-B-owt`.
For cluster smoke checks, use the provided Slurm scripts such as `sbatch run_elf_sanity_gpu.sbatch` or `sbatch run_elf_train_smoke_1gpu.sbatch`.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes, and uppercase constants only when truly constant. Keep modules focused by concern (`data_utils.py`, `metrics_utils.py`, etc.). Prefer short docstrings for public helpers and maintain importable script behavior by keeping repo-root path setup intact in entrypoints.

## Testing Guidelines
There is no dedicated `tests/` suite yet. Validate changes with a lightweight eval or smoke run before opening a PR. Prefer the smallest reproducible command, for example `python src/eval.py ... --config_override num_samples=8 --config_override use_wandb=false` or the `run_elf_train_smoke_*.sbatch` jobs. If you add new logic, include a clear reproduction command in the PR.

## Commit & Pull Request Guidelines
Git history is currently very small (`init`), so follow a simple imperative style: `add xsum config override`, `fix pmap batch sharding`. Keep commits scoped to one change. PRs should explain the motivation, list config or dataset assumptions, link related issues, and include sample metrics or log snippets when behavior changes.
