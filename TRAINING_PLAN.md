# ELF Training Bring-Up Plan on a GPU Slurm Cluster

## Goal
Bring up training safely in stages:

1. verify training works on **1 GPU**
2. verify short-run stability on **1 GPU**
3. verify **single-node multi-GPU** training
4. only then attempt a longer or more realistic run

This repo was developed for TPU, so the main risks on GPU are:
- memory
- JAX multi-device behavior
- runtime/throughput
- checkpoint/restart behavior
- eval overhead during training

---

## Stage 0: Freeze a Known-Good Environment

### Objective
Use the same environment that already worked for inference.

### Actions
- Keep using the conda env:
  - `/home/claudiashi/.conda/envs/jax`
- In Slurm jobs, invoke Python directly via:
  - `/home/claudiashi/.conda/envs/jax/bin/python`
- Keep proxy vars unset in the job script:
  - `unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY`
- Keep Hugging Face caches on shared storage.

### Success Criteria
- training and inference use the exact same environment
- no environment-related import failures inside jobs

---

## Stage 1: Single-GPU Compile-and-Step Smoke Test

### Objective
Answer the first question:

**Can training run at all on 1 GPU?**

### Starting Point
Use:
- `src/configs/training_configs/train_owt_ELF-B.yml`

### Recommended Overrides
Use very conservative settings:
- `global_batch_size=1`
- `epochs=1`
- `use_wandb=false`
- `eval_freq=999999`
- `save_freq=1`
- `output_dir=outputs/elf_b_train_smoke_1gpu`
- optionally `log_freq=1` or `10`

### Why
The default config is TPU-scale and not appropriate for a first GPU training attempt.

### What to Look For
- JAX compiles successfully
- first training step runs
- loss logs appear
- checkpoint gets written

### Success Criteria
- log shows initial training step completed
- training loss is printed
- checkpoint exists in output dir

If this fails, do **not** proceed to multi-GPU yet.

---

## Stage 2: Single-GPU Short Stability Run

### Objective
Test whether 1-GPU training is stable for more than just the first compiled step.

### Run Setup
Still use `ELF-B`, but allow a longer short run:
- `global_batch_size=1` or `2`
- `epochs=1`
- `use_wandb=false`
- keep eval disabled
- new output dir, for example:
  - `outputs/elf_b_train_short_1gpu`

### What to Measure
- whether memory remains stable
- step time after compilation
- checkpoint saving
- whether loss behaves sensibly

### Success Criteria
- no OOM
- step rate is usable
- checkpoint save succeeds

---

## Stage 3: Verify Resume on Single GPU

### Objective
Confirm that interrupted or time-limited training jobs can resume correctly.

### Actions
- run a short job until at least one checkpoint is written
- stop the job
- relaunch using the same output dir

### Repo Behavior
The repo auto-resumes if checkpoints are present in `output_dir`.

### Success Criteria
- logs show resume from latest checkpoint
- training continues cleanly from resumed step/epoch

This is important before attempting longer jobs.

---

## Stage 4: Single-Node Multi-GPU Smoke Test

### Objective
Test whether the training code scales from 1 GPU to multiple GPUs on a **single node**.

### Why Single Node First
The training code uses:
- `jax.distributed.initialize()`
- `jax.pmap(...)`

So the safest scaling path is:
1. single GPU
2. single-node multi-GPU
3. only later consider multi-node

### What to Request
Start with:
- `--gres=gpu:2`

Only after that works, try:
- `--gres=gpu:4`

Do **not** start with 8 GPUs.

### Config Strategy
Keep settings conservative:
- for 2 GPUs: try `global_batch_size=2`
- for 4 GPUs: try `global_batch_size=4`
- still `epochs=1`
- still `use_wandb=false`
- still disable eval

### What to Check
- JAX sees multiple devices
- `pmap` runs correctly
- batch sharding works
- no distributed init hangs or deadlocks

### Success Criteria
- logs show multiple JAX devices
- training runs without device mismatch errors
- no deadlock at startup

---

## Stage 5: Multi-GPU Short Run

### Objective
Check whether multiple GPUs improve throughput and remain stable.

### Suggested Progression
After 2-GPU smoke test succeeds:
- try a slightly larger batch, e.g. `global_batch_size=4`

After 4-GPU smoke test succeeds:
- try `global_batch_size=8` if memory allows

### What to Compare Against 1 GPU
Measure:
- steps/sec
- memory stability
- checkpoint behavior
- overall runtime per training step

### Success Criteria
- throughput improves over 1 GPU
- no OOM
- no instability from multi-GPU execution

---

## Stage 6: Re-enable Evaluation During Training

### Objective
Verify that the full training workflow works, not just optimization.

### What to Re-enable
After training itself is stable:
- restore `eval_freq`
- restore generation during training
- optionally restore `use_wandb=true`

### Why This Is Separate
Eval introduces extra:
- runtime
- memory use
- checkpoint/output activity

### Success Criteria
- training checkpoints still save correctly
- eval runs complete
- generation outputs appear under `output_dir`

---

## Stage 7: Decide Whether Real Training Is Feasible

### Objective
Use the results from staged tests to decide whether larger-scale training is practical.

### Questions to Answer
1. What batch size fits on 1 GPU?
2. What batch size fits on 2 GPUs?
3. What batch size fits on 4 GPUs?
4. What throughput do we get at each scale?
5. Is `max_length=1024` practical for sustained training?
6. Do we need gradient accumulation?
7. Is ELF-B the only realistic model, or can ELF-M be attempted later?

### Likely Outcome
A realistic GPU training setup will likely require:
- `ELF-B`
- much smaller batch than TPU defaults
- longer wall-clock training time
- possibly gradient accumulation for larger effective batch

---

## Recommended Concrete Progression

### Run 1 — 1 GPU, minimal smoke test
- model: `ELF-B`
- `global_batch_size=1`
- `epochs=1`
- no eval
- no wandb

### Run 2 — 1 GPU, longer short run
- same as Run 1
- allow it to run long enough to verify stable steps and checkpointing

### Run 3 — 1 GPU, try larger batch
- try `global_batch_size=2` only if batch size 1 is comfortable

### Run 4 — 2 GPUs, smoke test
- `global_batch_size=2`
- no eval
- no wandb

### Run 5 — 2 GPUs, short run
- try `global_batch_size=4` if memory allows

### Run 6 — 4 GPUs, smoke test
- `global_batch_size=4`

### Run 7 — 4 GPUs, short run with eval restored
- restore eval
- optionally enable wandb

### Run 8 — actual experiment
- only after all earlier stages are successful

---

## Things to Keep Fixed Initially
To reduce variables during bring-up, keep these fixed at first:
- model = `ELF-B`
- dataset = OpenWebText config
- `max_length = 1024`
- same Python env
- same Hugging Face cache location
- same checkpoint behavior

Only vary:
- number of GPUs
- batch size
- eval on/off

---

## Important Repo-Specific Cautions

### 1. Default config is TPU-scale
Do not use `global_batch_size=512` for the first GPU training attempt.

### 2. Eval adds overhead
Disable eval until training itself is confirmed stable.

### 3. Auto-resume is enabled
Use fresh output dirs for test runs unless resume is intentional.

Examples:
- `outputs/elf_b_train_smoke_1gpu`
- `outputs/elf_b_train_short_1gpu`
- `outputs/elf_b_train_smoke_2gpu`

### 4. Multi-node should not be the first scaling step
First prove:
- single GPU training works
- single-node multi-GPU works

Only then think about multi-node training.

---

## Practical First Recommendation
If starting now, the best first training test is:

- model: `ELF-B`
- config base: `train_owt_ELF-B.yml`
- `global_batch_size=1`
- `epochs=1`
- `use_wandb=false`
- `eval_freq=999999`
- `save_freq=1`
- fresh `output_dir`

That is the safest way to determine whether training is feasible on your cluster.

---

## Bottom Line
The right path is:

1. **single-GPU smoke test**
2. **single-GPU short stable run**
3. **single-GPU resume test**
4. **2-GPU single-node smoke test**
5. **2/4-GPU short run**
6. **re-enable eval**
7. **decide whether full training is realistic**

This minimizes wasted cluster time and gives you a clear decision point at each stage.
