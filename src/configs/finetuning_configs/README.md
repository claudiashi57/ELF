# Finetuning Configs

Use this directory for configs that start from a pretrained ELF checkpoint and adapt to a new dataset.

Suggested convention:

- `finetune_<dataset>_ELF-B.yml`
- keep `training_configs/` for pretraining or baseline repo configs
- point `data_path` / `eval_data_path` at processed datasets saved with `datasets.save_to_disk()`

Recommended future distinction:

- `resume`: continue the same run
- `init_checkpoint`: initialize a new fine-tuning run from a pretrained checkpoint

For now, place all task-specific fine-tuning YAMLs here.
