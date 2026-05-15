# Pew Survey Fine-Tuning Plan

## Goal
Post-train ELF on Pew survey data so the model can condition on a respondent profile and a survey prompt, then predict the respondent's preferred answer in natural language.

## Plan

### 1. Start with one wave
Use `w152.csv` first.

Why:
- focused AI topic
- smaller and cleaner first experiment
- easier to debug than mixing multiple waves

### 2. Define conditioning features
Use a small stable subset of respondent attributes from the `F_*` columns, for example:
- `F_AGECAT`
- `F_GENDER`
- `F_EDUCCAT` or `F_EDUCCAT2`
- `F_RACECMB`
- `F_RACETHNMOD`
- `F_RELIG` or `F_RELIGCAT1`
- `F_PARTY_FINAL`
- `F_PARTYSUM_FINAL`
- `F_PARTYSUMIDEO_FINAL`
- `F_IDEO`
- optionally `F_CREGION`, `F_CDIVISION`, `F_METRO`

These become the profile text in the conditioning string.

### 3. Pick a small question set
Select 5-10 target question columns from `w152` for the first pass.

Keep the first version narrow and avoid sparse or form-specific fields.

### 4. Recover text from the codebook
Use `codebook_w152.xlsx` to map:
- survey column name -> question text
- coded response value -> answer label text

This is required because the CSV contains integer-coded responses, not natural-language answers.

### 5. Build training examples
For each respondent-question pair:
- **condition** = serialized respondent attributes + survey question text
- **target** = natural-language or canonical verbalization of the coded answer

Recommended first target style:
- canonical answer verbalization
- short and consistent labels

### 6. Preprocess into ELF format
Use `scripts/data/preprocess_pew_survey.py` as the preprocessing entry point.

Output datasets should be saved under:
- `/checkpoint/fsx-ai-society/claudiashi/datasets/processed/pew_survey_w152/train_t5`
- `/checkpoint/fsx-ai-society/claudiashi/datasets/processed/pew_survey_w152/validation_t5`

Expected processed fields:
- `condition_input_ids`
- `input_ids`

### 7. Add a fine-tuning config
Create and use a config under:
- `src/configs/finetuning_configs/`

This config should point to the processed Pew dataset and use ELF-B first.

### 8. Run a smoke fine-tune
Start with a tiny run:
- 1 GPU
- very small batch
- eval disabled initially
- confirm loss, checkpointing, and W&B logging

### 9. Scale only after the smoke run works
After the first run is stable:
- increase data coverage
- add more question columns
- optionally add more survey waves
- then try longer runs or 2-GPU fine-tuning

## Immediate Next Step
Pick the exact `w152` question columns for the first experiment and extract their question/answer text from `codebook_w152.xlsx`.
