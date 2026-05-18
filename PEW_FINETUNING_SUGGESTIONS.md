# Pew Finetuning Suggestions

- **Training is stable, but task quality is still poor.**
- **Loss drops nicely, but generated outputs are still mostly long off-task gibberish.**
- **Eval metrics improve over epochs, but absolute BLEU/ROUGE remain very low.**
- **`ce` goes to ~0 very quickly; training seems dominated by `l2`.**

## Suggestions

- **Add better task metrics**
  - exact match
  - normalized exact match
  - valid-option accuracy

- **Use a fresh output dir per run**
  - avoids mixing metrics/checkpoints across runs

- **Increase decoder pressure**
  - try `decoder_prob: 0.5`
  - maybe compare against `decoder_prob: 1.0`

- **Tune generation settings**
  - try `num_sampling_steps: [16, 32]`
  - try `cfgs: [1, 2, 3]`

- **Constrain outputs to the valid answer space**
  - if possible, map generations to valid answer options

- **Try lower conditioning dropout**
  - `label_drop_prob: 0.0`
  - or `label_drop_prob: 0.05`

## Best next experiment

- fresh output dir
- add exact-match-style eval
- set `decoder_prob: 0.5`
- test `steps=[16, 32]`, `cfg=[1, 2, 3]`
