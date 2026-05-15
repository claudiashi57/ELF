#!/usr/bin/env python3
"""Preprocess Pew survey data into ELF conditional-generation datasets.

Expected input: JSONL with one example per line. Each example should include:
- `prompt`: survey question or prompt text
- `answer_text`: target answer in natural language or canonical verbalization
- optional user/profile fields such as `age`, `gender`, `party`, `ideology`, etc.

Output: Hugging Face datasets saved with `save_to_disk`, containing:
- `condition_input_ids`
- `input_ids`
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset
from transformers import T5Tokenizer


DEFAULT_EXCLUDE = {"prompt", "answer_text", "split", "id"}


def build_condition(example):
    prompt = example["prompt"].strip()
    attrs = []
    for key, value in example.items():
        if key in DEFAULT_EXCLUDE or value is None:
            continue
        attrs.append(f"- {key}: {value}")
    attr_block = "\n".join(attrs) if attrs else "- none provided"
    return (
        "User attributes:\n"
        f"{attr_block}\n\n"
        "Survey prompt:\n"
        f'"{prompt}"\n\n'
        "Write the respondent's preferred answer in natural language."
    )


def encode_examples(rows, tokenizer):
    def encode(ex):
        return {
            "condition_input_ids": tokenizer(
                build_condition(ex), add_special_tokens=False,
            )["input_ids"],
            "input_ids": tokenizer(
                ex["answer_text"].strip(), add_special_tokens=False,
            )["input_ids"],
        }

    ds = Dataset.from_list(rows)
    return ds.map(encode, remove_columns=ds.column_names)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tokenizer_name", default="google-t5/t5-small")
    args = parser.parse_args()

    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(args.train_jsonl)
    train_ds = encode_examples(train_rows, tokenizer)
    train_path = output_dir / "train_t5"
    train_ds.save_to_disk(str(train_path))
    print(f"saved train dataset to {train_path}")

    if args.val_jsonl:
        val_rows = load_jsonl(args.val_jsonl)
        val_ds = encode_examples(val_rows, tokenizer)
        val_path = output_dir / "validation_t5"
        val_ds.save_to_disk(str(val_path))
        print(f"saved validation dataset to {val_path}")


if __name__ == "__main__":
    main()
