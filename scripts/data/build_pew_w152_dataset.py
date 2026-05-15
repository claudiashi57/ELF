#!/usr/bin/env python3
"""Build a conditional-generation dataset from Pew Wave 152.

Creates one example per (respondent, survey-question) pair.
Conditioning text contains a rich respondent profile plus the survey prompt and
valid answer options. Target text is the survey-faithful answer label.
"""

import argparse
import csv
import hashlib
import json
import os
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from datasets import Dataset
from transformers import T5Tokenizer

NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
MISSING_LABEL_SNIPPETS = [
    'refused', "don't know", 'web blank', 'dk/', 'missing', 'min. val', 'max. val'
]
EXCLUDE_TARGET_PREFIXES = ('F_',)
EXCLUDE_TARGETS = {
    'QKEY', 'INTERVIEW_START_W152', 'INTERVIEW_END_W152', 'DEVICE_TYPE_W152',
    'SVYMODE_W152', 'LANG_W152', 'FORM_W152', 'WEIGHT_W152',
}
EXCLUDE_PROFILES = {'WEIGHT_W152'}


def normalize_code(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == '' or value.lower() == 'nan':
        return None
    try:
        num = float(value)
        if num.is_integer():
            return str(int(num))
    except ValueError:
        pass
    return value


def is_missing_label(label):
    l = label.strip().lower()
    return any(snippet in l for snippet in MISSING_LABEL_SNIPPETS)


def parse_codebook(path):
    with zipfile.ZipFile(path) as zf:
        shared = []
        sroot = ET.fromstring(zf.read('xl/sharedStrings.xml'))
        for si in sroot.findall(f'{NS}si'):
            shared.append(''.join(t.text or '' for t in si.iter(f'{NS}t')))
        root = ET.fromstring(zf.read('xl/worksheets/sheet1.xml'))

    meta = {}
    for row in root.find(f'{NS}sheetData').findall(f'{NS}row'):
        vals = []
        for cell in row.findall(f'{NS}c'):
            kind = cell.attrib.get('t')
            v = cell.find(f'{NS}v')
            if v is None:
                vals.append('')
            elif kind == 's':
                vals.append(shared[int(v.text)])
            else:
                vals.append(v.text)
        if not vals or vals[0] == 'Variable':
            continue
        var = vals[0]
        label = vals[1] if len(vals) > 1 and vals[1] else meta.get(var, {}).get('label', '')
        code = vals[2] if len(vals) > 2 else ''
        value_label = vals[3] if len(vals) > 3 else ''
        entry = meta.setdefault(var, {'label': label, 'value_labels': {}})
        if label and not entry['label']:
            entry['label'] = label
        norm_code = normalize_code(code)
        if norm_code is not None and value_label:
            entry['value_labels'][norm_code] = value_label
    return meta


def hash_split_user_disjoint(qkey, val_pct=10):
    digest = hashlib.md5(str(qkey).encode()).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return 'validation' if bucket < val_pct else 'train'


def hash_split_user_overlap(qkey, question_id, val_pct=10):
    digest = hashlib.md5(f'{qkey}::{question_id}'.encode()).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return 'validation' if bucket < val_pct else 'train'


def assign_split(split_mode, qkey, question_id, val_pct=10):
    if split_mode == 'user_disjoint':
        return hash_split_user_disjoint(qkey, val_pct=val_pct)
    if split_mode == 'user_overlap':
        return hash_split_user_overlap(qkey, question_id, val_pct=val_pct)
    raise ValueError(f'Unknown split_mode: {split_mode}')


def clean_label(label):
    return ' '.join(str(label).strip().split())


def build_profile(row, meta):
    parts = []
    for key in row:
        if not key.startswith('F_') or key in EXCLUDE_PROFILES:
            continue
        if key not in meta:
            continue
        code = normalize_code(row[key])
        if code is None:
            continue
        value_label = meta[key]['value_labels'].get(code)
        if not value_label or is_missing_label(value_label):
            continue
        field_name = meta[key]['label'] or key
        parts.append((field_name, clean_label(value_label)))
    return parts


def target_variables(fieldnames, meta):
    out = []
    for key in fieldnames:
        if key in EXCLUDE_TARGETS:
            continue
        if key.startswith(EXCLUDE_TARGET_PREFIXES):
            continue
        if key not in meta:
            continue
        label = meta[key]['label']
        value_labels = {
            code: clean_label(lbl)
            for code, lbl in meta[key]['value_labels'].items()
            if not is_missing_label(lbl)
        }
        if not label or len(value_labels) < 2:
            continue
        out.append(key)
    return out


def build_condition(profile_items, question_text, answer_options):
    profile_block = '\n'.join(f'- {k}: {v}' for k, v in profile_items)
    answers_block = '\n'.join(f'- {opt}' for opt in answer_options)
    return (
        'Respondent profile:\n'
        f'{profile_block}\n\n'
        'Survey question:\n'
        f'{question_text}\n\n'
        'Valid answer options:\n'
        f'{answers_block}\n\n'
        'Predict this respondent\'s preferred answer. Respond with one survey-faithful natural-language answer.'
    )


def tokenize_rows(rows, tokenizer):
    ds = Dataset.from_list(rows)

    def encode(ex):
        return {
            'condition_input_ids': tokenizer(ex['condition_text'], add_special_tokens=False)['input_ids'],
            'input_ids': tokenizer(ex['answer_text'], add_special_tokens=False)['input_ids'],
        }

    return ds.map(encode, remove_columns=ds.column_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', default='/checkpoint/fsx-ai-society/claudiashi/geom-bridging-pew/w152.csv')
    parser.add_argument('--codebook_path', default='/checkpoint/fsx-ai-society/claudiashi/geom-bridging-pew/codebook_w152.xlsx')
    parser.add_argument('--output_dir', default='/checkpoint/fsx-ai-society/claudiashi/datasets/processed/pew_survey_w152_full')
    parser.add_argument('--tokenizer_name', default='google-t5/t5-small')
    parser.add_argument('--split_mode', choices=['user_disjoint', 'user_overlap'], default='user_disjoint')
    parser.add_argument('--val_pct', type=int, default=10)
    parser.add_argument('--skip_tokenization', action='store_true')
    args = parser.parse_args()

    meta = parse_codebook(args.codebook_path)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw_rows = {'train': [], 'validation': []}
    with open(args.csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        targets = target_variables(reader.fieldnames, meta)
        for row in reader:
            qkey = row['QKEY']
            profile_items = build_profile(row, meta)
            if not profile_items:
                continue
            for var in targets:
                split = assign_split(args.split_mode, qkey, var, val_pct=args.val_pct)
                code = normalize_code(row[var])
                if code is None:
                    continue
                answer = meta[var]['value_labels'].get(code)
                if not answer or is_missing_label(answer):
                    continue
                options = [lbl for _, lbl in sorted(meta[var]['value_labels'].items(), key=lambda kv: kv[0]) if not is_missing_label(lbl)]
                if len(options) < 2:
                    continue
                raw_rows[split].append({
                    'id': f'{qkey}_{var}',
                    'respondent_id': qkey,
                    'question_id': var,
                    'condition_text': build_condition(profile_items, clean_label(meta[var]['label']), options),
                    'answer_text': clean_label(answer),
                })

    for split, rows in raw_rows.items():
        raw_path = outdir / f'{split}.jsonl'
        with open(raw_path, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f'saved raw {split} examples: {len(rows)} -> {raw_path}')

    if not args.skip_tokenization:
        os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
        tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name)
        train_ds = tokenize_rows(raw_rows['train'], tokenizer)
        val_ds = tokenize_rows(raw_rows['validation'], tokenizer)
        train_path = outdir / 'train_t5'
        val_path = outdir / 'validation_t5'
        train_ds.save_to_disk(str(train_path))
        val_ds.save_to_disk(str(val_path))
        print(f'saved tokenized train dataset to {train_path}')
        print(f'saved tokenized validation dataset to {val_path}')

    summary = {
        'split_mode': args.split_mode,
        'train_examples': len(raw_rows['train']),
        'validation_examples': len(raw_rows['validation']),
        'target_question_count': len({r['question_id'] for r in raw_rows['train'] + raw_rows['validation']}),
    }
    with open(outdir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
