#!/usr/bin/env python3

import os
import numpy as np
from pathlib import Path
from itertools import chain
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
from seqeval.metrics import classification_report, f1_score

# Constants
MODEL_NAME = "bert-base-multilingual-cased"
DATA_PATH = Path(__file__).parent / "labeling" / "labeled_data.conll.txt"
OUTPUT_DIR = Path("models") / "bert_ner"

# 1. Manual CoNLL reader

def read_conll(path: Path):
    tokens_batch, tags_batch = [], []
    tokens, tags = [], []
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    tokens_batch.append(tokens)
                    tags_batch.append(tags)
                    tokens, tags = [], []
            else:
                parts = line.split()
                if len(parts) != 2:
                    continue
                tok, tag = parts
                tokens.append(tok)
                tags.append(tag)
        if tokens:
            tokens_batch.append(tokens)
            tags_batch.append(tags)
    return {"tokens": tokens_batch, "ner_tags": tags_batch}

# 2. Load raw data and extract label list
raw_dict = read_conll(DATA_PATH)
# unique labels (preserve order O, then others)
all_tags = list(dict.fromkeys(chain.from_iterable(raw_dict["ner_tags"])))
labels = all_tags

# 3. Create Hugging Face Dataset and split
full_dataset = Dataset.from_dict(raw_dict)
splits = full_dataset.train_test_split(test_size=0.1, seed=42)
dataset = DatasetDict({"train": splits["train"], "eval": splits["test"]})

# 4. Prepare output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 5. Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME, num_labels=len(labels)
)

# 6. Tokenize & align labels

def tokenize_and_align(examples):
    tokenized = tokenizer(
        examples["tokens"], is_split_into_words=True,
        truncation=True, padding="max_length", max_length=128
    )
    word_ids = tokenized.word_ids()
    batch_labels = []
    for tag_seq in examples["ner_tags"]:
        aligned = []
        prev_idx = None
        for idx in word_ids:
            if idx is None or idx >= len(tag_seq):
                aligned.append(-100)
            elif idx != prev_idx:
                aligned.append(labels.index(tag_seq[idx]))
            else:
                tag = tag_seq[idx]
                if tag.startswith("B-"):
                    new_tag = "I-" + tag.split("-",1)[1]
                    aligned.append(labels.index(new_tag))
                else:
                    aligned.append(labels.index(tag))
            prev_idx = idx
        batch_labels.append(aligned)
    tokenized["labels"] = batch_labels
    return tokenized

tokenized_datasets = dataset.map(
    tokenize_and_align, batched=True, remove_columns=["tokens", "ner_tags"]
)

# 7. Data collator
collator = DataCollatorForTokenClassification(tokenizer)

# 8. Metrics

def compute_metrics(p):
    preds, true = np.argmax(p.predictions, axis=-1), p.label_ids
    true_preds, true_labels = [], []
    for pred_row, label_row in zip(preds, true):
        p_seq, l_seq = [], []
        for p_, l_ in zip(pred_row, label_row):
            if l_ != -100:
                p_seq.append(labels[p_])
                l_seq.append(labels[l_])
        true_preds.append(p_seq)
        true_labels.append(l_seq)
    print(classification_report(true_labels, true_preds, digits=4))
    return {"f1": f1_score(true_labels, true_preds)}

# 9. Training arguments
args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="logs_bert",
    logging_steps=50,
    seed=42,
    save_total_limit=2
)

# 10. Trainer setup
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["eval"],
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics
)

# 11. Train, evaluate, and save
trainer.train()
metrics = trainer.evaluate()
trainer.save_model(str(OUTPUT_DIR))
with open("metrics_bert.md", "w", encoding="utf-8") as f:
    f.write("## BERT Multilingual NER Metrics\n\n")
    for k, v in metrics.items():
        f.write(f"- **{k}**: {v:.4f}\n")
print("✅ BERT fine-tuning done; metrics saved.")
