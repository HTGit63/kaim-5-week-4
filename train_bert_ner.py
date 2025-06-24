#!/usr/bin/env python3

import numpy as np
from pathlib import Path
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    DataCollatorForTokenClassification, TrainingArguments, Trainer
)
from seqeval.metrics import classification_report, f1_score

# Constants
MODEL_NAME = "bert-base-multilingual-cased"
DATA_PATH  = str((Path(__file__).parent / "labeling" / "labeled_data.conll.txt").resolve())
OUTPUT_DIR = "models/bert_ner"

# Load & split
raw = load_dataset("conll2003", data_files={"train": DATA_PATH}, split="train", trust_remote_code=True)
splits = raw.train_test_split(test_size=0.1)
dataset = DatasetDict({"train": splits["train"], "eval": splits["test"]})

# Labels
labels = dataset["train"].features["ner_tags"].feature.names

# Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(labels))

# Align function (same logic as train_ner.py)
def tokenize_and_align(examples):
    tokenized = tokenizer(examples["tokens"], is_split_into_words=True,
                          truncation=True, padding="max_length", max_length=128)
    wids = tokenized.word_ids()
    all_labs = []
    for seq in examples["ner_tags"]:
        labs, prev = [], None
        for idx in wids:
            if idx is None or idx >= len(seq):
                labs.append(-100)
            elif idx != prev:
                labs.append(seq[idx])
            else:
                tag = labels[seq[idx]]
                if tag.startswith("B-"):
                    labs.append(labels.index("I-" + tag.split("-",1)[1]))
                else:
                    labs.append(seq[idx])
            prev = idx
        all_labs.append(labs)
    tokenized["labels"] = all_labs
    return tokenized

tokenized = dataset.map(tokenize_and_align, batched=True, remove_columns=["tokens","ner_tags"])
collator  = DataCollatorForTokenClassification(tokenizer)

# Metrics
def compute_metrics(p):
    preds, labs = np.argmax(p.predictions, axis=-1), p.label_ids
    tp, pr, rc = [], [], []
    tp, prc = [], []
    true_preds, true_labels = [], []
    for pr_row, lb_row in zip(preds, labs):
        p_seq, l_seq = [], []
        for p_, l_ in zip(pr_row, lb_row):
            if l_ != -100:
                p_seq.append(labels[p_]); l_seq.append(labels[l_])
        true_preds.append(p_seq); true_labels.append(l_seq)
    print(classification_report(true_labels, true_preds, digits=4))
    return {"f1": f1_score(true_labels, true_preds)}

# Training
args = TrainingArguments(
    output_dir=OUTPUT_DIR, num_train_epochs=3, per_device_train_batch_size=16,
    per_device_eval_batch_size=16, learning_rate=5e-5, weight_decay=0.01,
    logging_dir="logs_bert", logging_steps=50
)
trainer = Trainer(
    model=model, args=args,
    train_dataset=tokenized["train"], eval_dataset=tokenized["eval"],
    tokenizer=tokenizer, data_collator=collator, compute_metrics=compute_metrics
)
trainer.train()
metrics = trainer.evaluate()
trainer.save_model(OUTPUT_DIR)

# Write metrics
with open("metrics_bert.md", "w") as f:
    f.write("## BERT Multilingual NER Metrics\n\n")
    for k,v in metrics.items():
        f.write(f"- **{k}**: {v:.4f}\n")
print("✅ BERT fine-tuning done; metrics in metrics_bert.md")
