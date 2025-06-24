
"""
Fine-tune DistilBERT multilingual for Amharic NER.
Loads local CoNLL file (labeling/labeled_data.conll.txt), splits train/eval 90/10,
tokenizes with `is_split_into_words=True`, aligns BIO tags, trains for N epochs,
evaluates (precision/recall/F1 via seqeval), saves model under models/distilbert_ner/,
writes evaluation_report.md.
Usage:
    python train_ner.py
"""


import os
import numpy as np
from pathlib import Path
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
from seqeval.metrics import classification_report, f1_score

# 1. Constants
MODEL_NAME = "distilbert-base-multilingual-cased"
DATA_PATH  = str((Path(__file__).parent / "labeling" / "labeled_data.conll.txt").resolve())
OUTPUT_DIR = "models/distilbert_ner"

# 2. Load and split dataset
raw = load_dataset(
    "conll2003",
    data_files={"train": DATA_PATH},
    split="train",
    trust_remote_code=True
)
splits = raw.train_test_split(test_size=0.1)
dataset = DatasetDict({
    "train": splits["train"],
    "eval":  splits["test"]
})

# 3. Extract label names
labels = dataset["train"].features["ner_tags"].feature.names

# 4. Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels)
)

# 5. Tokenize & align labels
def tokenize_and_align(examples):
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=128
    )
    word_ids = tokenized.word_ids()
    all_labels = []
    for label_seq in examples["ner_tags"]:
        padded = []
        prev_idx = None
        for idx in word_ids:
            if idx is None or idx >= len(label_seq):
                padded.append(-100)
            elif idx != prev_idx:
                padded.append(label_seq[idx])
            else:
                # continuation: if B- → I-
                tag = labels[label_seq[idx]]
                if tag.startswith("B-"):
                    new_tag = "I-" + tag.split("-", 1)[1]
                    padded.append(labels.index(new_tag))
                else:
                    padded.append(label_seq[idx])
            prev_idx = idx
        all_labels.append(padded)
    tokenized["labels"] = all_labels
    return tokenized

tokenized_datasets = dataset.map(tokenize_and_align, batched=True, remove_columns=["tokens", "ner_tags"])

# 6. Data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# 7. Metrics
def compute_metrics(p):
    preds, labs = p
    preds = np.argmax(preds, axis=-1)
    true_preds, true_labels = [], []
    for pred_row, lab_row in zip(preds, labs):
        pr, lb = [], []
        for p_, l_ in zip(pred_row, lab_row):
            if l_ != -100:
                pr.append(labels[p_])
                lb.append(labels[l_])
        true_preds.append(pr)
        true_labels.append(lb)
    report = classification_report(true_labels, true_preds, digits=4)
    print(report)
    return {"f1": f1_score(true_labels, true_preds)}

# 8. TrainingArguments (compatible)
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=10
)

# 9. Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["eval"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 10. Train & evaluate
trainer.train()
metrics = trainer.evaluate()

# 11. Save model & metrics
trainer.save_model(OUTPUT_DIR)
with open("evaluation_report.md", "w") as f:
    f.write("## NER Fine-Tuning Evaluation\n\n")
    for k, v in metrics.items():
        f.write(f"- **{k}**: {v:.4f}\n")
print(f"\n✅ Saved model to {OUTPUT_DIR} and metrics to evaluation_report.md")
