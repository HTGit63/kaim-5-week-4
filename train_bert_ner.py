import os
import numpy as np
from pathlib import Path
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
    """
    Reads a CoNLL-style file and returns a dict with 'tokens' and 'ner_tags'.
    Each sentence is separated by a blank line.
    """
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
                    continue  # skip malformed lines
                tok, tag = parts
                tokens.append(tok)
                tags.append(tag)
        # catch last sentence
        if tokens:
            tokens_batch.append(tokens)
            tags_batch.append(tags)
    return {"tokens": tokens_batch, "ner_tags": tags_batch}

# 2. Load data and split
raw_dict = read_conll(DATA_PATH)
full_dataset = Dataset.from_dict(raw_dict)
splits = full_dataset.train_test_split(test_size=0.1, seed=42)
dataset = DatasetDict({"train": splits["train"], "eval": splits["test"]})

# 3. Labels
labels = dataset["train"].features["ner_tags"].feature.names

# 4. Tokenizer & Model
os.makedirs(OUTPUT_DIR, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME, num_labels=len(labels)
)

# 5. Tokenize & align labels

def tokenize_and_align(examples):
    tokenized = tokenizer(
        examples["tokens"], is_split_into_words=True,
        truncation=True, padding="max_length", max_length=128
    )
    word_ids = tokenized.word_ids()
    all_labels = []
    for label_seq in examples["ner_tags"]:
        padded_labels = []
        prev_word_idx = None
        for word_idx in word_ids:
            if word_idx is None or word_idx >= len(label_seq):
                padded_labels.append(-100)
            elif word_idx != prev_word_idx:
                padded_labels.append(label_seq[word_idx])
            else:
                tag = labels[label_seq[word_idx]]
                if tag.startswith("B-"):
                    new_tag = "I-" + tag.split("-", 1)[1]
                    padded_labels.append(labels.index(new_tag))
                else:
                    padded_labels.append(label_seq[word_idx])
            prev_word_idx = word_idx
        all_labels.append(padded_labels)
    tokenized["labels"] = all_labels
    return tokenized

tokenized_datasets = dataset.map(
    tokenize_and_align, batched=True, remove_columns=["tokens", "ner_tags"]
)

# 6. Data collator
collator = DataCollatorForTokenClassification(tokenizer)

# 7. Metrics

def compute_metrics(p):
    preds, labels_id = np.argmax(p.predictions, axis=-1), p.label_ids
    true_preds, true_labels = [], []
    for pred_row, label_row in zip(preds, labels_id):
        pr, lb = [], []
        for p_, l_ in zip(pred_row, label_row):
            if l_ != -100:
                pr.append(labels[p_])
                lb.append(labels[l_])
        true_preds.append(pr)
        true_labels.append(lb)
    print(classification_report(true_labels, true_preds, digits=4))
    return {"f1": f1_score(true_labels, true_preds)}

# 8. Training arguments
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

# 9. Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["eval"],
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics
)

# 10. Train & evaluate
trainer.train()
metrics = trainer.evaluate()

# 11. Save outputs
trainer.save_model(str(OUTPUT_DIR))
with open("metrics_bert.md", "w", encoding="utf-8") as f:
    f.write("## BERT Multilingual NER Metrics\n\n")
    for k, v in metrics.items():
        f.write(f"- **{k}**: {v:.4f}\n")
print("✅ BERT fine-tuning done; metrics saved.")
