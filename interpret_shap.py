
"""
Compute leave-one-word-out importances for sample Amharic sentences.
For each sentence: split on whitespace, remove each word in turn, run model to get 'score'
(sum of max logits over tokens), measure delta from baseline. Print top tokens by Δscore.
Usage:
    python interpret_shap.py
"""


import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
import numpy as np

# 1. Load your trained DistilBERT NER model
MODEL_DIR = "models/distilbert_ner"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
model.eval()

# 2. Sample sentences (whitespace-separated “words” will be removed one by one)
samples = [
    "8420 ላይ በመደወል ወይም መልዕክት በመላክ ምቹ የህፃናት ተንቀሳቃሽ መተኛ ከነትራሱ ይዘዙ!",
    "በሚቀጥለው ሳምንት Apple iPhone 13 በቅናሽ ገንዘብ ይሸጣል።",
    "እቃዎቻችን በአካባቢያችን የተለያዩ ቦታዎች እየደረሱ ነው።",
    "በ3000 ብር ብቻ ።",
    "አድራሻ:-መገናኛ መሰረት ደፋር ሞል ሁለተኛ ፎቅ"
]

def compute_score(text: str) -> float:
    """
    Tokenize the text (plain string), run model, and return a scalar “score”.
    We take sum of max logits across tokens (ignoring padding via attention_mask).
    """
    # Tokenize on plain string
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length"
    )
    # Run model
    with torch.no_grad():
        out = model(**encoding).logits  # shape: (1, seq_len, num_labels)
    # For each token position, take max logit across labels
    # out[0]: (seq_len, num_labels)
    max_logits = out[0].max(dim=-1).values.cpu().numpy()  # (seq_len,)
    # Only consider non-pad tokens
    mask = encoding["attention_mask"][0].cpu().numpy()  # (seq_len,)
    # Score = sum of max logits over real tokens
    return float((max_logits * mask).sum())

def leave_one_out_importance(text: str):
    """
    For each whitespace "word" in text, remove it and compute the drop in score.
    Returns list of (word, delta_score, normalized_delta).
    """
    # Split on whitespace. We preserve tokens exactly, but removing punctuation is possible if desired.
    words = text.split()
    if len(words) == 0:
        return []
    # Baseline
    base_score = compute_score(text)
    importances = []
    for i in range(len(words)):
        # Reconstruct sentence without words[i]
        new_words = words[:i] + words[i+1:]
        # Join with spaces; if empty, use empty string
        new_text = " ".join(new_words)
        # Compute new score
        score_i = compute_score(new_text) if new_text.strip() else 0.0
        delta = base_score - score_i
        importances.append((words[i], delta))
    # Normalize (absolute sum)
    arr = np.array([abs(d) for _, d in importances], dtype=float)
    s = arr.sum()
    normalized = [(imp[0], imp[1], (abs(imp[1]) / s) if s != 0 else 0.0) for imp in importances]
    return normalized

def display_importances(text: str):
    imp = leave_one_out_importance(text)
    # Sort by absolute delta descending
    imp_sorted = sorted(imp, key=lambda x: abs(x[1]), reverse=True)
    print(f"\nSentence:\n{text}\nWord importances:")
    for word, delta, norm in imp_sorted[:10]:  # top 10
        print(f"  '{word}': Δscore={delta:.4f}, normalized={norm:.4f}")
    print()

if __name__ == "__main__":
    for sent in samples:
        display_importances(sent)
