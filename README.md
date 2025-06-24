# EthioMart Telegram NER & Vendor Scoring Pipeline

**Author:** Hunde Tefera  
**Repository:** [https://github.com/HTGit63/kaim-5-week-4](https://github.com/HTGit63/kaim-5-week-4)

---

## Project Overview

This project implements an end-to-end pipeline for extracting, preprocessing, and analyzing Ethiopian Telegram channel data, focusing on Named Entity Recognition (NER) and vendor scoring for EthioMart.

### Key Objectives

1. **Data Ingestion:**  
    - Connect to multiple Telegram channels  
    - Fetch real-time messages  
    - Capture metadata

2. **Data Preprocessing:**  
    - Normalize Amharic text (strip diacritics, standardize numbers)  
    - Clean messages (remove URLs, collapse whitespace)  
    - Tokenize for downstream tasks

3. **NER Fine-Tuning:**  
    - Label a small sample in CoNLL format  
    - Fine-tune a multilingual transformer (DistilBERT)  
    - Recognize entities (LOC, PER, ORG, MISC) in Amharic text

4. **Model Comparison & Interpretability:**  
    - Compare DistilBERT vs. BERT performance  
    - Interpret model behavior via a leave-one-out importance script

5. **Vendor Analysis & Lending Score:**  
    - Extract price mentions  
    - Compute posting frequency per channel  
    - Derive a normalized “lending score” to inform EthioMart’s financial decisions

6. **Code Quality & Documentation:**  
    - Provide clear scripts, inline comments, and a comprehensive README for reproducibility

**Reproducibility:**  
By following the steps below, anyone can reproduce the pipeline: ingest Telegram data, fine-tune the NER model, interpret results, and compute vendor scores.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Configuration File (`.env`)](#configuration-file-env)
4. [Directory Structure](#directory-structure)
5. [Installation](#installation)
6. [Usage & Scripts](#usage--scripts)
    - [Stage 1: Data Ingestion & Preprocessing](#stage-1-data-ingestion--preprocessing)
    - [Stage 2: NER Fine-Tuning](#stage-2-ner-fine-tuning)
    - [Stage 3: Interpretability](#stage-3-interpretability)
    - [Stage 4: Vendor Analysis & Lending Score](#stage-4-vendor-analysis--lending-score)
7. [Key Files & Functions](#key-files--functions)
8. [Outputs & Artifacts](#outputs--artifacts)
9. [Extending & Troubleshooting](#extending--troubleshooting)
10. [Contributing](#contributing)
11. [License](#license)

---

## Prerequisites

- **Python:** ≥ 3.8
- **Git:** To clone and manage the repository
- **Telegram account:** For API access to channels you have permission to read
- **Hugging Face Hub access (optional):** If you wish to push or pull models, but not required for local runs
- **Compute:** CPU-only is supported; GPU accelerates model fine-tuning

---

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/HTGit63/kaim-5-week-4.git
cd kaim-5-week-4
```

### 2. Create a Python Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate     # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**requirements.txt includes:**

- telethon
- python-dotenv
- transformers
- datasets
- seqeval
- accelerate
- pandas
- numpy
- shap (optional; used if SHAP interpretability is desired)
- Any other listed scripts’ dependencies

---

## Configuration File (.env)

Some scripts require Telegram API credentials. In the repo root, create a file named `.env` with the following keys:

```dotenv
# Telegram API credentials (get from https://my.telegram.org)
TG_API_ID=YOUR_API_ID
TG_API_HASH=YOUR_API_HASH

# (Optional) phone number if Telethon needs interactive login
# phone=+2517XXXXXXXX
```

> **Note:** Do not commit `.env` to version control. It is included in `.gitignore`.

---

## Directory Structure

```
kaim-5-week-4/
├── .venv/                          # Python virtual environment (ignored)
├── .gitignore
├── README.md
├── requirements.txt
├── ingestion/
│   └── telegram_fetcher.py         
├── preprocessing/
│   ├── preprocess.py               
│   └── raw_messages_*.json         
│   └── processed_*.json            
├── labeling/
│   ├── labeled_data.conll.txt      
│   └── validate_conll.py          
├── run_pipeline.py                 
├── final_dataset.json              
├── train_ner.py                    
├── train_bert_ner.py               
├── evaluation_report.md            
├── metrics_bert.md                 
├── metrics_comparison.md               
├── interpret_shap.py               
├── analysis/
│   └── vendor_scorecard.py         
│   └── vendor_scorecard.csv        
├── docs/
│   ├── Week4_Interim.md           
│   └── Week4_Interim_Report.pdf    
└── models/                         
    └── distilbert_ner/
    └── bert_ner/

```

---

## Installation

After cloning and activating your environment, ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

Check that you have a valid `.env` with `TG_API_ID` and `TG_API_HASH`. Then proceed to usage.

---

## Usage & Scripts

### Stage 1: Data Ingestion & Preprocessing

#### 1.1 Fetch Telegram Messages

```bash
python ingestion/telegram_fetcher.py
```

- Prompts for phone or bot token if not already authorized
- Connects to channels listed in `ingestion/telegram_fetcher.py` (`CHANNELS` list)
- Fetches recent messages (configurable limit) with metadata: id, date, sender, text, channel
- Saves raw JSON under `preprocessing/` as `raw_messages_<timestamp>.json`

#### 1.2 Preprocess Messages

```bash
python preprocessing/preprocess.py
```

- Locates the latest `raw_messages_*.json` in `preprocessing/`
- Loads messages, applies:
  - Amharic normalization: strips diacritics, normalizes Ethiopian numerals to Latin digits
  - Cleaning: removes URLs, collapses whitespace, strips control characters
  - Tokenization: simple whitespace split (placeholder for more advanced Amharic tokenizer)
- Skips empty messages; outputs processed JSON: `processed_<timestamp>.json` in `preprocessing/`

#### 1.3 Run End-to-End Pipeline

```bash
python run_pipeline.py
```

- Orchestrates:
  - Data ingestion (`telegram_fetcher.py`)
  - Data preprocessing (`preprocess.py`)
- Aggregates processed messages into `final_dataset.json` in repo root
- Check logs to confirm counts of messages processed

---

### Stage 2: NER Fine-Tuning

Assume you have labeled a small sample in `labeling/labeled_data.conll.txt` (validated with `labeling/validate_conll.py`).

#### 2.1 Fine-Tune DistilBERT

```bash
python train_ner.py
```

- Loads CoNLL file from `labeling/labeled_data.conll.txt`
- Splits 90% train / 10% eval
- Tokenizes with `distilbert-base-multilingual-cased`, aligns labels; fine-tunes for 3 epochs
- Saves checkpoint under `models/distilbert_ner/`
- Writes evaluation metrics (precision, recall, F1) to `evaluation_report.md`

> If download speed for XLM-RoBERTa is slow, use DistilBERT as in this script. Likewise, ensure `accelerate` is installed for Trainer.

#### 2.2 (Optional) Fine-Tune BERT

```bash
python train_bert_ner.py
```

- Similar flow but uses `bert-base-multilingual-cased`
- Saves to `models/bert_ner/`, writes metrics to `metrics_bert.md`
- Compare with DistilBERT’s metrics in `metrics_comparison.md`

---

### Stage 3: Interpretability

#### 3.1 Leave-One-Out Importance

```bash
python interpret_leave_one_out.py
```

- For sample Amharic sentences (edit samples in script), removes one whitespace-token at a time
- Runs the DistilBERT NER model, measures change in “score” (sum of max logits across tokens)
- Prints each word’s Δscore and normalized importance
- Use these insights to understand which tokens drive model confidence

#### 3.2 (Optional) SHAP Interpretability

```bash
python interpret_shap.py
```

- A more complex approach using SHAP’s text masking
- May require additional adjustments; see comments in script
- If SHAP masking mismatches, prefer leave-one-out for robust results

---

### Stage 4: Vendor Analysis & Lending Score

```bash
python analysis/vendor_scorecard.py
```

- Loads `final_dataset.json`
- If text field is missing, reconstructs it by joining tokens from preprocessing
- Parses dates, groups by channel & week, computes average `posts_per_week`
- Extracts prices from message text via regex for patterns like `(\d+) ብር`
- Computes `avg_price` per channel
- Normalizes metrics (posts, price) into [0,1], combines with weighted formula (`0.6*freq + 0.4*price`) into `lending_score`
- Outputs `analysis/vendor_scorecard.csv` and prints top channels by score

> Adjust regex or weighting as needed for EthioMart’s priorities.

---

## Key Files & Functions

- **ingestion/telegram_fetcher.py**
  - Loads Telegram API credentials from `.env`
  - Defines `CHANNELS` list to crawl
  - Uses Telethon to fetch messages (ids, dates, senders, text) with retry/error handling
  - Saves raw JSON for preprocessing

- **preprocessing/preprocess.py**
  - Normalizes Amharic text: strips diacritics, standardizes Ethiopian numerals
  - Cleans URLs, whitespace, control chars
  - Tokenizes via whitespace
  - Saves processed JSON

- **run_pipeline.py**
  - Calls ingestion and preprocessing in sequence
  - Produces `final_dataset.json`: a unified JSON array with fields like channel, id, date, tokens (or text)

- **labeling/labeled_data.conll.txt & labeling/validate_conll.py**
  - Sample of ~30 messages manually labeled with BIO tags for NER
  - Validation script checks CoNLL formatting

- **train_ner.py**
  - Loads the CoNLL file locally (absolute path)
  - Splits into train/eval, tokenizes, aligns labels for DistilBERT, fine-tunes with Trainer
  - Saves model and writes `evaluation_report.md`

- **train_bert_ner.py (optional)**
  - Same flow but uses BERT multilingual
  - Writes `metrics_bert.md`

- **interpret_leave_one_out.py**
  - Simple interpretability via leave-one-word-out
  - Uses tokenizer & model saved checkpoints; prints word-level importance

- **interpret_shap.py (optional)**
  - Attempts SHAP explanations for token-level importances. May require additional debugging

- **analysis/vendor_scorecard.py**
  - Loads `final_dataset.json`, reconstructs text if needed
  - Parses ISO dates, groups by week, extracts prices (regex), computes average price and posting frequency per channel
  - Normalizes and combines into `lending_score`. Outputs CSV and prints summary

---

## Outputs & Artifacts

- **Raw & processed message files**
  - `preprocessing/raw_messages_<timestamp>.json`
  - `preprocessing/processed_<timestamp>.json`

- **Final dataset**
  - `final_dataset.json` (list of processed messages ready for analysis)

- **Model checkpoints** (ignored by default, large)
  - `models/distilbert_ner/`
  - `models/bert_ner/` (if run)

- **Evaluation metrics**
  - `evaluation_report.md` (DistilBERT: precision/recall/F1)
  - `metrics_bert.md` (BERT metrics)
  - `metrics_comparison.md` (comparison table)

- **Interpretability results**
  - Printed leave-one-out importances (no file; captured in console or redirect to text file)

- **Vendor scoring**
  - `analysis/vendor_scorecard.csv`: columns `[channel, posts_per_week, avg_price, lending_score]`

- **Documentation**
  - `docs/Week4_Interim.md` & PDF: interim report
  - This `README.md` for end-to-end instructions

---

## Extending & Troubleshooting

- **Adding channels:**  
  Edit `ingestion/telegram_fetcher.py` → `CHANNELS` list (strip leading “@”). Ensure you are joined or have permission to read public channels.

- **Increasing fetch limits:**  
  Modify limit in `iter_messages(channel, limit=...)`. Beware of rate limits/timeouts for large volumes.

- **Advanced Amharic tokenization:**  
  Replace `tokenize()` in `preprocess.py` with a dedicated Amharic tokenizer library when available.

- **NER labeling:**  
  Increase labeled sample size and refine labels to improve model. Consider active learning: run model predictions on unlabeled data, correct mistakes.

- **Model hyperparameters:**  
  In `train_ner.py`, adjust `num_train_epochs`, `learning_rate`, `batch_size`. Monitor overfitting on small dataset.

- **Interpretability:**  
  Use Captum (PyTorch) for integrated gradients if deeper attribution is desired. SHAP may be unstable for token classification.

- **Price extraction refinement:**  
  Expand regex or use the NER model’s PRICE tags when performance improves. Handle different currency formats or multilingual numerals.

- **Views or engagement data:**  
  If Telegram API provides view counts or reaction data in messages, extend the ingestion pipeline to capture these fields and incorporate into vendor scoring.

- **Scheduling & automation:**  
  Use cron jobs or background tasks to run `run_pipeline.py` periodically for fresh data ingestion. Save intermediate state or use a database if scaling beyond JSON files.

- **Error handling & logging:**  
  Expand try/except blocks in ingestion to robustly handle network failures, invalid channels, or authentication issues. Use Python’s logging module for configurable verbosity.

- **Reproducibility:**  
  Pin exact versions in `requirements.txt`. Use Docker container if sharing with others.

