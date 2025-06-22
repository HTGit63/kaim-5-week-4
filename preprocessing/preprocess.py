# preprocessing/preprocess.py

import re
import json
from pathlib import Path
from datetime import datetime

# 1. Helper: Amharic-specific normalization (strip diacritics, standardize numbers)
def normalize_amharic(text: str) -> str:
    text = re.sub(r'[\u135B-\u1363]', '', text)  # strip common diacritics
    ethi_nums = '፩፪፫፬፭፮፯፰፱'
    for i, en in enumerate('123456789'):
        text = text.replace(ethi_nums[i], en)
    return text

# 2. Cleaning: strip URLs, extra whitespace, control chars
def clean_text(text: str) -> str:
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[\r\n]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

# 3. Tokenization: simple whitespace split
def tokenize(text: str) -> list[str]:
    return text.split(' ')

def main():
    # where this script lives
    base = Path(__file__).parent

    # Find all raw JSON files in this same folder
    raw_files = sorted(base.glob('raw_messages_*.json'))
    if not raw_files:
        print("❌ No raw_messages_*.json in preprocessing/. Run the fetcher first.")
        return

    raw_path = raw_files[-1]
    print(f"→ Preprocessing {raw_path.name}")

    # Load
    msgs = json.loads(raw_path.read_text(encoding='utf-8'))

    processed = []
    for m in msgs:
        txt = m.get('text') or ''
        txt = normalize_amharic(txt)
        txt = clean_text(txt)
        if not txt:
            continue
        tokens = tokenize(txt)
        processed.append({
            'channel': m['channel'],
            'id':      m['id'],
            'date':    m['date'],
            'tokens':  tokens
        })

    # Save processed JSON alongside the raw files
    out_name = f'processed_{datetime.utcnow():%Y%m%d%H%M%S}.json'
    out_path = base / out_name
    out_path.write_text(json.dumps(processed, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"✅ Saved {len(processed)} processed messages to {out_name}")

if __name__ == '__main__':
    main()
