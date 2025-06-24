
"""
Orchestrate full pipeline: ingestion → preprocessing → produce final_dataset.json.
Usage:
    python run_pipeline.py
Steps:
 1. Run telegram_fetcher to fetch raw JSON
 2. Run preprocess to clean/tokenize
 3. Aggregate processed files into final_dataset.json
Logs counts at each step.
"""


import subprocess
import logging
import sys
from pathlib import Path
import json

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

ROOT          = Path(__file__).parent
INGEST_SCRIPT = ROOT / 'ingestion' / 'telegram_fetcher.py'
PREP_SCRIPT   = ROOT / 'preprocessing' / 'preprocess.py'
PREP_DIR      = ROOT / 'preprocessing'
FINAL_JSON    = ROOT / 'final_dataset.json'

def run_command(cmd):
    """Run a command, streaming stdout/stderr live."""
    logger.info(f"Running: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

def load_latest_processed():
    files = sorted(PREP_DIR.glob('processed_*.json'))
    if not files:
        raise FileNotFoundError("No processed_*.json found—did preprocessing run?")
    return files[-1]

def main():
    # Step 1: Ingestion
    logger.info("STEP 1: Data Ingestion")
    run_command([sys.executable, str(INGEST_SCRIPT)])

    # Step 2: Preprocessing
    logger.info("STEP 2: Data Preprocessing")
    run_command([sys.executable, str(PREP_SCRIPT)])

    # Step 3: Consolidate
    proc_file = load_latest_processed()
    logger.info(f"STEP 3: Loading processed data from {proc_file.name}")
    data = json.loads(proc_file.read_text(encoding='utf-8'))

    FINAL_JSON.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    logger.info(f"Saved final dataset with {len(data)} messages to {FINAL_JSON.name}")

if __name__ == '__main__':
    main()
