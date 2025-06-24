#!/usr/bin/env python3
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np

# CONFIGURABLE PARAMETERS
FINAL_JSON_PATH = Path(__file__).parent.parent / "final_dataset.json"
OUTPUT_CSV = Path(__file__).parent / "vendor_scorecard.csv"

# Regex pattern to extract price amounts followed by 'ብር' (Ethiopian Birr)
PRICE_PATTERN = re.compile(r"(\d{1,3}(?:[,\d]{0,})?)\s*ብር")

def extract_price(text: str):
    """
    Extract the first numeric amount preceding 'ብር'. 
    Returns float amount or np.nan if not found.
    """
    if not text:
        return np.nan
    match = PRICE_PATTERN.search(text)
    if match:
        num_str = match.group(1).replace(",", "")
        try:
            return float(num_str)
        except:
            return np.nan
    return np.nan

def main():
    # 1. Load final_dataset.json
    if not FINAL_JSON_PATH.exists():
        print(f"❌ {FINAL_JSON_PATH} not found. Run pipeline first.")
        return
    print("Loading data...")
    with open(FINAL_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    # Ensure at least 'channel' and 'date' exist
    if 'channel' not in df.columns or 'date' not in df.columns:
        print("❌ final_dataset.json missing required fields ('channel','date').")
        return

    # 2. Reconstruct 'text' if missing but 'tokens' present
    if 'text' not in df.columns:
        if 'tokens' in df.columns:
            print("No 'text' field found; reconstructing from 'tokens' by joining with spaces.")
            # tokens should be list of strings
            df['text'] = df['tokens'].apply(lambda tok_list: " ".join(tok_list) if isinstance(tok_list, list) else "")
        else:
            print("❌ final_dataset.json missing both 'text' and 'tokens'; cannot extract prices.")
            return

    # 3. Parse dates, extract week
    print("Parsing dates...")
    df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
    df = df.dropna(subset=['date'])
    # Floor to week start (Monday)
    df['week_start'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)

    # 4. Compute posts per week per channel
    print("Computing posting frequency...")
    gp = df.groupby(['channel', 'week_start']).size().reset_index(name='posts_in_week')
    freq = gp.groupby('channel')['posts_in_week'].mean().reset_index(name='posts_per_week')

    # 5. Extract prices per message
    print("Extracting prices...")
    # Ensure text is string
    df['price'] = df['text'].astype(str).apply(extract_price)
    price_stats = df.groupby('channel')['price'].agg(['mean','count','size']).reset_index()
    price_stats.rename(columns={'mean':'avg_price', 'count':'count_with_price', 'size':'total_posts'}, inplace=True)

    # 6. Merge frequency and price stats
    print("Merging stats...")
    stats = pd.merge(freq, price_stats[['channel','avg_price']], on='channel', how='outer')
    stats['posts_per_week'] = stats['posts_per_week'].fillna(0)
    # avg_price remains NaN if no price found

    # 7. Normalize metrics for lending score
    # Posts per week normalization
    min_posts = stats['posts_per_week'].min()
    max_posts = stats['posts_per_week'].max()
    if max_posts > min_posts:
        stats['norm_posts'] = (stats['posts_per_week'] - min_posts) / (max_posts - min_posts)
    else:
        stats['norm_posts'] = 1.0

    # Avg price normalization, filling NaN with 0
    stats['avg_price_filled'] = stats['avg_price'].fillna(0.0)
    min_price = stats['avg_price_filled'].min()
    max_price = stats['avg_price_filled'].max()
    if max_price > min_price:
        stats['norm_price'] = (stats['avg_price_filled'] - min_price) / (max_price - min_price)
    else:
        stats['norm_price'] = 1.0

    WEIGHT_POSTS = 0.6
    WEIGHT_PRICE = 0.4
    stats['lending_score'] = WEIGHT_POSTS * stats['norm_posts'] + WEIGHT_PRICE * stats['norm_price']

    # 8. Sort descending by score
    stats = stats.sort_values(by='lending_score', ascending=False).reset_index(drop=True)

    # 9. Save to CSV
    print(f"Saving results to {OUTPUT_CSV} ...")
    stats[['channel','posts_per_week','avg_price','lending_score']].to_csv(OUTPUT_CSV, index=False)
    print("Done. Top channels by lending_score:")
    print(stats[['channel','posts_per_week','avg_price','lending_score']].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
