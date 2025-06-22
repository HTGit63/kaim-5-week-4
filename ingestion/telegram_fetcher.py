# ingestion/telegram_fetcher.py

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from telethon import TelegramClient, errors

# 1. Load .env variables
load_dotenv()  
API_ID   = int(os.getenv('TG_API_ID'))
API_HASH = os.getenv('TG_API_HASH')
SESSION  = 'kaim_week4_session'

# 2. The list of channels to crawl (strip leading '@')
CHANNELS = [
    'ZemenExpress', 'nevacomputer', 'meneshayeofficial', 'ethio_brand_collection',
    'Leyueqa', 'sinayelj', 'Shewabrand', 'helloomarketethiopia',
    'modernshoppingcenter', 'qnashcom', 'Fashiontera', 'kuruwear',
    'gebeyaadama', 'MerttEka', 'forfreemarket', 'classybrands',
    'marakibrand', 'aradabrand2', 'marakisat2', 'belaclassic',
    'AwasMart', 'qnashcom'  # note: qnashcom appears twice in your list; you may dedupe
]

# 3. Initialize Telegram client
client = TelegramClient(SESSION, API_ID, API_HASH)

async def fetch_messages():
    all_msgs = []
    for chan in CHANNELS:
        try:
            print(f"→ Fetching from @{chan}…")
            # iterate messages; change limit as needed
            async for msg in client.iter_messages(chan, limit=10000):
                all_msgs.append({
                    'channel': chan,
                    'id':      msg.id,
                    'date':    msg.date.isoformat(),
                    'sender':  getattr(msg.sender, 'username', None),
                    'text':    msg.message or ""
                })
        except errors.UsernameNotOccupiedError:
            print(f"⚠️  @{chan} not found or you’re not joined—skipping.")
        except Exception as e:
            print(f"⚠️  Error on @{chan}: {e}—skipping.")
    # 4. Save JSON
    ts    = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    fname = f'raw_messages_{ts}.json'
    out   = os.path.join(os.path.dirname(__file__), '..', 'preprocessing', fname)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(all_msgs, f, ensure_ascii=False, indent=2)
    print(f"\n✅  Saved {len(all_msgs)} messages to {fname}")

if __name__ == '__main__':
    with client:
        client.loop.run_until_complete(fetch_messages())
