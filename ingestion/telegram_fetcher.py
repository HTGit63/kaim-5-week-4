# ingestion/telegram_fetcher.py

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from telethon import TelegramClient, errors

# Load credentials
load_dotenv()
API_ID   = int(os.getenv('TG_API_ID'))
API_HASH = os.getenv('TG_API_HASH')
SESSION  = 'kaim_week4_session'

# Channels to crawl
CHANNELS = [
    'ZemenExpress', 'nevacomputer', 'meneshayeofficial', 'ethio_brand_collection',
    'Leyueqa', 'sinayelj', 'marakisat2', 'belaclassic',
    'AwasMart', 'qnashcom'
]

client = TelegramClient(SESSION, API_ID, API_HASH)

async def fetch_messages():
    all_msgs = []
    for chan in CHANNELS:
        try:
            print(f"→ Fetching from @{chan}…")
            async for msg in client.iter_messages(chan, limit=10000):
                all_msgs.append({
                    'channel': chan,
                    'id':      msg.id,
                    'date':    msg.date.isoformat(),
                    'sender':  getattr(msg.sender, 'username', None),
                    'text':    msg.message or ""
                })
        except errors.UsernameNotOccupiedError:
            print(f"⚠️  @{chan} not found—skipping.")
        except Exception as e:
            print(f"⚠️  Error on @{chan}: {e}—skipping.")

    ts    = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    fname = f'raw_messages_{ts}.json'
    out   = os.path.join(os.path.dirname(__file__), '..', 'preprocessing', fname)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(all_msgs, f, ensure_ascii=False, indent=2)
    print(f"\n✅  Saved {len(all_msgs)} messages to {fname}")

if __name__ == '__main__':
    # This prompts once for your phone/code and then reuses the session file
    client.start()
    client.loop.run_until_complete(fetch_messages())
