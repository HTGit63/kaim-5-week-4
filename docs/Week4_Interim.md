# Week 4 Interim Report

**Project:** KAIM 5 – EthioMart E-commerce Data Extractor  
**Branch:** week4-data-ingestion  
**Date:** 2025-06-22

## 1. Channels Selected  
• ZemenExpress  
• nevacomputer  
• meneshayeofficial  
• ethio_brand_collection  
• Leyueqa  
• sinayelj  
• Shewabrand  
• helloomarketethiopia  
• modernshoppingcenter  
• qnashcom  
*(…and 12 others as per assignment)*

## 2. Ingestion Workflow  
1. **Auth & Setup**  
   - Loaded `TG_API_ID` & `TG_API_HASH` from `.env`.  
   - Initialized `Telethon` client session (`kaim_week4_session`).  
2. **Channel Crawling**  
   - Iterated each channel via `client.iter_messages(limit=10000)`.  
   - Extracted `id`, `date`, `sender`, `text` fields.  
   - Gracefully skipped invalid usernames.  
3. **Output**  
   - Saved combined JSON to `preprocessing/raw_messages_<timestamp>.json`.

## 3. Preprocessing Steps  
1. **Normalization**  
   - Stripped Amharic diacritics (unicode U+135B–U+1363).  
   - Converted Ethiopian numerals (፩–፱) to Latin digits (1–9).  
2. **Cleaning**  
   - Removed URLs, collapsed line breaks, normalized whitespace.  
3. **Tokenization**  
   - Simple `str.split(' ')` into tokens.  
4. **Output**  
   - Saved token lists to `preprocessing/processed_<timestamp>.json`.

## 4. CoNLL Labeling  
- **Entities:** PRODUCT, PRICE, LOCATION (BIO tags).  
- Labeled **30** sample messages in `labeling/labeled_data.conll.txt`.  
- Ran `labeling/validate_conll.py` to ensure correct format.

