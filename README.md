# HAI Lab Doppel

## Setup

```bash
# 1. Install
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# 2. Configure (edit with your API keys)
cp .env.example .env && nano .env
Edit config.yaml with your name

# 3. Start
docker-compose up -d && python scripts/setup_db.py

# 4. Ingest sample corpus

Add your files to data/corpus/...
Allowed file types: txt, json (Claude/GPT Conversation exports), .mbox (email exports), pdfs.
Examples: your gmail exports, your pdfs (convert word or powerpoints and add), your AI chat histories, etc.

# 5. Ingest sample corpus
python scripts/ingest_corpus.py

# 6. Chat!
python scripts/chat.py 
```

## What You Need

- Python 3.9+
- Docker
- API keys
