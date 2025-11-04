# Castor - Quick Start

Get up and running in 5 minutes!

## One-Command Setup

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
- API keys:
  - `ANTHROPIC_API_KEY` (for Claude) OR
  - `DEEPSEEK_API_KEY` (for DeepSeek)
  - `OPENAI_API_KEY` (for embeddings)

## First Query

```bash
python scripts/chat.py -q "What are my thoughts on AI alignment?"
```

## Add Your Own Writing

```bash
# Add files to data/corpus/
cp ~/my_writing.txt data/corpus/documents/

# Re-ingest
python scripts/ingest_corpus.py --force
```


## Model Selection

```bash
# Claude (highest quality)
python scripts/chat.py --model claude

# DeepSeek (most cost-effective)
python scripts/chat.py --model deepseek

# Hermes (self-hosted, requires vLLM setup)
python scripts/chat.py --model hermes
```

## Troubleshooting

**Qdrant not starting?**
```bash
docker-compose restart
```

**No results from search?**
```bash
python scripts/ingest_corpus.py --force
```

**API errors?**
- Check `.env` has correct keys
- Verify keys start with `sk-`

## Example Output

```
You: What are my thoughts on AI alignment?

Thinking...