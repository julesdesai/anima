# Anima

**Breathe life into ideas** — Animate any persona through their writings.

A multi-persona RAG system that creates AI agents grounded in the writing style and knowledge of specific individuals or historical figures.

## Quick Start

```bash
# 1. Install
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# 2. Configure
cp .env.example .env && nano .env  # Add your API keys
nano config.yaml                    # Configure personas

# 3. Start vector database
docker-compose up -d && python scripts/setup_db.py

# 4. Add corpus files
# Create a persona directory and add their writings
mkdir -p data/corpus/your_persona
# Add PDFs, text files, emails, chat logs, etc.

# 5. Ingest corpus
python scripts/ingest_corpus.py --persona your_persona --force

# 6. Animate!
python scripts/chat.py --persona your_persona
```

## Features

- **Multi-Persona System**: Animate multiple distinct personas (yourself, historical figures, philosophers, etc.)
- **Isolated Collections**: Each persona has its own vector database collection
- **Style Grounding**: Responses match the writing style of the source corpus
- **Hybrid RAG**: Combines semantic and keyword search for better retrieval
- **Streaming Responses**: Real-time response generation
- **Multiple LLM Support**: OpenAI (GPT-4o), Claude, DeepSeek, Hermes

## Creating a Persona

1. **Add to config.yaml**:
```yaml
personas:
  heidegger:
    name: "Martin Heidegger"
    corpus_path: "data/corpus/heidegger/"
    collection_name: "persona_heidegger"
    description: "German philosopher, Being and Time"
```

2. **Add corpus files**:
```bash
mkdir -p data/corpus/heidegger
cp ~/being_and_time.pdf data/corpus/heidegger/
```

3. **Ingest and chat**:
```bash
python scripts/ingest_corpus.py --persona heidegger --force
python scripts/chat.py --persona heidegger
```

## Requirements

- Python 3.9+
- Docker
- OpenAI API key (or other LLM provider)
