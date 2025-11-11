# Anima

**HAI Lab Anima** - Animate (bring to life) any persona through their writings.

Anima is an inference-only compute system that brings anyone to life from their past writings. It uses advanced agential self-orchestration and RAG to create AI agents that write in the exact style of specific individuals or historical figures. Unlike standard chatbots that *talk about* someone's ideas, Anima produces text that reads as if the person themselves wrote it. The system can be used in two main ways: public figures (e.g. academics and thinkers of the past) can be reanimated, and we can engage with them in simulated dialogue; or, one's personal writing (email, chat histories, written works) can be animated, to create a simulacra of one's own personality. We are working to allow  local embedding and inference, so that this latter is completely private and does not rely on centralised infrastructure.

## Quick Start

**Included personas** (Taylor, Heidegger, Wittgenstein):

```bash
# 1. Install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Add your OpenAI API key
cp .env.example .env
echo "OPENAI_API_KEY=your-key-here" >> .env

# 3. Start Qdrant vector database
docker-compose up -d
python scripts/setup_db.py

# 4. Ingest a persona (e.g., Charles Taylor)
python scripts/ingest_corpus.py --persona taylor --force

# 5. Chat with Taylor
python scripts/chat.py --persona taylor
```

Now ask: *"Tell me about the ethics of inarticulacy"* and watch Anima respond in Taylor's distinctive philosophical style.

**Try other included personas:**
```bash
# Ingest Heidegger
python scripts/ingest_corpus.py --persona heidegger --force
python scripts/chat.py --persona heidegger

# Ingest Wittgenstein
python scripts/ingest_corpus.py --persona wittgenstein --force
python scripts/chat.py --persona wittgenstein
```

## How It Works

Anima uses a **self-orchestrating RAG protocol** where the LLM itself decides retrieval strategy:

### The Two-Stage Retrieval Process

1. **Content Retrieval** (k=80-100): Fetch relevant knowledge about the query topic
2. **Style Retrieval** (k=80-100): Fetch examples of how the persona writes on related topics

This exposes the model to **160-200 text chunks** (~100k+ characters) per response, providing deep immersion in both what they think and how they express it.

### Style Emulation Features

The system captures 5 dimensions of writing style:
- **Sentence Structure**: Length patterns, nested clauses, rhythm
- **Vocabulary**: Characteristic terminology, metaphors, formality level
- **Rhetorical Moves**: How arguments are built, use of questions/qualifications
- **Voice & Tone**: Intellectual personality, emotional register
- **Thought Patterns**: How ideas connect, conceptual frameworks

### Architecture

```
User Query
    ↓
System Prompt (includes 15 baseline style samples)
    ↓
LLM decides to search
    ↓
Hybrid Search (semantic + keyword, RRF fusion)
    ↓
80-100 content chunks + 80-100 style chunks
    ↓
LLM generates response in persona's voice
```

Each persona has an **isolated Qdrant collection** (e.g., `persona_taylor`, `persona_heidegger`) ensuring no data leakage.

## Creating Your Own Persona

Add anyone—yourself, a colleague, a historical figure, a fictional character:

### 1. Add to `config.yaml`

```yaml
personas:
  dostoyevsky:
    name: "Fyodor Dostoyevsky"
    corpus_path: "data/corpus/dostoyevsky/"
    collection_name: "persona_dostoyevsky"
    description: "Russian novelist, Crime and Punishment"
    chunk_size: 1000  # Optional: override for narrative text
```

### 2. Gather Corpus Files

**Directory structure:**
```
data/corpus/{persona_id}/
├── audio/              # Audio samples for voice cloning (optional)
│   ├── sample1.mp3
│   └── sample2.wav
├── writings.txt        # Text corpus files
├── papers.pdf
└── emails.mbox
```

Supported formats:
- **PDFs** (`.pdf`) - Books, papers, articles
- **Text files** (`.txt`, `.md`) - Any written content
- **Emails** (`.mbox`) - Gmail/Outlook exports
- **Chat logs** (`.json`) - Claude/GPT conversation exports
- **Audio** (`.mp3`, `.wav`, `.m4a`, etc.) - For voice cloning (optional)

```bash
# Create directory structure
mkdir -p data/corpus/dostoyevsky/audio

# Add text corpus
cp ~/crime_and_punishment.pdf data/corpus/dostoyevsky/
cp ~/brothers_karamazov.pdf data/corpus/dostoyevsky/
cp ~/notes_from_underground.txt data/corpus/dostoyevsky/

# Add audio for voice cloning (optional)
cp ~/dostoyevsky_reading.mp3 data/corpus/dostoyevsky/audio/
```

### 3. Ingest and Animate

```bash
python scripts/ingest_corpus.py --persona dostoyevsky --force
python scripts/chat.py --persona dostoyevsky
```

**For your own writing**, export:
- Gmail as `.mbox` (Google Takeout)
- ChatGPT/Claude conversations as `.json`
- Your documents as PDFs or text files
- Voice recordings as audio files (for TTS)

## Voice Cloning (Text-to-Speech)

Anima can speak in the persona's voice using either free local TTS or cloud-based services.

### Setup TTS

**1. Install TTS dependencies:**
```bash
pip install -r requirements-tts.txt
```

**2. Choose provider in `config.yaml`:**
```yaml
tts:
  enabled: true
  provider: "local"  # Options: "local" (free, Coqui XTTS-v2) or "elevenlabs" (paid)
  auto_play: true    # Automatically play audio
```

**3. Add audio sample to persona directory:**
```bash
# Add 6-30 seconds of clear speech
cp voice_sample.mp3 data/corpus/{persona_id}/audio/
```

**4. Clone the voice:**
```bash
# Automatically finds audio in data/corpus/{persona}/audio/
python scripts/clone_voice.py --persona dostoyevsky --auto-discover

# Or specify audio files explicitly
python scripts/clone_voice.py --persona dostoyevsky \
  --audio-files path/to/audio.mp3
```

**5. Enable voice for persona in `config.yaml`:**
```yaml
personas:
  dostoyevsky:
    voice_enabled: true  # Script sets this automatically
    voice_id: "dostoyevsky"  # Set by clone_voice.py
```

**6. Chat with voice:**
```bash
python scripts/chat.py --persona dostoyevsky
# Responses will be spoken out loud!
```

### TTS Providers Comparison

| Provider | Cost | Quality | Cloning | Speed |
|----------|------|---------|---------|-------|
| **Local** (Coqui XTTS-v2) | Free | High | ✅ Yes | Medium |
| **ElevenLabs** | Paid API | Very High | ✅ Yes | Fast |

**Local TTS** (recommended):
- Free and open source
- Voice cloning from 6+ seconds of audio
- Runs entirely offline
- ~2GB model download on first use

**ElevenLabs**:
- Requires API key: `export ELEVENLABS_API_KEY="your-key"`
- Very high quality professional voices
- Paid service ($5-$11/month)

### Audio Requirements

For best voice cloning results:
- **Duration**: 6-30 seconds (local), 1-3 minutes (ElevenLabs)
- **Quality**: Clear speech, single speaker
- **Format**: MP3, WAV, M4A, OGG, FLAC, AAC
- **Content**: Natural conversation or reading
- **Environment**: Quiet, minimal background noise

## Requirements

- **Python 3.9+**
- **Docker** (for Qdrant vector database)
- **OpenAI API key** (or Claude/DeepSeek/Hermes)
- ~2GB disk space per persona (for embeddings)
- **TTS (optional)**: Additional 2GB for Coqui XTTS-v2 model

## CLI Usage

### Chat Interface

```bash
# Basic usage
python scripts/chat.py --persona taylor

# Use different LLM model
python scripts/chat.py --persona heidegger --model claude

# Non-interactive single query
python scripts/chat.py --persona wittgenstein --query "What is the meaning of a word?"

# Save conversation history
python scripts/chat.py --persona taylor --save-history

# Debug mode (show retrieval stats)
python scripts/chat.py --persona taylor --debug
```

**Commands during chat:**
- `exit` / `quit` - End session
- `clear` - Reset conversation history
- `history` - Show conversation history

### Ingestion

```bash
# Full ingestion with force recreate
python scripts/ingest_corpus.py --persona taylor --force

# Incremental ingestion (only new files)
python scripts/ingest_corpus.py --persona taylor

# Custom directory
python scripts/ingest_corpus.py --persona taylor --directory ~/my_texts/

# Non-recursive (single directory only)
python scripts/ingest_corpus.py --persona taylor --no-recursive
```

## Configuration

Key settings in `config.yaml`:

```yaml
# Retrieval tuning
retrieval:
  default_k: 80              # Chunks per search
  max_k: 120                 # Maximum allowed
  similarity_threshold: 0.5  # Minimum cosine similarity
  style_pack_size: 15        # Baseline style samples

# Corpus processing
corpus:
  chunk_size: 800            # Characters per chunk
  chunk_overlap: 100         # Overlap between chunks

# Per-persona overrides
personas:
  philosopher:
    chunk_size: 1200  # Larger chunks for dense text
```

## Features

- **Multi-Persona System**: Unlimited personas with complete isolation
- **Voice Cloning**: Free local TTS or cloud-based voice synthesis
- **Style Grounding**: 15 baseline samples + 160-200 chunks per response
- **Hybrid RAG**: Semantic (text-embedding-3-large) + keyword search with RRF fusion
- **Streaming Responses**: Real-time generation with Rich terminal UI
- **Multiple LLMs**: OpenAI (GPT-4o), Claude Sonnet, DeepSeek, Hermes
- **Self-Orchestrating**: LLM decides retrieval strategy, no hardcoded rules
- **Conversation History**: Multi-turn dialogues with context preservation

## Use Cases

- **Personal Knowledge Base**: Animate yourself to recall your own writing and thoughts
- **Historical Research**: Converse with historical figures in their authentic voice
- **Literary Analysis**: Engage with authors in their distinctive style
- **Philosophy Education**: Dialogue with philosophers as if they were present
- **Creative Writing**: Collaborate with fictional characters or real authors
- **Academic Research**: Query scholars' work in their own argumentative style

## Technical Details

**Embeddings**: OpenAI `text-embedding-3-large` (3072 dimensions)
**Vector DB**: Qdrant with hybrid search (semantic + keyword, RRF fusion)
**Context Window**: Up to 128k tokens (GPT-4o)
**Retrieval**: 160-200 chunks per response (~100k chars, ~25k tokens)
**Protocol**: Natural language specification, self-orchestrating

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Areas of interest:
- Additional LLM providers
- Enhanced style analysis metrics
- Alternative embedding models
- Performance optimizations

## Acknowledgments

Inspired by the concept of **heteronyms** (Fernando Pessoa) - distinct literary personas with their own voice, style, and worldview.
