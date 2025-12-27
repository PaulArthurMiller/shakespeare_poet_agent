# Shakespeare Poet - Agentic Scene Generator

An AI-powered system that generates Shakespearean dramatic scenes using **only authentic Shakespeare quotes**. The system uses vector embeddings, LLM tool calling, and semantic search to intelligently select and combine real Shakespeare fragments into coherent new scenes.

## Overview

Shakespeare Poet takes scene descriptions and character profiles, then constructs dialogue speech-by-speech by intelligently selecting from a vector database of Shakespeare fragments. The LLM orchestrator never generates fake Shakespeare - it only selects and combines authentic quotes.

### Key Features

- **Authentic Quotes Only**: Every line comes from real Shakespeare texts
- **Rich Metadata**: Chunks include emotional tone, themes, character types, context, and more
- **Semantic Search**: Vector embeddings enable intelligent quote selection
- **No Repetition**: Session tracking ensures quotes aren't reused within a scene
- **LLM Orchestration**: Uses Claude with tool calling to intelligently compose scenes
- **Multiple Granularities**: Full lines, phrases, and 3-8 word fragments

## Architecture

```
Shakespeare Source Texts
    ↓
[Chunker + Metadata Extraction] → chunks with rich metadata
    ↓
[Embedding Generator] → vector embeddings
    ↓
[ChromaDB Storage] → queryable vector database
    ↓
Scene Prompt → [Scene Generator + Claude]
    ↓
[Quote Selector Tool] ← queries database, enforces no-repeats
    ↓
Speech by Speech → Assembled Scene
```

## Installation

### Prerequisites

- Python 3.8+
- Anthropic API key (for Claude)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd shakespeare_poet_agent
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download spaCy language model (optional, for advanced chunking):
```bash
python -m spacy download en_core_web_sm
```

5. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

## Usage

### 1. Setup Database

First, populate the database with Shakespeare texts:

```bash
python main.py setup \
    --source ./data/raw/hamlet.txt \
    --title "Hamlet" \
    --db-path ./data/embeddings
```

You can add multiple plays by running setup multiple times with different source files.

**Where to get Shakespeare texts:**
- [Project Gutenberg](https://www.gutenberg.org/)
- [Folger Shakespeare Library](https://www.folgerdigitaltexts.org/)
- [MIT Shakespeare](http://shakespeare.mit.edu/)

### 2. Generate a Scene

Generate a new Shakespearean scene:

```bash
python main.py generate \
    --scene "Two lovers meet in a secret garden at night" \
    --characters "Romeo: passionate young man; Juliet: conflicted maiden" \
    --themes "love,secrecy,danger" \
    --length 10 \
    --output ./examples/my_scene.txt
```

**Parameters:**
- `--scene`: Description of the scene setting and action
- `--characters`: Character definitions (format: "Name: Description; Name2: Description2")
- `--themes`: Comma-separated themes (e.g., love, death, power, betrayal)
- `--length`: Target number of speeches (default: 10)
- `--output`: Optional output file path

### 3. Query the Database

Search for specific quotes:

```bash
python main.py query \
    --query "contemplating the meaning of existence" \
    --play "Hamlet" \
    --max-results 5
```

**Parameters:**
- `--query`: Semantic search query
- `--play`: Filter by specific play (optional)
- `--character`: Filter by character (optional)
- `--max-results`: Number of results (default: 5)

## Project Structure

```
shakespeare_poet/
├── src/
│   ├── chunker.py                  # Text chunking with metadata
│   ├── metadata_extractor.py       # Extract rich metadata
│   ├── embeddings_generator.py     # Generate embeddings
│   ├── quote_database.py           # ChromaDB interface
│   ├── quote_selector.py           # Quote selection tool
│   ├── scene_generator.py          # Main orchestrator
│   ├── session_manager.py          # Track used quotes
│   └── utils.py                    # Utility functions
├── data/
│   ├── raw/                        # Shakespeare source texts
│   ├── processed/                  # Chunked data with metadata
│   └── embeddings/                 # ChromaDB storage
├── tests/                          # Unit tests
├── examples/                       # Example generated scenes
├── main.py                         # CLI interface
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── CLAUDE.md                       # Detailed architecture docs
└── README.md                       # This file
```

## How It Works

### 1. Chunking & Metadata Extraction

Shakespeare texts are chunked into three granularities:
- **Full lines**: Complete verse lines or sentences
- **Phrases**: Clause-level semantic units
- **Fragments**: 3-8 word meaningful units

Each chunk gets rich metadata:
- Emotional tone (joyful, melancholy, angry, etc.)
- Themes (love, death, power, betrayal, etc.)
- Character type (royalty, comic relief, commoner, etc.)
- Context (soliloquy, dialogue, aside, monologue)
- Formality level (high, medium, low)
- Literary devices (alliteration, imagery, personification)
- And more...

### 2. Vector Embeddings

Chunks are converted to vector embeddings using sentence-transformers, enabling semantic search.

### 3. LLM Orchestration

Claude uses tool calling to:
1. Plan the scene structure
2. For each speech, call `get_shakespeare_quote()` tool
3. Select appropriate quotes based on semantic meaning and metadata
4. Chain multiple quotes together for longer speeches
5. Ensure variety and coherence

### 4. Session Management

Tracks used quotes to prevent repetition within a scene.

## Example Output

```
============================================================
SHAKESPEAREAN SCENE
============================================================

Description: Two lovers meet in secret garden at night

CHARACTERS:
  ROMEO - A passionate young man, deeply in love
  JULIET - A young woman, torn between love and family

Themes: love, secrecy, danger

------------------------------------------------------------
ROMEO
What light through yonder window breaks?
It is my lady, O, it is my love!

JULIET
O Romeo, Romeo! wherefore art thou Romeo?
Deny thy father and refuse thy name.

ROMEO
Call me but love, and I'll be new baptized.
Henceforth I never will be Romeo.
------------------------------------------------------------

Quotes used: 6
============================================================
```

## Configuration

### Environment Variables

Create a `.env` file from `.env.example`:

```bash
# LLM API Key
ANTHROPIC_API_KEY=your_key_here

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Database Path
CHROMA_DB_PATH=./data/embeddings

# Shakespeare Source Path
SHAKESPEARE_SOURCE_PATH=./data/raw

# LLM Model
LLM_MODEL=claude-3-5-sonnet-20241022
```

### Embedding Models

Choose based on your needs:
- `all-MiniLM-L6-v2`: Fast, 384 dimensions (default)
- `all-mpnet-base-v2`: Better quality, 768 dimensions

## Advanced Usage

### Python API

You can use the system programmatically:

```python
from src.scene_generator import SceneGenerator
from src.quote_database import QuoteDatabase
from src.embeddings_generator import EmbeddingsGenerator

# Initialize
db = QuoteDatabase()
embeddings = EmbeddingsGenerator()
generator = SceneGenerator(database=db, embedding_generator=embeddings)

# Generate scene
scene = generator.generate_scene(
    scene_description="A confrontation between rivals",
    characters=[
        {"name": "MACBETH", "description": "Ambitious king"},
        {"name": "MACDUFF", "description": "Vengeful warrior"}
    ],
    themes=["revenge", "power"],
    target_length=8
)

print(generator.format_scene(scene))
```

### Custom Metadata Filters

When querying, you can use rich filters:

```python
from src.quote_selector import QuoteSelector

results = selector.get_shakespeare_quote(
    semantic_query="expressing deep sorrow",
    emotional_tone=["melancholy", "desperate"],
    themes=["death", "loss"],
    context_type="soliloquy",
    chunk_type="full_line",
    formality_level="high",
    max_results=5
)
```

## Testing

Run tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=src tests/
```

## Troubleshooting

### Database is empty
Make sure you've run `setup` command to populate the database with Shakespeare texts.

### API key errors
Ensure your `ANTHROPIC_API_KEY` is set in `.env` file.

### Import errors
Make sure you're running from the project root and have activated the virtual environment.

### Model download issues
First run may download embedding models (~500MB). Ensure good internet connection.

## Future Enhancements

- [ ] Web interface for scene generation
- [ ] Export to screenplay/PDF format
- [ ] Multi-scene play generation
- [ ] Character voice consistency analysis
- [ ] Rhyme scheme detection and matching
- [ ] Support for OpenAI models
- [ ] Real-time collaboration features

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Shakespeare texts from public domain sources
- Sentence-transformers for embeddings
- ChromaDB for vector storage
- Anthropic Claude for LLM orchestration

## Citation

If you use this project in research, please cite:

```
@software{shakespeare_poet,
  title={Shakespeare Poet: Agentic Scene Generator},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/shakespeare_poet}
}
```
