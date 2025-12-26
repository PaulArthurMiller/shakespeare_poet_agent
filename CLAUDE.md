# Shakespeare Poet - Agentic Scene Generator

## Project Overview
Build a system that generates Shakespearean scenes using only authentic Shakespeare quotes/fragments selected via LLM tool calling. The system takes scene descriptions and character profiles, then constructs dialogue speech-by-speech by intelligently selecting from a vector database of Shakespeare fragments.

## Core Architecture

### Modules to Create

1. **chunker.py** - Shakespeare text chunking with metadata
2. **embeddings_generator.py** - Create vector embeddings for chunks
3. **quote_database.py** - Vector database interface (using ChromaDB)
4. **quote_selector.py** - MCP-style tool for quote retrieval
5. **scene_generator.py** - Main orchestrator that uses LLM + tools
6. **metadata_extractor.py** - Extract rich metadata from Shakespeare source
7. **session_manager.py** - Track used quotes per scene/conversation
8. **main.py** - CLI interface for testing

### Dependencies to Use
- **langchain** or **llama-index** - LLM orchestration and tool calling
- **chromadb** - Vector database (lightweight, no server needed)
- **sentence-transformers** - Embedding generation
- **openai** or **anthropic** SDK - LLM API access
- **pydantic** - Data validation and structured outputs
- **python-dotenv** - Environment management
- **beautifulsoup4** - If parsing HTML Shakespeare sources
- **pandas** - For metadata manipulation

### Dependencies to Avoid
- Heavy frameworks (Django, Flask) - keep it simple CLI for now
- Pinecone, Weaviate - stick with ChromaDB for simplicity
- Custom transformer implementations - use sentence-transformers

## Data Flow
```
Shakespeare Source Texts
    ↓
[Chunker with Metadata] → chunks with rich metadata
    ↓
[Embedding Generator] → vector embeddings
    ↓
[ChromaDB Storage] → queryable vector database
    ↓
Scene Prompt → [Scene Generator + LLM]
    ↓
[Quote Selector Tool] ← queries database, enforces no-repeats
    ↓
Speech by Speech → Assembled Scene
```

## Implementation Steps

### Phase 1: Chunking & Metadata (chunker.py, metadata_extractor.py)

Create semantic chunks with comprehensive metadata:

**Chunk Types:**
- Full lines (verse lines, typically iambic pentameter)
- Phrases (clause-level semantic units)
- Fragments (3-8 word meaningful units)

**Metadata Fields:**
- `chunk_id`: Unique identifier
- `chunk_text`: The actual quote
- `chunk_type`: "full_line" | "phrase" | "fragment"
- `play_title`: Source play
- `act`: Act number
- `scene`: Scene number
- `character`: Speaking character
- `character_type`: "protagonist" | "antagonist" | "comic_relief" | "royalty" | "commoner" | etc.
- `emotional_tone`: ["joyful", "melancholy", "angry", "fearful", "loving", etc.] - can be multi-label
- `themes`: ["love", "power", "betrayal", "nature", "death", "fate", etc.] - multi-label
- `speaking_to`: Character(s) being addressed (if identifiable)
- `context`: "soliloquy" | "dialogue" | "aside" | "monologue"
- `meter_type`: "iambic_pentameter" | "prose" | "irregular"
- `contains_metaphor`: boolean
- `contains_question`: boolean
- `contains_exclamation`: boolean
- `word_count`: int
- `formality_level`: "high" | "medium" | "low" (based on thee/thou vs you, Latin words, etc.)
- `time_reference`: "past" | "present" | "future" | "timeless" (if discernible)
- `literary_devices`: ["alliteration", "assonance", "imagery", "personification", etc.]

**Chunking Strategy:**
1. Parse Shakespeare source (Folger, MIT, or Project Gutenberg format)
2. Identify line boundaries (verse vs prose)
3. Extract full lines with character attribution
4. For phrases: split on major punctuation (periods, semicolons, question marks, exclamation marks) but keep semantic coherence
5. For fragments: use spaCy or similar to identify noun phrases, verb phrases, prepositional phrases of 3-8 words
6. Extract metadata using pattern matching + optional LLM enhancement

### Phase 2: Embeddings (embeddings_generator.py)

**Process:**
1. Load chunks with metadata
2. Generate embeddings using `sentence-transformers/all-MiniLM-L6-v2` (fast) or `sentence-transformers/all-mpnet-base-v2` (better quality)
3. Store in ChromaDB with metadata as filters
4. Create indices on key metadata fields for fast filtering

**ChromaDB Collection Schema:**
- Collection name: "shakespeare_quotes"
- Embedding dimension: 384 (MiniLM) or 768 (MPNet)
- Metadata: All fields listed above
- Distance metric: Cosine similarity

### Phase 3: Quote Selection Tool (quote_selector.py)

Implement as a tool/function that the LLM can call:
```python
def get_shakespeare_quote(
    semantic_query: str,
    character_type: Optional[List[str]] = None,
    emotional_tone: Optional[List[str]] = None,
    themes: Optional[List[str]] = None,
    context_type: Optional[str] = None,
    chunk_type: Optional[str] = None,
    formality_level: Optional[str] = None,
    exclude_chunk_ids: List[str] = [],  # Already used quotes
    max_results: int = 5
) -> List[Dict]:
    """
    Query Shakespeare database for relevant quotes.
    
    Returns: List of dicts with chunk_text, metadata, and similarity score
    """
```

**Tool Description for LLM:**
Provide the LLM with a clear function schema so it knows when/how to call this tool during scene generation.

### Phase 4: Scene Generator (scene_generator.py)

**Main Orchestration Logic:**

1. **Parse Scene Input:**
   - Scene description (setting, action, emotional arc)
   - Character profiles (names, traits, motivations, emotional states)
   - Play themes (revenge, love, power, etc.)
   - Target length (number of speeches or word count)

2. **Scene Planning (LLM reasoning):**
   - Break scene into speech sequence
   - Determine which character speaks when
   - Identify emotional/thematic needs for each speech

3. **Speech-by-Speech Generation:**
   - For each speech:
     - LLM determines: character, emotional tone, themes, context
     - Calls `get_shakespeare_quote()` multiple times if needed
     - Chains fragments together to form coherent speech
     - Tracks used quotes in session manager
   
4. **Assembly:**
   - Format as scene with character labels
   - Add stage directions if desired
   - Validate no quote repetition

**LLM Prompt Structure:**
```
You are a Shakespeare scene composer. Your task is to create a scene 
using ONLY authentic Shakespeare quotes from the database.

Scene Description: {scene_description}
Characters: {character_profiles}
Themes: {themes}

For each speech:
1. Determine the speaking character and their emotional state
2. Call get_shakespeare_quote() to find appropriate fragments
3. Sequence fragments to form a coherent speech
4. Ensure variety and no repetition

Output the scene in this format:
CHARACTER_NAME
[Quote fragment 1] [Quote fragment 2]...

Use the tool multiple times per speech if needed to build longer dialogue.
```

### Phase 5: Session Management (session_manager.py)

Track used quotes per scene/session to prevent repetition:
```python
class SessionManager:
    def __init__(self):
        self.used_chunk_ids = set()
    
    def mark_used(self, chunk_id: str):
        self.used_chunk_ids.add(chunk_id)
    
    def get_exclusion_list(self) -> List[str]:
        return list(self.used_chunk_ids)
    
    def reset(self):
        self.used_chunk_ids.clear()
```

### Phase 6: CLI Interface (main.py)

Simple interface for testing:
```
$ python main.py --scene "Two lovers meet in secret garden at night" \
                 --characters "Romeo: passionate young man; Juliet: conflicted maiden" \
                 --themes "love,secrecy,danger" \
                 --length 10
```

Should output a formatted scene with character names and quotes.

## Testing Strategy

1. **Unit Tests:**
   - Test chunker on sample Shakespeare text
   - Verify metadata extraction accuracy
   - Test quote selector with known queries

2. **Integration Tests:**
   - Generate a simple 2-character, 4-speech scene
   - Verify no quote repetition
   - Check that quotes match requested themes/emotions

3. **Quality Tests:**
   - Human review: Do selected quotes make sense?
   - Coherence: Do speeches flow logically?
   - Variety: Are quotes from diverse plays/characters?

## Future Enhancements (Post-POC)

- Web interface for scene generation
- Export to screenplay format
- Multi-scene play generation
- Character voice consistency analysis
- Rhyme scheme detection and matching
- Real-time collaboration features

## File Structure
```
shakespeare_poet/
├── src/
│   ├── chunker.py
│   ├── metadata_extractor.py
│   ├── embeddings_generator.py
│   ├── quote_database.py
│   ├── quote_selector.py
│   ├── scene_generator.py
│   ├── session_manager.py
│   └── utils.py
├── data/
│   ├── raw/                    # Shakespeare source texts
│   ├── processed/              # Chunked data with metadata
│   └── embeddings/             # ChromaDB storage
├── tests/
│   ├── test_chunker.py
│   ├── test_quote_selector.py
│   └── test_scene_generator.py
├── examples/
│   └── sample_scenes.txt
├── main.py
├── requirements.txt
├── .env.example
├── CLAUDE.md                   # This file
└── README.md
```

## Environment Variables
```
ANTHROPIC_API_KEY=your_key_here
# or
OPENAI_API_KEY=your_key_here

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_DB_PATH=./data/embeddings
SHAKESPEARE_SOURCE_PATH=./data/raw
```

## Notes for Claude Code

- Prioritize working code over perfection
- Use type hints throughout
- Add docstrings to all functions
- Keep modules focused and single-purpose
- Handle errors gracefully (missing API keys, empty database, etc.)
- Log important decisions (which quotes were selected and why)
- Make it easy to swap LLM providers (OpenAI ↔ Anthropic)
```

---

## Detailed Prompt for Claude Code

Here's what you'd paste into Claude Code:
```
I want to build a "Shakespeare Poet" system that generates dramatic scenes 
using only authentic Shakespeare quotes. Please implement this following the 
architecture in CLAUDE.md.

Key Requirements:

1. CHUNKING SYSTEM (chunker.py + metadata_extractor.py):
   - Create three chunk types: full lines, phrases, and 3-8 word fragments
   - Extract comprehensive metadata for each chunk (see CLAUDE.md for full list)
   - Metadata should include: emotional tone, themes, character type, context, 
     literary devices, formality level, etc.
   - Support parsing standard Shakespeare text formats (Folger, MIT, Gutenberg)

2. EMBEDDING & DATABASE (embeddings_generator.py + quote_database.py):
   - Use sentence-transformers for embeddings (all-MiniLM-L6-v2 for speed)
   - Store in ChromaDB with full metadata as filterable fields
   - Create efficient query methods that combine semantic search + metadata filters

3. QUOTE SELECTOR TOOL (quote_selector.py):
   - Implement as a callable function/tool for LLM
   - Accept parameters: semantic_query, emotional_tone, themes, character_type, 
     context_type, formality_level, exclude_chunk_ids, etc.
   - Return ranked quotes with metadata and similarity scores
   - This is the core "tool" the LLM will call during scene generation

4. SCENE GENERATOR (scene_generator.py):
   - Main orchestration using Anthropic or OpenAI LLM with tool calling
   - Takes: scene description, character profiles, themes
   - Process:
     a. LLM plans the speech sequence
     b. For each speech, LLM calls get_shakespeare_quote() tool
     c. LLM may chain multiple tool calls to build longer speeches
     d. Session manager tracks used quotes to prevent repetition
   - Output: Formatted scene with character names and dialogue

5. SESSION MANAGEMENT (session_manager.py):
   - Track used chunk_ids per scene
   - Provide exclusion list for quote selector
   - Allow reset between scenes

6. CLI INTERFACE (main.py):
   - Accept scene description, characters, themes via command line
   - Output formatted scene to console
   - Example: python main.py --scene "..." --characters "..." --themes "..."

Please create this system module by module, starting with the chunking system. 
Follow the dependencies and structure outlined in CLAUDE.md. Use type hints, 
docstrings, and error handling throughout.

After the core system works, we'll test it by generating a simple 2-character 
scene and iterating from there.
