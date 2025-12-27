"""
Shakespeare Poet - Agentic Scene Generator

A system that generates Shakespearean scenes using only authentic Shakespeare quotes
selected via LLM tool calling.
"""

__version__ = "0.1.0"

from .chunker import ShakespeareChunker
from .metadata_extractor import MetadataExtractor
from .embeddings_generator import EmbeddingsGenerator
from .quote_database import QuoteDatabase
from .quote_selector import QuoteSelector
from .scene_generator import SceneGenerator
from .session_manager import SessionManager

__all__ = [
    "ShakespeareChunker",
    "MetadataExtractor",
    "EmbeddingsGenerator",
    "QuoteDatabase",
    "QuoteSelector",
    "SceneGenerator",
    "SessionManager",
]
