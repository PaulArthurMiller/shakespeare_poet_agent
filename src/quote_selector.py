"""
Quote selector tool for LLM to retrieve Shakespeare quotes from the database.
"""
from typing import List, Dict, Optional, Any
from .quote_database import QuoteDatabase
from .embeddings_generator import EmbeddingsGenerator
from .session_manager import SessionManager


class QuoteSelector:
    """
    Tool for selecting appropriate Shakespeare quotes based on semantic query and metadata filters.
    This is designed to be called by an LLM during scene generation.
    """

    def __init__(
        self,
        database: QuoteDatabase,
        embedding_generator: EmbeddingsGenerator,
        session_manager: Optional[SessionManager] = None
    ):
        """
        Initialize the quote selector.

        Args:
            database: QuoteDatabase instance
            embedding_generator: EmbeddingsGenerator instance
            session_manager: Optional SessionManager for tracking used quotes
        """
        self.database = database
        self.embedding_generator = embedding_generator
        self.session_manager = session_manager or SessionManager()

    def get_shakespeare_quote(
        self,
        semantic_query: str,
        character_type: Optional[List[str]] = None,
        emotional_tone: Optional[List[str]] = None,
        themes: Optional[List[str]] = None,
        context_type: Optional[str] = None,
        chunk_type: Optional[str] = None,
        formality_level: Optional[str] = None,
        play_title: Optional[str] = None,
        exclude_chunk_ids: Optional[List[str]] = None,
        max_results: int = 5
    ) -> List[Dict]:
        """
        Query Shakespeare database for relevant quotes.

        This is the main tool function that the LLM will call during scene generation.

        Args:
            semantic_query: Natural language query describing desired quote meaning
            character_type: Filter by character type (e.g., ["royalty", "comic_relief"])
            emotional_tone: Filter by emotional tone (e.g., ["melancholy", "joyful"])
            themes: Filter by themes (e.g., ["love", "death"])
            context_type: Filter by context ("soliloquy", "dialogue", "aside", "monologue")
            chunk_type: Filter by chunk type ("full_line", "phrase", "fragment")
            formality_level: Filter by formality ("high", "medium", "low")
            play_title: Filter by specific play
            exclude_chunk_ids: List of chunk IDs to exclude (in addition to session exclusions)
            max_results: Maximum number of results to return

        Returns:
            List of quote dictionaries with text, metadata, and similarity scores
        """
        # Build metadata filters
        where_filters = self._build_where_filters(
            character_type=character_type,
            context_type=context_type,
            chunk_type=chunk_type,
            formality_level=formality_level,
            play_title=play_title
        )

        # Get initial results from database
        results = self.database.query_by_text(
            query_text=semantic_query,
            embedding_generator=self.embedding_generator,
            n_results=max_results * 3,  # Get more results to filter
            where=where_filters if where_filters else None
        )

        # Post-filter results based on list-based metadata (emotional_tone, themes)
        filtered_results = self._post_filter_results(
            results=results,
            emotional_tone=emotional_tone,
            themes=themes,
            exclude_chunk_ids=exclude_chunk_ids
        )

        # Limit to max_results
        filtered_results = filtered_results[:max_results]

        return filtered_results

    def _build_where_filters(
        self,
        character_type: Optional[List[str]] = None,
        context_type: Optional[str] = None,
        chunk_type: Optional[str] = None,
        formality_level: Optional[str] = None,
        play_title: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Build ChromaDB where filters from parameters.

        Args:
            character_type: Character type filter
            context_type: Context type filter
            chunk_type: Chunk type filter
            formality_level: Formality level filter
            play_title: Play title filter

        Returns:
            Where filter dictionary or None
        """
        where = {}

        if context_type:
            where['context'] = context_type

        if chunk_type:
            where['chunk_type'] = chunk_type

        if formality_level:
            where['formality_level'] = formality_level

        if play_title:
            where['play_title'] = play_title

        # Note: character_type is handled in post-filtering because it's a list in metadata

        return where if where else None

    def _post_filter_results(
        self,
        results: List[Dict],
        emotional_tone: Optional[List[str]] = None,
        themes: Optional[List[str]] = None,
        exclude_chunk_ids: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Post-filter results based on list-based metadata and exclusions.

        Args:
            results: Initial query results
            emotional_tone: Emotional tone filter
            themes: Themes filter
            exclude_chunk_ids: Chunk IDs to exclude

        Returns:
            Filtered results
        """
        # Combine session exclusions with provided exclusions
        all_exclusions = set(self.session_manager.get_exclusion_list())
        if exclude_chunk_ids:
            all_exclusions.update(exclude_chunk_ids)

        filtered = []

        for result in results:
            chunk_id = result['chunk_id']

            # Skip if excluded
            if chunk_id in all_exclusions:
                continue

            metadata = result['metadata']

            # Filter by emotional_tone
            if emotional_tone:
                result_emotions = metadata.get('emotional_tone', '').split(',')
                if not any(emotion in result_emotions for emotion in emotional_tone):
                    continue

            # Filter by themes
            if themes:
                result_themes = metadata.get('themes', '').split(',')
                if not any(theme in result_themes for theme in themes):
                    continue

            filtered.append(result)

        return filtered

    def select_and_mark_used(
        self,
        semantic_query: str,
        **kwargs
    ) -> List[Dict]:
        """
        Select quotes and automatically mark them as used in the session.

        Args:
            semantic_query: Semantic query
            **kwargs: Additional parameters for get_shakespeare_quote

        Returns:
            List of selected quotes
        """
        results = self.get_shakespeare_quote(semantic_query, **kwargs)

        # Mark all returned quotes as used
        for result in results:
            self.session_manager.mark_used(
                chunk_id=result['chunk_id'],
                metadata={
                    'query': semantic_query,
                    'chunk_text': result['chunk_text']
                }
            )

        return results

    def get_tool_description(self) -> Dict:
        """
        Get the tool description for LLM function calling.

        Returns:
            Tool description dictionary compatible with LLM APIs
        """
        return {
            "name": "get_shakespeare_quote",
            "description": (
                "Search the Shakespeare quote database for relevant quotes based on semantic meaning "
                "and metadata filters. Returns authentic Shakespeare fragments that match the query. "
                "Use this tool to find appropriate quotes for each speech in the scene."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "semantic_query": {
                        "type": "string",
                        "description": "Natural language description of the desired quote meaning/content"
                    },
                    "character_type": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by character type (e.g., royalty, comic_relief, commoner)",
                        "optional": True
                    },
                    "emotional_tone": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by emotional tone (e.g., joyful, melancholy, angry, fearful, loving)",
                        "optional": True
                    },
                    "themes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by themes (e.g., love, death, power, betrayal, nature, fate)",
                        "optional": True
                    },
                    "context_type": {
                        "type": "string",
                        "description": "Filter by context type: soliloquy, dialogue, aside, or monologue",
                        "optional": True
                    },
                    "chunk_type": {
                        "type": "string",
                        "description": "Filter by chunk type: full_line, phrase, or fragment",
                        "optional": True
                    },
                    "formality_level": {
                        "type": "string",
                        "description": "Filter by formality level: high, medium, or low",
                        "optional": True
                    },
                    "play_title": {
                        "type": "string",
                        "description": "Filter by specific Shakespeare play",
                        "optional": True
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of quotes to return (default: 5)",
                        "optional": True
                    }
                },
                "required": ["semantic_query"]
            }
        }


def main():
    """Example usage of QuoteSelector."""
    from quote_database import QuoteDatabase
    from embeddings_generator import EmbeddingsGenerator

    # Initialize components
    db = QuoteDatabase()
    generator = EmbeddingsGenerator()
    session = SessionManager()
    selector = QuoteSelector(db, generator, session)

    # Example query
    results = selector.get_shakespeare_quote(
        semantic_query="contemplating the meaning of existence",
        emotional_tone=["melancholy"],
        themes=["death", "fate"],
        max_results=3
    )

    print(f"\nFound {len(results)} quotes:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['chunk_text']}")
        print(f"   Play: {result['metadata'].get('play_title')}")
        print(f"   Character: {result['metadata'].get('character')}")
        print(f"   Distance: {result['distance']:.4f}")


if __name__ == "__main__":
    main()
