"""
Main orchestrator for generating Shakespearean scenes using LLM + quote selection.
"""
import os
import json
from typing import List, Dict, Optional, Any
from anthropic import Anthropic
import anthropic

from .quote_selector import QuoteSelector
from .session_manager import SessionManager
from .quote_database import QuoteDatabase
from .embeddings_generator import EmbeddingsGenerator


class SceneGenerator:
    """
    Generate Shakespearean scenes by orchestrating LLM with quote selection tool.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        database: Optional[QuoteDatabase] = None,
        embedding_generator: Optional[EmbeddingsGenerator] = None
    ):
        """
        Initialize the scene generator.

        Args:
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
            model: Model name to use
            database: QuoteDatabase instance (creates new if not provided)
            embedding_generator: EmbeddingsGenerator instance (creates new if not provided)
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided and ANTHROPIC_API_KEY not set")

        self.model = model
        self.client = Anthropic(api_key=self.api_key)

        # Initialize components
        self.database = database or QuoteDatabase()
        self.embedding_generator = embedding_generator or EmbeddingsGenerator()
        self.session_manager = SessionManager()
        self.quote_selector = QuoteSelector(
            database=self.database,
            embedding_generator=self.embedding_generator,
            session_manager=self.session_manager
        )

        # Tool definition for Claude
        self.tools = [self._create_tool_definition()]

    def _create_tool_definition(self) -> Dict:
        """Create the tool definition for Claude's function calling."""
        return {
            "name": "get_shakespeare_quote",
            "description": (
                "Search the Shakespeare quote database for relevant quotes based on semantic meaning "
                "and metadata filters. Returns authentic Shakespeare fragments that match the query. "
                "You MUST use this tool to find quotes - never generate Shakespeare text yourself. "
                "Call this tool multiple times per speech if you need to build longer dialogue."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "semantic_query": {
                        "type": "string",
                        "description": "Natural language description of the desired quote meaning/content"
                    },
                    "emotional_tone": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by emotional tone (e.g., joyful, melancholy, angry, fearful, loving)"
                    },
                    "themes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by themes (e.g., love, death, power, betrayal, nature, fate)"
                    },
                    "context_type": {
                        "type": "string",
                        "description": "Filter by context type: soliloquy, dialogue, aside, or monologue"
                    },
                    "chunk_type": {
                        "type": "string",
                        "description": "Filter by chunk type: full_line, phrase, or fragment"
                    },
                    "formality_level": {
                        "type": "string",
                        "description": "Filter by formality level: high, medium, or low"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of quotes to return (default: 5)"
                    }
                },
                "required": ["semantic_query"]
            }
        }

    def generate_scene(
        self,
        scene_description: str,
        characters: List[Dict[str, str]],
        themes: Optional[List[str]] = None,
        target_length: int = 10,
        max_turns: int = 20
    ) -> Dict:
        """
        Generate a Shakespearean scene.

        Args:
            scene_description: Description of the scene (setting, action, emotional arc)
            characters: List of character dicts with 'name' and 'description'
            themes: List of themes for the scene
            target_length: Target number of speeches
            max_turns: Maximum conversation turns with Claude

        Returns:
            Dictionary with scene text and metadata
        """
        # Reset session for new scene
        self.session_manager.reset()

        # Build the system prompt
        system_prompt = self._build_system_prompt()

        # Build the initial user message
        user_message = self._build_scene_request(
            scene_description=scene_description,
            characters=characters,
            themes=themes,
            target_length=target_length
        )

        # Initialize conversation
        messages = [{"role": "user", "content": user_message}]

        scene_speeches = []
        turn_count = 0

        while turn_count < max_turns:
            turn_count += 1

            # Call Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                tools=self.tools,
                messages=messages
            )

            # Process response
            assistant_message = {"role": "assistant", "content": response.content}
            messages.append(assistant_message)

            # Check for tool use
            tool_uses = [block for block in response.content if block.type == "tool_use"]

            if tool_uses:
                # Execute tool calls
                tool_results = []

                for tool_use in tool_uses:
                    if tool_use.name == "get_shakespeare_quote":
                        # Call the quote selector
                        results = self.quote_selector.select_and_mark_used(**tool_use.input)

                        # Format results for Claude
                        formatted_results = [
                            {
                                "text": r["chunk_text"],
                                "play": r["metadata"].get("play_title"),
                                "character": r["metadata"].get("character"),
                                "emotional_tone": r["metadata"].get("emotional_tone"),
                                "themes": r["metadata"].get("themes")
                            }
                            for r in results
                        ]

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": json.dumps(formatted_results, indent=2)
                        })

                # Add tool results to conversation
                messages.append({"role": "user", "content": tool_results})

            else:
                # No tool use - check if scene is complete
                text_blocks = [block.text for block in response.content if hasattr(block, "text")]
                final_text = "\n".join(text_blocks)

                # Check for completion signal
                if "SCENE COMPLETE" in final_text or len(scene_speeches) >= target_length:
                    break

                # If no tool use and not complete, prompt for continuation
                if turn_count < max_turns:
                    messages.append({
                        "role": "user",
                        "content": "Continue building the scene. Use the get_shakespeare_quote tool to find more quotes."
                    })

        # Extract final scene from conversation
        scene_text = self._extract_scene_from_conversation(messages)

        return {
            "scene_text": scene_text,
            "characters": characters,
            "themes": themes,
            "quotes_used": len(self.session_manager.get_exclusion_list()),
            "session_id": self.session_manager.session_id,
            "metadata": {
                "scene_description": scene_description,
                "target_length": target_length,
                "turn_count": turn_count
            }
        }

    def _build_system_prompt(self) -> str:
        """Build the system prompt for Claude."""
        return """You are a Shakespeare scene composer. Your task is to create dramatic scenes using ONLY authentic Shakespeare quotes from the database.

CRITICAL RULES:
1. You MUST use the get_shakespeare_quote tool to find every piece of dialogue
2. NEVER generate or write Shakespeare-style text yourself - only use quotes from the tool
3. Each speech should be composed of one or more authentic Shakespeare fragments
4. Call the tool multiple times per speech if needed to build longer dialogue
5. Ensure quotes flow naturally together to form coherent speeches
6. Track which character is speaking and maintain their voice/personality
7. Vary your queries to find diverse quotes from different plays/contexts

OUTPUT FORMAT:
For each speech, output in this format:
CHARACTER_NAME
[quote 1] [quote 2] ...

When the scene is complete, end with: SCENE COMPLETE

PROCESS:
1. Understand the scene requirements (setting, characters, themes, emotions)
2. Plan the sequence of speeches
3. For each speech:
   - Determine the character and their emotional state
   - Call get_shakespeare_quote with appropriate semantic query and filters
   - Select from the returned quotes
   - Chain multiple tool calls if needed for longer speeches
4. Assemble the complete scene

Remember: Your creativity is in selecting and combining authentic quotes, not in writing new text."""

    def _build_scene_request(
        self,
        scene_description: str,
        characters: List[Dict[str, str]],
        themes: Optional[List[str]],
        target_length: int
    ) -> str:
        """Build the initial scene request message."""
        characters_str = "\n".join([
            f"- {char['name']}: {char['description']}"
            for char in characters
        ])

        themes_str = ", ".join(themes) if themes else "general"

        return f"""Create a Shakespearean scene with the following requirements:

SCENE DESCRIPTION:
{scene_description}

CHARACTERS:
{characters_str}

THEMES: {themes_str}

TARGET LENGTH: {target_length} speeches

Use the get_shakespeare_quote tool to find appropriate quotes for each speech. Build the scene speech by speech, using multiple tool calls as needed. Remember to vary emotional tones and themes to create a dynamic scene."""

    def _extract_scene_from_conversation(self, messages: List[Dict]) -> str:
        """Extract the final scene text from the conversation history."""
        scene_lines = []

        for message in messages:
            if message["role"] == "assistant":
                for block in message["content"]:
                    if hasattr(block, "text"):
                        # Extract scene content (skip meta-commentary)
                        lines = block.text.split("\n")
                        for line in lines:
                            line = line.strip()
                            # Include character names and dialogue
                            if line and not line.startswith("Let me") and not line.startswith("I'll") and "SCENE COMPLETE" not in line:
                                scene_lines.append(line)

        return "\n".join(scene_lines)

    def format_scene(self, scene_data: Dict) -> str:
        """
        Format scene data into a readable play format.

        Args:
            scene_data: Scene dictionary from generate_scene

        Returns:
            Formatted scene text
        """
        output = []
        output.append("=" * 60)
        output.append("SHAKESPEAREAN SCENE")
        output.append("=" * 60)
        output.append("")

        if "metadata" in scene_data:
            output.append(f"Description: {scene_data['metadata']['scene_description']}")
            output.append("")

        output.append("CHARACTERS:")
        for char in scene_data["characters"]:
            output.append(f"  {char['name']} - {char['description']}")
        output.append("")

        if scene_data.get("themes"):
            output.append(f"Themes: {', '.join(scene_data['themes'])}")
            output.append("")

        output.append("-" * 60)
        output.append(scene_data["scene_text"])
        output.append("-" * 60)
        output.append("")
        output.append(f"Quotes used: {scene_data['quotes_used']}")
        output.append("=" * 60)

        return "\n".join(output)


def main():
    """Example usage of SceneGenerator."""
    # Initialize generator
    generator = SceneGenerator()

    # Define scene
    scene_description = "Two lovers meet in a secret garden at night, torn between passion and duty"

    characters = [
        {
            "name": "ROMEO",
            "description": "A passionate young man, deeply in love but conflicted"
        },
        {
            "name": "JULIET",
            "description": "A young woman, torn between love and family loyalty"
        }
    ]

    themes = ["love", "conflict", "secrecy"]

    # Generate scene
    print("Generating scene...")
    scene = generator.generate_scene(
        scene_description=scene_description,
        characters=characters,
        themes=themes,
        target_length=6
    )

    # Format and print
    formatted = generator.format_scene(scene)
    print(formatted)


if __name__ == "__main__":
    main()
