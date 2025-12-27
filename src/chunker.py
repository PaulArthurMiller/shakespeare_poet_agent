"""
Shakespeare text chunking with metadata extraction.
"""
import re
import json
import hashlib
from typing import List, Dict, Optional
from pathlib import Path

from .utils import clean_text, extract_character_name, normalize_character_name
from .metadata_extractor import MetadataExtractor


class ShakespeareChunker:
    """
    Chunk Shakespeare texts into semantically meaningful units with metadata.
    Supports multiple chunk types: full_line, phrase, fragment.
    """

    def __init__(self):
        self.metadata_extractor = MetadataExtractor()
        self.chunks = []

    def chunk_play(
        self,
        play_text: str,
        play_title: str,
        chunk_types: List[str] = ["full_line", "phrase", "fragment"]
    ) -> List[Dict]:
        """
        Chunk a complete Shakespeare play into multiple granularities.

        Args:
            play_text: Raw play text
            play_title: Title of the play
            chunk_types: Types of chunks to create

        Returns:
            List of chunk dictionaries with metadata
        """
        chunks = []

        # Parse play structure
        acts = self._parse_acts(play_text)

        for act_num, act_text in enumerate(acts, 1):
            scenes = self._parse_scenes(act_text)

            for scene_num, scene_text in enumerate(scenes, 1):
                # Extract character speeches
                speeches = self._parse_speeches(scene_text)

                for character, speech_text in speeches:
                    # Determine context (soliloquy, dialogue, etc.)
                    context = self._determine_context(speeches, character)

                    # Create chunks of different types
                    if "full_line" in chunk_types:
                        line_chunks = self._create_line_chunks(
                            speech_text, play_title, act_num, scene_num, character, context
                        )
                        chunks.extend(line_chunks)

                    if "phrase" in chunk_types:
                        phrase_chunks = self._create_phrase_chunks(
                            speech_text, play_title, act_num, scene_num, character, context
                        )
                        chunks.extend(phrase_chunks)

                    if "fragment" in chunk_types:
                        fragment_chunks = self._create_fragment_chunks(
                            speech_text, play_title, act_num, scene_num, character, context
                        )
                        chunks.extend(fragment_chunks)

        self.chunks = chunks
        return chunks

    def _parse_acts(self, play_text: str) -> List[str]:
        """Parse play into acts."""
        # Pattern for ACT markers
        act_pattern = r'ACT\s+[IVX]+|Act\s+\d+'
        acts = re.split(act_pattern, play_text)
        # Remove empty and very short sections
        acts = [act.strip() for act in acts if len(act.strip()) > 100]
        return acts if acts else [play_text]  # If no acts found, treat whole text as one act

    def _parse_scenes(self, act_text: str) -> List[str]:
        """Parse act into scenes."""
        # Pattern for SCENE markers
        scene_pattern = r'SCENE\s+[IVX]+|Scene\s+\d+'
        scenes = re.split(scene_pattern, act_text)
        # Remove empty and very short sections
        scenes = [scene.strip() for scene in scenes if len(scene.strip()) > 50]
        return scenes if scenes else [act_text]  # If no scenes found, treat whole act as one scene

    def _parse_speeches(self, scene_text: str) -> List[tuple]:
        """
        Parse scene into character speeches.

        Returns:
            List of (character_name, speech_text) tuples
        """
        speeches = []
        lines = scene_text.split('\n')

        current_character = None
        current_speech = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this is a character name line
            character_name = extract_character_name(line)

            if character_name:
                # Save previous speech if exists
                if current_character and current_speech:
                    speech_text = ' '.join(current_speech)
                    speeches.append((current_character, speech_text))

                # Start new speech
                current_character = normalize_character_name(character_name)
                current_speech = []
            elif current_character:
                # Add to current speech
                current_speech.append(line)

        # Don't forget the last speech
        if current_character and current_speech:
            speech_text = ' '.join(current_speech)
            speeches.append((current_character, speech_text))

        return speeches

    def _determine_context(self, speeches: List[tuple], character: str) -> str:
        """
        Determine the context of a speech (soliloquy, dialogue, etc.).

        Args:
            speeches: All speeches in the scene
            character: Current character

        Returns:
            Context type
        """
        # If only one character speaks, it's likely a soliloquy
        characters = set(char for char, _ in speeches)
        if len(characters) == 1:
            return "soliloquy"
        else:
            return "dialogue"

    def _create_line_chunks(
        self,
        speech_text: str,
        play_title: str,
        act: int,
        scene: int,
        character: str,
        context: str
    ) -> List[Dict]:
        """Create full-line chunks."""
        chunks = []

        # Split into lines (by sentence or verse line)
        lines = self._split_into_lines(speech_text)

        for line in lines:
            line = clean_text(line)
            if len(line.split()) < 3:  # Skip very short lines
                continue

            chunk_id = self._generate_chunk_id(line, play_title, character)
            metadata = self.metadata_extractor.extract_metadata(
                chunk_text=line,
                play_title=play_title,
                act=act,
                scene=scene,
                character=character,
                chunk_type="full_line",
                context=context
            )
            metadata['chunk_id'] = chunk_id
            chunks.append(metadata)

        return chunks

    def _create_phrase_chunks(
        self,
        speech_text: str,
        play_title: str,
        act: int,
        scene: int,
        character: str,
        context: str
    ) -> List[Dict]:
        """Create phrase-level chunks."""
        chunks = []

        # Split by major punctuation
        phrases = re.split(r'[.!?;]', speech_text)

        for phrase in phrases:
            phrase = clean_text(phrase)
            if len(phrase.split()) < 3:  # Skip very short phrases
                continue

            chunk_id = self._generate_chunk_id(phrase, play_title, character)
            metadata = self.metadata_extractor.extract_metadata(
                chunk_text=phrase,
                play_title=play_title,
                act=act,
                scene=scene,
                character=character,
                chunk_type="phrase",
                context=context
            )
            metadata['chunk_id'] = chunk_id
            chunks.append(metadata)

        return chunks

    def _create_fragment_chunks(
        self,
        speech_text: str,
        play_title: str,
        act: int,
        scene: int,
        character: str,
        context: str
    ) -> List[Dict]:
        """
        Create fragment-level chunks (3-8 word meaningful units).
        Uses a sliding window approach.
        """
        chunks = []
        words = speech_text.split()

        # Sliding window for 3-8 word fragments
        for window_size in [8, 6, 5, 4, 3]:
            for i in range(len(words) - window_size + 1):
                fragment = ' '.join(words[i:i + window_size])
                fragment = clean_text(fragment)

                # Skip if fragment is just common words
                if self._is_meaningful_fragment(fragment):
                    chunk_id = self._generate_chunk_id(fragment, play_title, character)
                    metadata = self.metadata_extractor.extract_metadata(
                        chunk_text=fragment,
                        play_title=play_title,
                        act=act,
                        scene=scene,
                        character=character,
                        chunk_type="fragment",
                        context=context
                    )
                    metadata['chunk_id'] = chunk_id
                    chunks.append(metadata)

        return chunks

    def _split_into_lines(self, text: str) -> List[str]:
        """Split text into verse lines or sentences."""
        # Try to preserve verse line breaks if present
        if '\n' in text:
            lines = text.split('\n')
        else:
            # Split by sentence
            lines = re.split(r'[.!?]+', text)

        return [line.strip() for line in lines if line.strip()]

    def _is_meaningful_fragment(self, fragment: str) -> bool:
        """
        Check if a fragment is meaningful (not just common words).

        Args:
            fragment: Text fragment

        Returns:
            True if meaningful, False otherwise
        """
        # Skip fragments that are only common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were'}
        words = set(fragment.lower().split())

        # At least one non-common word
        return len(words - common_words) > 0

    def _generate_chunk_id(self, text: str, play_title: str, character: str) -> str:
        """Generate unique chunk ID based on content."""
        content = f"{play_title}:{character}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    def save_chunks(self, output_path: str) -> None:
        """
        Save chunks to JSON file.

        Args:
            output_path: Path to output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)

    def load_chunks(self, input_path: str) -> List[Dict]:
        """
        Load chunks from JSON file.

        Args:
            input_path: Path to input file

        Returns:
            List of chunk dictionaries
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        return self.chunks
