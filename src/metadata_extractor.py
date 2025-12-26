"""
Extract rich metadata from Shakespeare text chunks.
"""
import re
from typing import List, Dict, Optional
from .utils import (
    has_question, has_exclamation, detect_formality,
    count_syllables, is_verse
)


class MetadataExtractor:
    """Extract metadata from Shakespeare text chunks."""

    # Common themes in Shakespeare
    THEMES = {
        'love': ['love', 'heart', 'affection', 'passion', 'beloved', 'romance'],
        'death': ['death', 'die', 'dead', 'grave', 'tomb', 'mortality'],
        'power': ['power', 'king', 'queen', 'throne', 'crown', 'rule', 'command'],
        'betrayal': ['betray', 'traitor', 'false', 'deceit', 'treachery'],
        'nature': ['nature', 'earth', 'sky', 'sun', 'moon', 'star', 'flower', 'tree'],
        'fate': ['fate', 'fortune', 'destiny', 'star', 'doom'],
        'revenge': ['revenge', 'vengeance', 'avenge', 'retribution'],
        'honor': ['honor', 'noble', 'virtue', 'worthy', 'dignity'],
        'madness': ['mad', 'insane', 'crazy', 'lunacy', 'wit'],
        'time': ['time', 'hour', 'day', 'night', 'moment', 'age'],
        'jealousy': ['jealous', 'envy', 'envious', 'green-eyed'],
        'ambition': ['ambition', 'aspire', 'desire', 'seek'],
    }

    # Emotional tones
    EMOTIONS = {
        'joyful': ['joy', 'happy', 'merry', 'delight', 'glad', 'pleasure'],
        'melancholy': ['sad', 'sorrow', 'grief', 'woe', 'melancholy', 'heavy'],
        'angry': ['angry', 'rage', 'fury', 'wrath', 'mad', 'fierce'],
        'fearful': ['fear', 'afraid', 'terror', 'dread', 'fright'],
        'loving': ['love', 'dear', 'sweet', 'gentle', 'tender', 'fond'],
        'desperate': ['desperate', 'despair', 'hopeless', 'wretched'],
        'prideful': ['proud', 'pride', 'vain', 'glory', 'boast'],
        'contemptuous': ['scorn', 'contempt', 'despise', 'mock', 'disdain'],
    }

    # Literary devices patterns
    LITERARY_DEVICES = {
        'alliteration': r'\b(\w)\w*\s+\1\w*',
        'metaphor': ['like', 'as'],  # Simple heuristic
    }

    def extract_metadata(
        self,
        chunk_text: str,
        play_title: str,
        act: int,
        scene: int,
        character: str,
        chunk_type: str,
        context: str = "dialogue"
    ) -> Dict:
        """
        Extract comprehensive metadata from a text chunk.

        Args:
            chunk_text: The text chunk
            play_title: Source play name
            act: Act number
            scene: Scene number
            character: Speaking character
            chunk_type: "full_line", "phrase", or "fragment"
            context: "soliloquy", "dialogue", "aside", or "monologue"

        Returns:
            Dictionary with all metadata fields
        """
        metadata = {
            'chunk_text': chunk_text,
            'chunk_type': chunk_type,
            'play_title': play_title,
            'act': act,
            'scene': scene,
            'character': character,
            'context': context,
            'emotional_tone': self._extract_emotions(chunk_text),
            'themes': self._extract_themes(chunk_text),
            'meter_type': self._detect_meter(chunk_text),
            'contains_metaphor': self._contains_metaphor(chunk_text),
            'contains_question': has_question(chunk_text),
            'contains_exclamation': has_exclamation(chunk_text),
            'word_count': len(chunk_text.split()),
            'formality_level': detect_formality(chunk_text),
            'time_reference': self._detect_time_reference(chunk_text),
            'literary_devices': self._detect_literary_devices(chunk_text),
            'character_type': self._infer_character_type(character, play_title),
        }

        return metadata

    def _extract_emotions(self, text: str) -> List[str]:
        """Extract emotional tones from text."""
        text_lower = text.lower()
        emotions = []

        for emotion, keywords in self.EMOTIONS.items():
            if any(keyword in text_lower for keyword in keywords):
                emotions.append(emotion)

        return emotions if emotions else ['neutral']

    def _extract_themes(self, text: str) -> List[str]:
        """Extract themes from text."""
        text_lower = text.lower()
        themes = []

        for theme, keywords in self.THEMES.items():
            if any(keyword in text_lower for keyword in keywords):
                themes.append(theme)

        return themes if themes else ['general']

    def _detect_meter(self, text: str) -> str:
        """
        Detect meter type (iambic pentameter vs prose).

        Args:
            text: Text to analyze

        Returns:
            "iambic_pentameter", "prose", or "irregular"
        """
        if not is_verse(text):
            return "prose"

        # Count syllables
        words = text.split()
        syllable_count = sum(count_syllables(word) for word in words)

        # Iambic pentameter has ~10 syllables per line
        if 8 <= syllable_count <= 12:
            return "iambic_pentameter"
        else:
            return "irregular"

    def _contains_metaphor(self, text: str) -> bool:
        """
        Simple heuristic to detect metaphors.
        More sophisticated detection would require NLP.
        """
        text_lower = text.lower()
        # Simple check for comparison words
        metaphor_words = ['like', 'as', '似', 'than']
        return any(word in text_lower for word in metaphor_words)

    def _detect_time_reference(self, text: str) -> str:
        """
        Detect time reference in text.

        Returns:
            "past", "present", "future", or "timeless"
        """
        text_lower = text.lower()

        past_words = ['was', 'were', 'had', 'did', 'ago', 'yesterday', 'once']
        future_words = ['will', 'shall', 'tomorrow', 'hereafter', 'future']
        present_words = ['is', 'am', 'are', 'now', 'today']

        past_count = sum(1 for word in past_words if word in text_lower)
        future_count = sum(1 for word in future_words if word in text_lower)
        present_count = sum(1 for word in present_words if word in text_lower)

        if past_count > future_count and past_count > present_count:
            return "past"
        elif future_count > past_count and future_count > present_count:
            return "future"
        elif present_count > 0:
            return "present"
        else:
            return "timeless"

    def _detect_literary_devices(self, text: str) -> List[str]:
        """Detect literary devices in text."""
        devices = []

        # Check for alliteration
        if re.search(self.LITERARY_DEVICES['alliteration'], text):
            devices.append('alliteration')

        # Check for imagery (words related to senses)
        imagery_words = ['see', 'hear', 'smell', 'taste', 'touch', 'feel', 'look', 'sound']
        if any(word in text.lower() for word in imagery_words):
            devices.append('imagery')

        # Check for personification (simple heuristic)
        personification_words = ['speaks', 'weeps', 'laughs', 'smiles']
        if any(word in text.lower() for word in personification_words):
            devices.append('personification')

        return devices if devices else ['none']

    def _infer_character_type(self, character: str, play_title: str) -> str:
        """
        Infer character type based on name and play.
        This is a simplified heuristic - real implementation would use a knowledge base.

        Returns:
            Character type like "protagonist", "antagonist", "comic_relief", etc.
        """
        character_lower = character.lower()

        # Simple heuristics
        if any(title in character_lower for title in ['king', 'queen', 'prince', 'princess', 'duke', 'duchess']):
            return "royalty"
        elif any(word in character_lower for word in ['fool', 'clown', 'servant']):
            return "comic_relief"
        elif any(word in character_lower for word in ['lord', 'lady', 'sir']):
            return "nobility"
        else:
            return "commoner"
