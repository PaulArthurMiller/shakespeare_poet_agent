"""
Utility functions for the Shakespeare Poet system.
"""
import os
import re
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv


def load_env() -> None:
    """Load environment variables from .env file."""
    load_dotenv()


def get_env_var(key: str, default: Optional[str] = None) -> str:
    """
    Get environment variable with optional default.

    Args:
        key: Environment variable name
        default: Default value if not found

    Returns:
        Environment variable value

    Raises:
        ValueError: If variable not found and no default provided
    """
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(f"Environment variable {key} not found and no default provided")
    return value


def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def clean_text(text: str) -> str:
    """
    Clean Shakespeare text by removing extra whitespace and normalizing.

    Args:
        text: Raw text

    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def is_verse(line: str) -> bool:
    """
    Heuristic to determine if a line is verse (vs prose).
    Verse lines typically start with capital letters and have rhythm.

    Args:
        line: Text line

    Returns:
        True if likely verse, False if likely prose
    """
    line = line.strip()
    if not line:
        return False

    # Verse typically starts with capital letter
    if not line[0].isupper():
        return False

    # Verse lines are typically shorter (under 50 chars for pentameter)
    # but can be longer
    return True


def count_syllables(word: str) -> int:
    """
    Simple syllable counter for meter analysis.
    This is a basic heuristic, not perfect.

    Args:
        word: Word to count syllables in

    Returns:
        Estimated syllable count
    """
    word = word.lower().strip()
    if not word:
        return 0

    # Remove trailing e
    if word.endswith('e'):
        word = word[:-1]

    # Count vowel groups
    vowels = 'aeiouy'
    syllable_count = 0
    previous_was_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not previous_was_vowel:
            syllable_count += 1
        previous_was_vowel = is_vowel

    # Every word has at least one syllable
    return max(1, syllable_count)


def detect_formality(text: str) -> str:
    """
    Detect formality level based on archaic pronouns and vocabulary.

    Args:
        text: Text to analyze

    Returns:
        "high", "medium", or "low"
    """
    text_lower = text.lower()

    # High formality indicators
    high_indicators = ['thou', 'thee', 'thy', 'thine', 'hath', 'doth', 'wherefore', 'whence']
    # Low formality indicators
    low_indicators = ['you', 'your', "i'm", "it's", 'will not']

    high_count = sum(1 for word in high_indicators if word in text_lower)
    low_count = sum(1 for word in low_indicators if word in text_lower)

    if high_count > low_count:
        return "high"
    elif low_count > high_count:
        return "low"
    else:
        return "medium"


def has_question(text: str) -> bool:
    """Check if text contains a question."""
    return '?' in text


def has_exclamation(text: str) -> bool:
    """Check if text contains an exclamation."""
    return '!' in text


def extract_character_name(line: str) -> Optional[str]:
    """
    Extract character name from a line in standard Shakespeare format.
    Character names are typically in ALL CAPS at the start of a line.

    Args:
        line: Text line

    Returns:
        Character name if found, None otherwise
    """
    # Pattern: Start of line, all caps word(s), followed by period or newline
    pattern = r'^([A-Z][A-Z\s]+)[\.\:]'
    match = re.match(pattern, line.strip())
    if match:
        return match.group(1).strip()
    return None


def normalize_character_name(name: str) -> str:
    """
    Normalize character name (remove titles, standardize format).

    Args:
        name: Character name

    Returns:
        Normalized name
    """
    # Remove common titles
    name = re.sub(r'\b(LORD|LADY|SIR|KING|QUEEN|PRINCE|PRINCESS|DUKE|DUCHESS)\b\s*', '', name)
    return name.strip().title()
