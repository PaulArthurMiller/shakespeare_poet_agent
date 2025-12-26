"""
Session manager for tracking used quotes to prevent repetition.
"""
from typing import Set, List, Dict, Optional
import json
from pathlib import Path
from datetime import datetime


class SessionManager:
    """
    Track used quotes during a scene/session to prevent repetition.
    """

    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize the session manager.

        Args:
            session_id: Optional session identifier
        """
        self.session_id = session_id or self._generate_session_id()
        self.used_chunk_ids: Set[str] = set()
        self.usage_history: List[Dict] = []

    def _generate_session_id(self) -> str:
        """Generate a unique session ID based on timestamp."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def mark_used(self, chunk_id: str, metadata: Optional[Dict] = None) -> None:
        """
        Mark a chunk as used in this session.

        Args:
            chunk_id: Chunk ID to mark as used
            metadata: Optional metadata about the usage (e.g., character, position in scene)
        """
        self.used_chunk_ids.add(chunk_id)

        # Record usage history
        usage_record = {
            'chunk_id': chunk_id,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.usage_history.append(usage_record)

    def is_used(self, chunk_id: str) -> bool:
        """
        Check if a chunk has been used in this session.

        Args:
            chunk_id: Chunk ID to check

        Returns:
            True if used, False otherwise
        """
        return chunk_id in self.used_chunk_ids

    def get_exclusion_list(self) -> List[str]:
        """
        Get list of chunk IDs to exclude from queries.

        Returns:
            List of used chunk IDs
        """
        return list(self.used_chunk_ids)

    def get_usage_count(self) -> int:
        """
        Get the number of chunks used in this session.

        Returns:
            Usage count
        """
        return len(self.used_chunk_ids)

    def get_usage_history(self) -> List[Dict]:
        """
        Get the full usage history for this session.

        Returns:
            List of usage records
        """
        return self.usage_history

    def reset(self) -> None:
        """Reset the session (clear all used chunks)."""
        self.used_chunk_ids.clear()
        self.usage_history.clear()
        print(f"Session {self.session_id} reset")

    def save_session(self, output_path: str) -> None:
        """
        Save session data to a JSON file.

        Args:
            output_path: Path to output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        session_data = {
            'session_id': self.session_id,
            'used_chunk_ids': list(self.used_chunk_ids),
            'usage_history': self.usage_history,
            'usage_count': len(self.used_chunk_ids)
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

        print(f"Session saved to {output_path}")

    def load_session(self, input_path: str) -> None:
        """
        Load session data from a JSON file.

        Args:
            input_path: Path to input file
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            session_data = json.load(f)

        self.session_id = session_data.get('session_id', self.session_id)
        self.used_chunk_ids = set(session_data.get('used_chunk_ids', []))
        self.usage_history = session_data.get('usage_history', [])

        print(f"Session loaded from {input_path}")
        print(f"Session ID: {self.session_id}")
        print(f"Used chunks: {len(self.used_chunk_ids)}")

    def get_statistics(self) -> Dict:
        """
        Get session statistics.

        Returns:
            Dictionary with session statistics
        """
        stats = {
            'session_id': self.session_id,
            'total_chunks_used': len(self.used_chunk_ids),
            'usage_events': len(self.usage_history),
            'start_time': self.usage_history[0]['timestamp'] if self.usage_history else None,
            'last_usage': self.usage_history[-1]['timestamp'] if self.usage_history else None
        }

        return stats

    def merge_session(self, other_session: 'SessionManager') -> None:
        """
        Merge another session into this one.

        Args:
            other_session: Another SessionManager instance
        """
        self.used_chunk_ids.update(other_session.used_chunk_ids)
        self.usage_history.extend(other_session.usage_history)

        print(f"Merged session {other_session.session_id} into {self.session_id}")
        print(f"Total chunks now: {len(self.used_chunk_ids)}")


def main():
    """Example usage of SessionManager."""
    # Create session
    session = SessionManager()

    print(f"Session ID: {session.session_id}")

    # Mark some chunks as used
    session.mark_used("chunk_1", metadata={'character': 'Hamlet', 'position': 1})
    session.mark_used("chunk_2", metadata={'character': 'Ophelia', 'position': 2})
    session.mark_used("chunk_3", metadata={'character': 'Hamlet', 'position': 3})

    # Check usage
    print(f"\nUsage count: {session.get_usage_count()}")
    print(f"Is chunk_1 used? {session.is_used('chunk_1')}")
    print(f"Is chunk_99 used? {session.is_used('chunk_99')}")

    # Get exclusion list
    print(f"\nExclusion list: {session.get_exclusion_list()}")

    # Get statistics
    stats = session.get_statistics()
    print(f"\nSession statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Save session
    session.save_session("./data/processed/test_session.json")


if __name__ == "__main__":
    main()
