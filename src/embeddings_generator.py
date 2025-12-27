"""
Generate vector embeddings for Shakespeare text chunks.
"""
import os
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingsGenerator:
    """
    Generate embeddings for Shakespeare chunks using sentence-transformers.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embeddings generator.

        Args:
            model_name: Name of the sentence-transformers model to use
                       - all-MiniLM-L6-v2: Fast, 384 dimensions
                       - all-mpnet-base-v2: Better quality, 768 dimensions
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def generate_embeddings(
        self,
        chunks: List[Dict],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Generate embeddings for a list of chunks.

        Args:
            chunks: List of chunk dictionaries with 'chunk_text' field
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            List of chunks with 'embedding' field added
        """
        if not chunks:
            return []

        # Extract texts
        texts = [chunk['chunk_text'] for chunk in chunks]

        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )

        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding.tolist()

        print(f"Generated {len(embeddings)} embeddings")
        return chunks

    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.

        Args:
            query: Query text

        Returns:
            Embedding vector
        """
        return self.model.encode(query, convert_to_numpy=True)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self.model.get_sentence_embedding_dimension()


def main():
    """Example usage of EmbeddingsGenerator."""
    # Example chunks
    example_chunks = [
        {
            'chunk_id': '1',
            'chunk_text': 'To be, or not to be: that is the question',
            'play_title': 'Hamlet',
            'character': 'Hamlet'
        },
        {
            'chunk_id': '2',
            'chunk_text': 'All the world\'s a stage',
            'play_title': 'As You Like It',
            'character': 'Jaques'
        }
    ]

    # Generate embeddings
    generator = EmbeddingsGenerator()
    chunks_with_embeddings = generator.generate_embeddings(example_chunks)

    # Print results
    for chunk in chunks_with_embeddings:
        print(f"\nText: {chunk['chunk_text']}")
        print(f"Embedding shape: {len(chunk['embedding'])}")
        print(f"First 5 values: {chunk['embedding'][:5]}")


if __name__ == "__main__":
    main()
