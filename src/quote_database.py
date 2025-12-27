"""
Vector database interface for Shakespeare quotes using ChromaDB.
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Any
from pathlib import Path


class QuoteDatabase:
    """
    Interface to ChromaDB for storing and querying Shakespeare quotes.
    """

    def __init__(self, db_path: str = "./data/embeddings", collection_name: str = "shakespeare_quotes"):
        """
        Initialize the quote database.

        Args:
            db_path: Path to ChromaDB storage directory
            collection_name: Name of the collection
        """
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize ChromaDB client and collection."""
        # Ensure directory exists
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Initialize client
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"Loaded existing collection '{self.collection_name}' with {self.collection.count()} items")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Shakespeare quotes with rich metadata"}
            )
            print(f"Created new collection '{self.collection_name}'")

    def add_chunks(self, chunks: List[Dict]) -> None:
        """
        Add chunks to the database.

        Args:
            chunks: List of chunk dictionaries with embeddings and metadata
        """
        if not chunks:
            print("No chunks to add")
            return

        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for chunk in chunks:
            chunk_id = chunk.get('chunk_id')
            if not chunk_id:
                print(f"Warning: Chunk missing chunk_id, skipping")
                continue

            # Extract embedding
            embedding = chunk.get('embedding')
            if not embedding:
                print(f"Warning: Chunk {chunk_id} missing embedding, skipping")
                continue

            # Prepare metadata (exclude embedding and convert lists to strings)
            metadata = self._prepare_metadata(chunk)

            ids.append(chunk_id)
            embeddings.append(embedding)
            documents.append(chunk['chunk_text'])
            metadatas.append(metadata)

        # Add to collection
        if ids:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            print(f"Added {len(ids)} chunks to database")

    def _prepare_metadata(self, chunk: Dict) -> Dict:
        """
        Prepare metadata for ChromaDB (convert lists to strings, remove embedding).

        Args:
            chunk: Chunk dictionary

        Returns:
            Prepared metadata dictionary
        """
        metadata = {}

        for key, value in chunk.items():
            # Skip embedding and chunk_id (used as id)
            if key in ['embedding', 'chunk_id']:
                continue

            # Convert lists to comma-separated strings
            if isinstance(value, list):
                metadata[key] = ','.join(str(v) for v in value)
            # Convert other types to strings
            elif value is not None:
                metadata[key] = str(value)

        return metadata

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """
        Query the database with an embedding vector.

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Metadata filters (e.g., {"play_title": "Hamlet"})
            where_document: Document content filters

        Returns:
            Query results dictionary
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            where_document=where_document
        )

        return results

    def query_by_text(
        self,
        query_text: str,
        embedding_generator,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict]:
        """
        Query the database using text (generates embedding automatically).

        Args:
            query_text: Query text
            embedding_generator: EmbeddingsGenerator instance
            n_results: Number of results to return
            where: Metadata filters

        Returns:
            List of result dictionaries with metadata and similarity scores
        """
        # Generate query embedding
        query_embedding = embedding_generator.generate_query_embedding(query_text)

        # Query database
        results = self.query(
            query_embedding=query_embedding.tolist(),
            n_results=n_results,
            where=where
        )

        # Format results
        formatted_results = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                result = {
                    'chunk_id': results['ids'][0][i],
                    'chunk_text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                formatted_results.append(result)

        return formatted_results

    def get_by_id(self, chunk_id: str) -> Optional[Dict]:
        """
        Get a chunk by its ID.

        Args:
            chunk_id: Chunk ID

        Returns:
            Chunk dictionary or None if not found
        """
        try:
            result = self.collection.get(ids=[chunk_id])
            if result['ids']:
                return {
                    'chunk_id': result['ids'][0],
                    'chunk_text': result['documents'][0],
                    'metadata': result['metadatas'][0]
                }
        except:
            pass
        return None

    def count(self) -> int:
        """Get the number of chunks in the database."""
        return self.collection.count()

    def reset(self) -> None:
        """Reset the database (delete all data)."""
        self.client.delete_collection(name=self.collection_name)
        self._initialize_db()
        print(f"Database reset")

    def delete_chunks(self, chunk_ids: List[str]) -> None:
        """
        Delete chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs to delete
        """
        if chunk_ids:
            self.collection.delete(ids=chunk_ids)
            print(f"Deleted {len(chunk_ids)} chunks")


def main():
    """Example usage of QuoteDatabase."""
    from embeddings_generator import EmbeddingsGenerator

    # Create database
    db = QuoteDatabase()

    # Example chunks with embeddings
    generator = EmbeddingsGenerator()

    example_chunks = [
        {
            'chunk_id': 'test_1',
            'chunk_text': 'To be, or not to be: that is the question',
            'play_title': 'Hamlet',
            'character': 'Hamlet',
            'act': 3,
            'scene': 1,
            'emotional_tone': ['melancholy', 'contemplative'],
            'themes': ['death', 'existence']
        }
    ]

    # Generate embeddings
    chunks_with_embeddings = generator.generate_embeddings(example_chunks)

    # Add to database
    db.add_chunks(chunks_with_embeddings)

    # Query
    results = db.query_by_text(
        query_text="contemplating mortality",
        embedding_generator=generator,
        n_results=1
    )

    print("\nQuery results:")
    for result in results:
        print(f"Text: {result['chunk_text']}")
        print(f"Distance: {result['distance']}")


if __name__ == "__main__":
    main()
