#!/usr/bin/env python3
"""
Main CLI interface for the Shakespeare Poet system.
"""
import os
import sys
import argparse
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import load_env, ensure_dir
from src.chunker import ShakespeareChunker
from src.embeddings_generator import EmbeddingsGenerator
from src.quote_database import QuoteDatabase
from src.scene_generator import SceneGenerator


def setup_database(args):
    """Setup and populate the database from Shakespeare texts."""
    print("=" * 60)
    print("SHAKESPEARE DATABASE SETUP")
    print("=" * 60)

    # Load Shakespeare text
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: Source file not found: {source_path}")
        return

    print(f"\n1. Loading Shakespeare text from: {source_path}")
    with open(source_path, 'r', encoding='utf-8') as f:
        play_text = f.read()

    # Extract play title from filename or use provided
    play_title = args.title or source_path.stem

    # Chunk the text
    print(f"\n2. Chunking text from '{play_title}'...")
    chunker = ShakespeareChunker()
    chunks = chunker.chunk_play(
        play_text=play_text,
        play_title=play_title,
        chunk_types=["full_line", "phrase", "fragment"]
    )
    print(f"   Created {len(chunks)} chunks")

    # Save chunks
    chunks_path = Path(args.output) / f"{play_title}_chunks.json"
    ensure_dir(chunks_path.parent)
    chunker.save_chunks(str(chunks_path))
    print(f"   Saved chunks to: {chunks_path}")

    # Generate embeddings
    print(f"\n3. Generating embeddings...")
    generator = EmbeddingsGenerator(model_name=args.embedding_model)
    chunks_with_embeddings = generator.generate_embeddings(chunks)

    # Initialize database
    print(f"\n4. Storing in database...")
    db = QuoteDatabase(db_path=args.db_path)
    db.add_chunks(chunks_with_embeddings)

    print(f"\n✓ Database setup complete!")
    print(f"  Total chunks in database: {db.count()}")
    print("=" * 60)


def generate_scene(args):
    """Generate a Shakespearean scene."""
    print("=" * 60)
    print("GENERATING SHAKESPEAREAN SCENE")
    print("=" * 60)

    # Parse characters (format: "Name: Description; Name2: Description2")
    characters = []
    if args.characters:
        for char_str in args.characters.split(';'):
            char_str = char_str.strip()
            if ':' in char_str:
                name, desc = char_str.split(':', 1)
                characters.append({
                    "name": name.strip().upper(),
                    "description": desc.strip()
                })

    if len(characters) < 2:
        print("Error: Please provide at least 2 characters")
        print("Format: 'Name: Description; Name2: Description2'")
        return

    # Parse themes
    themes = [t.strip() for t in args.themes.split(',')] if args.themes else None

    # Initialize generator
    print(f"\nInitializing scene generator...")
    generator = SceneGenerator(
        database=QuoteDatabase(db_path=args.db_path),
        embedding_generator=EmbeddingsGenerator(model_name=args.embedding_model)
    )

    # Generate scene
    print(f"\nGenerating scene...")
    print(f"  Description: {args.scene}")
    print(f"  Characters: {len(characters)}")
    print(f"  Themes: {themes}")
    print(f"  Target length: {args.length} speeches")
    print()

    scene = generator.generate_scene(
        scene_description=args.scene,
        characters=characters,
        themes=themes,
        target_length=args.length
    )

    # Format and display
    formatted = generator.format_scene(scene)
    print(formatted)

    # Save if requested
    if args.output:
        output_path = Path(args.output)
        ensure_dir(output_path.parent)

        # Save formatted scene
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted)

        # Also save JSON data
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(scene, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Scene saved to: {output_path}")
        print(f"✓ Scene data saved to: {json_path}")


def query_database(args):
    """Query the database for quotes."""
    print("=" * 60)
    print("QUERYING SHAKESPEARE DATABASE")
    print("=" * 60)

    # Initialize components
    db = QuoteDatabase(db_path=args.db_path)
    generator = EmbeddingsGenerator(model_name=args.embedding_model)

    print(f"\nQuery: {args.query}")
    print(f"Max results: {args.max_results}")

    # Build filters
    where = {}
    if args.play:
        where['play_title'] = args.play
    if args.character:
        where['character'] = args.character

    # Query
    results = db.query_by_text(
        query_text=args.query,
        embedding_generator=generator,
        n_results=args.max_results,
        where=where if where else None
    )

    # Display results
    print(f"\nFound {len(results)} results:")
    print("-" * 60)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['chunk_text']}")
        metadata = result['metadata']
        print(f"   Play: {metadata.get('play_title', 'Unknown')}")
        print(f"   Character: {metadata.get('character', 'Unknown')}")
        print(f"   Act/Scene: {metadata.get('act', '?')}.{metadata.get('scene', '?')}")
        print(f"   Emotional tone: {metadata.get('emotional_tone', 'N/A')}")
        print(f"   Themes: {metadata.get('themes', 'N/A')}")
        if result['distance'] is not None:
            print(f"   Distance: {result['distance']:.4f}")

    print("-" * 60)


def main():
    """Main CLI entry point."""
    # Load environment variables
    load_env()

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Shakespeare Poet - Generate scenes using authentic Shakespeare quotes",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup database from Shakespeare texts')
    setup_parser.add_argument('--source', required=True, help='Path to Shakespeare text file')
    setup_parser.add_argument('--title', help='Play title (default: filename)')
    setup_parser.add_argument('--output', default='./data/processed', help='Output directory for chunks')
    setup_parser.add_argument('--db-path', default='./data/embeddings', help='Database path')
    setup_parser.add_argument('--embedding-model', default='sentence-transformers/all-MiniLM-L6-v2', help='Embedding model')

    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate a Shakespearean scene')
    generate_parser.add_argument('--scene', required=True, help='Scene description')
    generate_parser.add_argument('--characters', required=True, help='Characters (format: "Name: Desc; Name2: Desc2")')
    generate_parser.add_argument('--themes', help='Themes (comma-separated)')
    generate_parser.add_argument('--length', type=int, default=10, help='Target number of speeches')
    generate_parser.add_argument('--output', help='Output file path')
    generate_parser.add_argument('--db-path', default='./data/embeddings', help='Database path')
    generate_parser.add_argument('--embedding-model', default='sentence-transformers/all-MiniLM-L6-v2', help='Embedding model')

    # Query command
    query_parser = subparsers.add_parser('query', help='Query the quote database')
    query_parser.add_argument('--query', required=True, help='Search query')
    query_parser.add_argument('--play', help='Filter by play title')
    query_parser.add_argument('--character', help='Filter by character')
    query_parser.add_argument('--max-results', type=int, default=5, help='Maximum results')
    query_parser.add_argument('--db-path', default='./data/embeddings', help='Database path')
    query_parser.add_argument('--embedding-model', default='sentence-transformers/all-MiniLM-L6-v2', help='Embedding model')

    args = parser.parse_args()

    # Execute command
    if args.command == 'setup':
        setup_database(args)
    elif args.command == 'generate':
        generate_scene(args)
    elif args.command == 'query':
        query_database(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
