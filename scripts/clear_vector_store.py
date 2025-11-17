#!/usr/bin/env python3
"""
Clear Vector Store Utility
Provides options to clear data from the Qdrant vector database
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.embeddings.vector_store import VectorStore


def main():
    """Main function to clear vector store"""
    
    if len(sys.argv) < 2:
        print("Usage: python scripts/clear_vector_store.py <option>")
        print("\nOptions:")
        print("  all              - Delete all points from the collection")
        print("  collection       - Delete the entire collection (and recreate it)")
        print("  paper <title>    - Delete chunks for a specific paper by title")
        print("  filename <name>  - Delete chunks for a specific paper by filename")
        print("\nExample:")
        print("  python scripts/clear_vector_store.py all")
        print("  python scripts/clear_vector_store.py paper 'BERT: Pre-training of Deep Bidirectional Transformers'")
        print("  python scripts/clear_vector_store.py filename bert.pdf")
        return
    
    option = sys.argv[1].lower()
    
    # Initialize vector store
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    use_openai = embedding_model.startswith("text-embedding") or embedding_model.startswith("ada")
    
    vector_store = VectorStore(
        collection_name="research_papers",
        embedding_model=embedding_model,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        use_openai=use_openai
    )
    
    # Show current collection info
    info = vector_store.get_collection_info()
    if "error" not in info:
        print(f"\nCurrent collection status:")
        print(f"  Points: {info.get('points_count', 0)}")
        print(f"  Vectors: {info.get('vectors_count', 0)}")
    print()
    
    # Execute based on option
    if option == "all":
        print("⚠️  WARNING: This will delete ALL points from the collection!")
        response = input("Are you sure? (yes/no): ")
        if response.lower() == "yes":
            success = vector_store.delete_all_points()
            if success:
                print("\n✅ Successfully deleted all points")
            else:
                print("\n❌ Failed to delete points")
        else:
            print("Cancelled.")
    
    elif option == "collection":
        print("⚠️  WARNING: This will delete the ENTIRE collection!")
        response = input("Are you sure? (yes/no): ")
        if response.lower() == "yes":
            success = vector_store.recreate_collection()
            if success:
                print("\n✅ Successfully recreated collection")
            else:
                print("\n❌ Failed to recreate collection")
        else:
            print("Cancelled.")
    
    elif option == "paper":
        if len(sys.argv) < 3:
            print("❌ Error: Please provide a paper title")
            print("Usage: python scripts/clear_vector_store.py paper '<title>'")
            return
        
        title = " ".join(sys.argv[2:])
        print(f"Deleting chunks for paper: {title}")
        success = vector_store.delete_by_paper(title=title)
        if success:
            print("\n✅ Successfully deleted chunks for paper")
        else:
            print("\n❌ Failed to delete chunks")
    
    elif option == "filename":
        if len(sys.argv) < 3:
            print("❌ Error: Please provide a filename")
            print("Usage: python scripts/clear_vector_store.py filename <filename>")
            return
        
        filename = sys.argv[2]
        print(f"Deleting chunks for filename: {filename}")
        success = vector_store.delete_by_paper(filename=filename)
        if success:
            print("\n✅ Successfully deleted chunks for filename")
        else:
            print("\n❌ Failed to delete chunks")
    
    else:
        print(f"❌ Unknown option: {option}")
        print("Use 'all', 'collection', 'paper', or 'filename'")
        return
    
    # Show updated collection info
    info = vector_store.get_collection_info()
    if "error" not in info:
        print(f"\nUpdated collection status:")
        print(f"  Points: {info.get('points_count', 0)}")
        print(f"  Vectors: {info.get('vectors_count', 0)}")


if __name__ == "__main__":
    main()

