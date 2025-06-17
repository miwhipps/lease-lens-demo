#!/usr/bin/env python3
"""Test only the vector store component"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def test_vector_store():
    """Test vector store functionality"""
    print("🧠 Testing Vector Store...")

    try:
        from embeddings.vector_store import LeaseVectorStore

        # Initialize vector store
        print("  Initializing vector store...")
        vs = LeaseVectorStore()
        print(f"  ✅ Model loaded: {vs.model_name}")

        # Test document addition
        print("  Adding test document...")
        test_text = """
        Monthly rent is $2000. Security deposit is $4000.
        Pets are not allowed without permission.
        Parking space included. Utilities: tenant pays electric and gas.
        """

        vs.add_document(test_text, "test_doc", {"source": "test"})
        print(f"  ✅ Document added: {len(vs.texts)} chunks created")

        # Test search
        print("  Testing search functionality...")
        queries = ["What is the rent?", "Are pets allowed?", "What about parking?", "Who pays utilities?"]

        for query in queries:
            results = vs.search(query, k=2)
            if results:
                print(f"    ✅ '{query}': {len(results)} results (score: {results[0]['score']:.3f})")
            else:
                print(f"    ❌ '{query}': No results")

        # Test statistics
        stats = vs.get_document_stats()
        if stats:
            print(f"  ✅ Statistics: {stats['total_chunks']} chunks, {stats['total_documents']} docs")
            return True
        else:
            print("  ❌ Statistics failed")
            return False

    except Exception as e:
        print(f"❌ Vector store test error: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Vector Store Component Test")
    print("-" * 30)

    success = test_vector_store()

    if success:
        print("✅ Vector store test successful!")
    else:
        print("❌ Vector store test failed")
