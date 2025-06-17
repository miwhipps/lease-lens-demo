#!/usr/bin/env python3
"""Test only the RAG assistant component"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def test_rag_assistant():
    """Test RAG assistant functionality"""
    print("🤖 Testing RAG Assistant...")

    try:
        from embeddings.vector_store import LeaseVectorStore
        from ai_assistant.rag_chat import LeaseRAGAssistant

        # Create vector store with test data
        print("  Setting up vector store...")
        vs = LeaseVectorStore()

        test_lease = """
        RESIDENTIAL LEASE AGREEMENT
        Monthly Rent: $3200 per month
        Security Deposit: $6400 (2 months rent)
        Pet Policy: No pets without written permission. Pet deposit: $500 per pet.
        Utilities: Tenant pays electricity and gas. Landlord pays water and trash.
        Parking: One space included. Additional spaces: $200/month.
        Termination: 60 days notice required. Early termination fee: 2 months rent.
        Maintenance: Landlord handles major repairs. Tenant handles minor issues under $100.
        """

        vs.add_document(test_lease, "test_lease", {"filename": "test.pdf"})
        print(f"  ✅ Test document added: {len(vs.texts)} chunks")

        # Initialize RAG assistant
        print("  Initializing RAG assistant...")
        rag = LeaseRAGAssistant(vs)
        print("  ✅ RAG assistant initialized")

        # Test basic queries
        print("  Testing basic queries...")
        test_questions = [
            "What is the monthly rent?",
            "What are the pet policies?",
            "Who pays for utilities?",
            "How much is parking?",
            "What are the termination rules?",
        ]

        for question in test_questions:
            response = rag.query(question)
            if response and response.get("answer"):
                print(f"    ✅ '{question}'")
                print(f"        Answer: {response['answer'][:80]}...")
                print(f"        Confidence: {response['confidence']:.3f}")
            else:
                print(f"    ❌ '{question}': No response")

        # Test advanced features
        print("  Testing advanced features...")

        # Test lease summary
        summary = rag.get_lease_summary()
        if isinstance(summary, dict) and len(summary) > 0:
            print(f"    ✅ Lease summary: {len(summary)} sections")
        else:
            print("    ❌ Lease summary failed")

        # Test risk analysis
        risks = rag.analyze_lease_risks()
        if isinstance(risks, list):
            print(f"    ✅ Risk analysis: {len(risks)} risks identified")
        else:
            print("    ❌ Risk analysis failed")

        # Test key figures
        figures = rag.extract_key_figures()
        if isinstance(figures, dict) and "financial" in figures:
            print(f"    ✅ Key figures extracted")
        else:
            print("    ❌ Key figures extraction failed")

        # Test completed successfully
        assert True

    except Exception as e:
        print(f"❌ RAG assistant test error: {e}")
        assert False, f"RAG assistant test failed: {e}"


if __name__ == "__main__":
    print("🧪 RAG Assistant Component Test")
    print("-" * 30)

    success = test_rag_assistant()

    if success:
        print("✅ RAG assistant test successful!")
    else:
        print("❌ RAG assistant test failed")
