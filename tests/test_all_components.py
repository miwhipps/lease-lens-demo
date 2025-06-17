#!/usr/bin/env python3
"""
Comprehensive test script for all LeaseLens core components
Tests OCR, Vector Store, and RAG Assistant integration
"""

import sys
import os
import cv2
import numpy as np
import tempfile
import time
from pathlib import Path
import pytest

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import our components
try:
    from ocr_pipeline.preprocess import DocumentPreprocessor
    from ocr_pipeline.textract_extract import TextractExtractor
    from embeddings.vector_store import LeaseVectorStore
    from ai_assistant.rag_chat import LeaseRAGAssistant

    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def create_test_lease_image():
    """Create a test lease document image for OCR testing"""
    print("\n📄 Creating test lease document image...")

    # Create a white background image
    img_height, img_width = 800, 600
    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

    # Sample lease text
    lease_text = [
        "RESIDENTIAL LEASE AGREEMENT",
        "",
        "Property: 123 Test Street, Demo City, CA 94000",
        "Tenant: John Doe",
        "Landlord: Test Properties LLC",
        "",
        "LEASE TERMS:",
        "Monthly Rent: $2,500.00",
        "Security Deposit: $5,000.00",
        "Lease Duration: 12 months",
        "Start Date: January 1, 2024",
        "",
        "PET POLICY:",
        "No pets allowed without written permission.",
        "Pet deposit: $750 per pet if approved.",
        "",
        "UTILITIES:",
        "Tenant pays: Electricity, Gas, Internet",
        "Landlord pays: Water, Trash, Landscaping",
        "",
        "TERMINATION:",
        "60 days written notice required.",
        "Early termination fee: 2 months rent.",
        "",
        "PARKING:",
        "One parking space included.",
        "Additional spaces: $200/month.",
    ]

    # Add text to image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (0, 0, 0)  # Black text
    thickness = 1

    y_offset = 40
    line_height = 25

    for line in lease_text:
        if line:  # Skip empty lines for spacing
            cv2.putText(img, line, (30, y_offset), font, font_scale, color, thickness)
        y_offset += line_height

    # Save the test image
    test_image_path = "test_lease_document.png"
    cv2.imwrite(test_image_path, img)

    print(f"✅ Test lease image created: {test_image_path}")
    return test_image_path


def test_ocr_pipeline():
    """Test the OCR preprocessing and extraction pipeline"""
    # Create test image
    test_image = create_test_lease_image()
    
    # Test document preprocessing
    preprocessor = DocumentPreprocessor()
    processed_image = preprocessor.enhance_document(test_image)
    
    assert processed_image is not None
    assert processed_image.shape[0] > 0
    
    # Test mock extraction (for CI/CD without AWS)
    from ocr_pipeline.textract_extract import MockTextractExtractor
    extractor = MockTextractExtractor()
    result = extractor.extract_from_file(test_image)
    
    assert result is not None
    assert "text" in result
    assert len(result["text"]) > 0
    assert "confidence" in result
    
    # Clean up test file
    if os.path.exists(test_image):
        os.remove(test_image)


def test_vector_store():
    """Test the vector store functionality"""
    # Initialize vector store
    vector_store = LeaseVectorStore()
    
    # Test document addition
    sample_text = "Monthly rent is $3,200. Security deposit is $6,400."
    vector_store.add_document(sample_text, "test_doc", {"source": "test"})
    
    # Test search functionality
    results = vector_store.search("What is the rent?", k=2)
    
    assert len(vector_store.texts) > 0
    assert results is not None
    assert len(results) > 0
    assert "score" in results[0]
    
    # Test statistics
    stats = vector_store.get_document_stats()
    assert stats is not None
    assert "total_chunks" in stats
    assert stats["total_chunks"] > 0


@pytest.fixture
def vector_store():
    """Create a vector store fixture for testing"""
    # Initialize vector store
    vs = LeaseVectorStore()
    
    # Add sample test data
    sample_lease_text = """
    RESIDENTIAL LEASE AGREEMENT
    
    Property Address: 456 Demo Avenue, Test City, CA 90210
    Monthly Rent: $3,200 per month, due on the 1st of each month
    Security Deposit: $6,400 (equivalent to 2 months rent)
    
    PET POLICY: 
    No pets are allowed without written permission from landlord.
    If pets are approved, pet deposit of $500 per pet is required.
    Monthly pet rent: $75 per pet.
    
    UTILITIES:
    Tenant is responsible for: Electricity, Gas, Internet, Cable
    Landlord is responsible for: Water, Sewer, Trash, Landscaping
    
    PARKING:
    One assigned parking space included with rent.
    Additional parking spaces available for $150 per month.
    Guest parking limited to 2 hours in visitor spaces.
    
    TERMINATION CLAUSES:
    Either party may terminate this lease with 60 days written notice.
    Early termination by tenant requires payment of 2 months rent as penalty.
    Military clause allows early termination with PCS orders.
    
    MAINTENANCE:
    Landlord responsible for major repairs, HVAC, plumbing, electrical.
    Tenant responsible for minor maintenance under $75 per incident.
    Emergency maintenance hotline: (555) 123-4567
    """
    
    vs.add_document(sample_lease_text, "test_lease_001", {"filename": "test_lease.pdf", "source": "test"})
    return vs

def test_rag_assistant(vector_store):
    """Test the RAG assistant functionality"""
    # Test 1: RAG Initialization
    rag_assistant = LeaseRAGAssistant(vector_store)
    
    # Test basic functionality
    response = rag_assistant.query("What is the monthly rent?")
    
    assert response is not None
    assert "answer" in response
    assert len(response["answer"]) > 0
    assert "confidence" in response
    assert response["confidence"] >= 0



def test_end_to_end_integration():
    """Test complete end-to-end integration"""
    # Create components needed for integration test
    vector_store = LeaseVectorStore()
    
    # Add test document to vector store
    sample_text = """
    Monthly rent is $3,200 per month. Security deposit is $6,400.
    Parking space included. Additional parking available for $150/month.
    """
    vector_store.add_document(sample_text, "integration_test", {"source": "test"})
    
    # Create RAG assistant
    rag_assistant = LeaseRAGAssistant(vector_store)
    
    # Test integration query
    integration_query = "What would be my total monthly cost including rent and parking?"
    response = rag_assistant.query(integration_query)
    
    # Assertions for integration test
    assert response is not None
    assert "answer" in response
    assert len(response["answer"]) > 0
    assert "confidence" in response
    assert response["confidence"] >= 0


def run_performance_benchmarks():
    """Run performance benchmarks"""
    print("\n⚡ Performance Benchmarks")
    print("-" * 25)

    try:
        # Create vector store with sample data
        vector_store = LeaseVectorStore()

        # Large sample text for performance testing
        large_sample = (
            """
        COMPREHENSIVE LEASE AGREEMENT
        
        This agreement contains multiple sections covering various aspects of the rental agreement.
        """
            * 50
        )  # Repeat to create larger document

        # Benchmark 1: Document Processing Speed
        print("Benchmark 1: Document Processing Speed")
        start_time = time.time()
        vector_store.add_document(large_sample, "perf_test_doc")
        processing_time = time.time() - start_time

        chunks_per_second = len(vector_store.texts) / processing_time
        print(f"  📊 Processing time: {processing_time:.2f}s")
        print(f"  📦 Chunks created: {len(vector_store.texts)}")
        print(f"  ⚡ Chunks per second: {chunks_per_second:.1f}")

        # Benchmark 2: Search Performance
        print("Benchmark 2: Search Performance")

        test_queries = ["rent", "deposit", "pets", "parking", "utilities"] * 20  # 100 queries

        start_time = time.time()
        for query in test_queries:
            vector_store.search(query, k=5)
        search_time = time.time() - start_time

        queries_per_second = len(test_queries) / search_time
        avg_query_time = search_time / len(test_queries) * 1000  # ms

        print(f"  📊 Total search time: {search_time:.2f}s")
        print(f"  🔍 Queries tested: {len(test_queries)}")
        print(f"  ⚡ Queries per second: {queries_per_second:.1f}")
        print(f"  📈 Average query time: {avg_query_time:.1f}ms")

        # Benchmark 3: RAG Performance
        print("Benchmark 3: RAG Performance")

        rag_assistant = LeaseRAGAssistant(vector_store)

        rag_queries = [
            "What is the monthly rent?",
            "What are the pet policies?",
            "Who handles maintenance?",
        ] * 10  # 30 queries

        start_time = time.time()
        for query in rag_queries:
            rag_assistant.query(query)
        rag_time = time.time() - start_time

        rag_qps = len(rag_queries) / rag_time
        avg_rag_time = rag_time / len(rag_queries)

        print(f"  📊 Total RAG time: {rag_time:.2f}s")
        print(f"  🤖 RAG queries tested: {len(rag_queries)}")
        print(f"  ⚡ RAG queries per second: {rag_qps:.1f}")
        print(f"  📈 Average RAG time: {avg_rag_time:.2f}s")

        return True

    except Exception as e:
        print(f"  ❌ Performance benchmark failed: {e}")
        return False


def main():
    """Main test execution function"""
    print("🚀 LeaseLens Core Components Test Suite")
    print("=" * 50)
    print("Testing OCR Pipeline, Vector Store, and RAG Assistant")
    print()

    start_time = time.time()

    # Run all tests
    test_results = []

    try:
        # Individual component tests
        ocr_result = test_ocr_pipeline()
        test_results.append(("OCR Pipeline", ocr_result))

        vector_result = test_vector_store()
        test_results.append(("Vector Store", vector_result is not False))

        if vector_result:
            vector_store, _ = vector_result
            rag_result = test_rag_assistant(vector_store)
            test_results.append(("RAG Assistant", rag_result is not False))

            # Integration test
            integration_result = test_end_to_end_integration()
            test_results.append(("End-to-End Integration", integration_result))

            # Performance benchmarks
            perf_result = run_performance_benchmarks()
            test_results.append(("Performance Benchmarks", perf_result))

    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
        return False

    # Calculate results
    total_time = time.time() - start_time
    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)

    # Print summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)

    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")

    print(f"\nTotal Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print(f"Total Time: {total_time:.2f} seconds")

    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! LeaseLens components are working correctly.")
        print("✅ Ready to proceed to UI development (Step 6)")
        return True
    else:
        print(f"\n⚠️ {total_tests - passed_tests} test(s) failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
