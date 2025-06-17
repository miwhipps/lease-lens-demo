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
        "Additional spaces: $200/month."
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
    print("\n🔍 Testing OCR Pipeline")
    print("-" * 30)
    
    # Create test image
    test_image = create_test_lease_image()
    
    try:
        # Test 1: Document Preprocessing
        print("Test 1: Document Preprocessing...")
        preprocessor = DocumentPreprocessor()
        
        start_time = time.time()
        processed_image = preprocessor.enhance_document(test_image)
        preprocessing_time = time.time() - start_time
        
        if processed_image is not None and processed_image.shape[0] > 0:
            print(f"  ✅ Preprocessing successful in {preprocessing_time:.2f}s")
            print(f"  📐 Output image shape: {processed_image.shape}")
        else:
            print("  ❌ Preprocessing failed")
            return False
        
        # Test 2: Mock Textract (since we might not have AWS credentials)
        print("Test 2: Text Extraction (Mock)...")
        
        # Create a mock extractor for testing without AWS
        class MockTextractExtractor:
            def extract_from_file(self, file_path, preprocess=True):
                return {
                    'text': """RESIDENTIAL LEASE AGREEMENT
Property: 123 Test Street, Demo City, CA 94000
Monthly Rent: $2,500.00
Security Deposit: $5,000.00
Pet Policy: No pets allowed without written permission.
Utilities: Tenant pays electricity, gas, internet.
Termination: 60 days written notice required.""",
                    'confidence': 92.5,
                    'line_count': 15,
                    'character_count': 234
                }
        
        mock_extractor = MockTextractExtractor()
        
        start_time = time.time()
        extraction_result = mock_extractor.extract_from_file(test_image)
        extraction_time = time.time() - start_time
        
        if extraction_result and len(extraction_result['text']) > 0:
            print(f"  ✅ Text extraction successful in {extraction_time:.2f}s")
            print(f"  📊 Confidence: {extraction_result['confidence']:.1f}%")
            print(f"  📝 Characters extracted: {extraction_result['character_count']}")
            print(f"  📄 Sample text: {extraction_result['text'][:100]}...")
        else:
            print("  ❌ Text extraction failed")
            return False
        
        return extraction_result
        
    except Exception as e:
        print(f"  ❌ OCR pipeline test failed: {e}")
        return False

def test_vector_store():
    """Test the vector store functionality"""
    print("\n🧠 Testing Vector Store")
    print("-" * 25)
    
    try:
        # Test 1: Vector Store Initialization
        print("Test 1: Vector Store Initialization...")
        start_time = time.time()
        vector_store = LeaseVectorStore()
        init_time = time.time() - start_time
        
        print(f"  ✅ Vector store initialized in {init_time:.2f}s")
        print(f"  🤖 Model: {vector_store.model_name}")
        print(f"  📐 Dimension: {vector_store.dimension}")
        
        # Test 2: Document Addition
        print("Test 2: Document Addition...")
        
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
        
        start_time = time.time()
        vector_store.add_document(
            sample_lease_text, 
            "test_lease_001",
            {"filename": "test_lease.pdf", "source": "test"}
        )
        embedding_time = time.time() - start_time
        
        print(f"  ✅ Document added in {embedding_time:.2f}s")
        print(f"  📦 Chunks created: {len(vector_store.texts)}")
        print(f"  💾 Vector index size: {vector_store.index.ntotal}")
        
        # Test 3: Search Functionality
        print("Test 3: Search Functionality...")
        
        test_queries = [
            "What is the monthly rent?",
            "What are the pet policies?",
            "Who is responsible for maintenance?",
            "What utilities are included?",
            "Are there parking arrangements?"
        ]
        
        search_results = {}
        total_search_time = 0
        
        for query in test_queries:
            start_time = time.time()
            results = vector_store.search(query, k=3)
            search_time = time.time() - start_time
            total_search_time += search_time
            
            search_results[query] = results
            
            if results:
                print(f"  ✅ '{query}': {len(results)} results ({search_time*1000:.1f}ms)")
                print(f"      Top result score: {results[0]['score']:.3f}")
            else:
                print(f"  ❌ '{query}': No results found")
        
        avg_search_time = total_search_time / len(test_queries)
        print(f"  📊 Average search time: {avg_search_time*1000:.1f}ms")
        
        # Test 4: Document Statistics
        print("Test 4: Document Statistics...")
        stats = vector_store.get_document_stats()
        
        if stats:
            print(f"  ✅ Statistics generated:")
            print(f"      Total chunks: {stats['total_chunks']}")
            print(f"      Total documents: {stats['total_documents']}")
            print(f"      Average chunk length: {stats['average_chunk_length']:.1f} chars")
        else:
            print("  ❌ Failed to generate statistics")
            return False
        
        return vector_store, search_results
        
    except Exception as e:
        print(f"  ❌ Vector store test failed: {e}")
        return False

def test_rag_assistant(vector_store):
    """Test the RAG assistant functionality"""
    print("\n🤖 Testing RAG Assistant")
    print("-" * 25)
    
    try:
        # Test 1: RAG Initialization
        print("Test 1: RAG Assistant Initialization...")
        start_time = time.time()
        rag_assistant = LeaseRAGAssistant(vector_store)
        init_time = time.time() - start_time
        
        print(f"  ✅ RAG assistant initialized in {init_time:.2f}s")
        
        # Test 2: Basic Query Processing
        print("Test 2: Basic Query Processing...")
        
        test_questions = [
            "What is the monthly rent amount?",
            "What are the pet policies?",
            "Are there any break clauses?",
            "What utilities does the tenant pay for?",
            "How much is parking?"
        ]
        
        query_results = []
        total_query_time = 0
        
        for question in test_questions:
            start_time = time.time()
            response = rag_assistant.query(question)
            query_time = time.time() - start_time
            total_query_time += query_time
            
            query_results.append(response)
            
            if response and response.get('answer'):
                print(f"  ✅ '{question}'")
                print(f"      Answer: {response['answer'][:100]}...")
                print(f"      Confidence: {response['confidence']:.3f}")
                print(f"      Sources: {len(response['sources'])}")
                print(f"      Time: {query_time*1000:.1f}ms")
            else:
                print(f"  ❌ '{question}': No response generated")
        
        avg_query_time = total_query_time / len(test_questions)
        print(f"  📊 Average query time: {avg_query_time:.2f}s")
        
        # Test 3: Batch Processing
        print("Test 3: Batch Processing...")
        
        batch_questions = [
            "What is the security deposit?",
            "Who handles maintenance?",
            "What are the termination rules?"
        ]
        
        start_time = time.time()
        batch_responses = rag_assistant.batch_query(batch_questions)
        batch_time = time.time() - start_time
        
        if len(batch_responses) == len(batch_questions):
            print(f"  ✅ Batch processing successful in {batch_time:.2f}s")
            print(f"      Processed {len(batch_responses)} questions")
        else:
            print(f"  ❌ Batch processing failed")
            return False
        
        # Test 4: Lease Summary
        print("Test 4: Lease Summary Generation...")
        
        start_time = time.time()
        summary = rag_assistant.get_lease_summary()
        summary_time = time.time() - start_time
        
        if isinstance(summary, dict) and len(summary) > 0:
            print(f"  ✅ Lease summary generated in {summary_time:.2f}s")
            print(f"      Summary sections: {len(summary)}")
            for key in list(summary.keys())[:3]:  # Show first 3 items
                print(f"        {key}: {summary[key][:50]}...")
        else:
            print(f"  ❌ Lease summary generation failed")
            return False
        
        # Test 5: Risk Analysis
        print("Test 5: Risk Analysis...")
        
        start_time = time.time()
        risks = rag_assistant.analyze_lease_risks()
        risk_time = time.time() - start_time
        
        if isinstance(risks, list):
            print(f"  ✅ Risk analysis completed in {risk_time:.2f}s")
            print(f"      Risk items identified: {len(risks)}")
            
            for risk in risks[:2]:  # Show first 2 risks
                print(f"        {risk['category']}: {risk['risk_level']} risk")
        else:
            print(f"  ❌ Risk analysis failed")
            return False
        
        # Test 6: Key Figures Extraction
        print("Test 6: Key Figures Extraction...")
        
        start_time = time.time()
        figures = rag_assistant.extract_key_figures()
        figures_time = time.time() - start_time
        
        if isinstance(figures, dict) and 'financial' in figures:
            print(f"  ✅ Key figures extracted in {figures_time:.2f}s")
            financial_items = len(figures['financial'])
            date_items = len(figures.get('dates', {}))
            print(f"      Financial items: {financial_items}")
            print(f"      Date items: {date_items}")
            
            if 'summary' in figures:
                print(f"      Monthly cost estimate: {figures['summary'].get('estimated_monthly_cost', 'N/A')}")
                print(f"      Move-in cost estimate: {figures['summary'].get('move_in_cost', 'N/A')}")
        else:
            print(f"  ❌ Key figures extraction failed")
            return False
        
        return rag_assistant, query_results
        
    except Exception as e:
        print(f"  ❌ RAG assistant test failed: {e}")
        return False

def test_end_to_end_integration():
    """Test complete end-to-end integration"""
    print("\n🔄 Testing End-to-End Integration")
    print("-" * 35)
    
    try:
        print("Running complete pipeline integration test...")
        
        # Step 1: Create test document
        test_image = create_test_lease_image()
        print("  ✅ Test document created")
        
        # Step 2: OCR Processing
        ocr_result = test_ocr_pipeline()
        if not ocr_result:
            return False
        print("  ✅ OCR processing completed")
        
        # Step 3: Vector Store Creation
        vector_result = test_vector_store()
        if not vector_result:
            return False
        vector_store, search_results = vector_result
        print("  ✅ Vector store created and populated")
        
        # Step 4: RAG Assistant Testing
        rag_result = test_rag_assistant(vector_store)
        if not rag_result:
            return False
        rag_assistant, query_results = rag_result
        print("  ✅ RAG assistant tested")
        
        # Step 5: Integration Test
        print("  🔗 Testing component integration...")
        
        # Simulate real workflow
        integration_query = "What would be my total monthly cost including rent and parking?"
        
        start_time = time.time()
        integration_response = rag_assistant.query(integration_query)
        integration_time = time.time() - start_time
        
        if integration_response and integration_response.get('answer'):
            print(f"  ✅ Integration test successful in {integration_time:.2f}s")
            print(f"      Query: {integration_query}")
            print(f"      Answer: {integration_response['answer'][:150]}...")
            print(f"      Confidence: {integration_response['confidence']:.3f}")
        else:
            print("  ❌ Integration test failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ End-to-end integration test failed: {e}")
        return False

def run_performance_benchmarks():
    """Run performance benchmarks"""
    print("\n⚡ Performance Benchmarks")
    print("-" * 25)
    
    try:
        # Create vector store with sample data
        vector_store = LeaseVectorStore()
        
        # Large sample text for performance testing
        large_sample = """
        COMPREHENSIVE LEASE AGREEMENT
        
        This agreement contains multiple sections covering various aspects of the rental agreement.
        """ * 50  # Repeat to create larger document
        
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
            "Who handles maintenance?"
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
