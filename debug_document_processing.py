#!/usr/bin/env python3
"""
Debug tool to diagnose document processing issues
Run this to check what's happening when documents are uploaded
"""

import sys
from pathlib import Path
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def debug_document_processing(file_path=None):
    """Debug the full document processing pipeline"""
    
    print("🔍 Document Processing Diagnostic Tool")
    print("=" * 50)
    
    # Import components
    try:
        from ocr_pipeline.textract_extract import TextractExtractor, MockTextractExtractor
        from embeddings.vector_store import LeaseVectorStore
        from ai_assistant.rag_chat import LeaseRAGAssistant
        print("✅ All components imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Step 1: Check OCR availability
    print("\n1. OCR Availability Check:")
    try:
        extractor = TextractExtractor()
        print("   ✅ AWS Textract available")
        textract_available = True
    except Exception as e:
        print(f"   ⚠️ AWS Textract not available: {e}")
        extractor = MockTextractExtractor()
        print("   📝 Using Mock Textract")
        textract_available = False
    
    # Step 2: Process test document
    print("\n2. Document Processing Test:")
    
    if file_path and os.path.exists(file_path):
        print(f"   📄 Processing provided file: {file_path}")
        try:
            result = extractor.extract_from_file(file_path)
        except Exception as e:
            print(f"   ❌ Processing failed: {e}")
            return
    else:
        print("   📄 Using built-in test document")
        # Create a test document
        test_content = """
        RESIDENTIAL LEASE AGREEMENT
        
        RENT SECTION:
        Monthly Rent Amount: $2,400.00
        Due Date: 1st of each month
        Late Fee: $50.00 after 5th day
        
        SECURITY DEPOSIT:
        Amount: $4,800.00
        Refundable upon inspection
        """
        
        if textract_available:
            # Save test content and process
            test_file = "temp_test_lease.txt"
            with open(test_file, 'w') as f:
                f.write(test_content)
            try:
                result = extractor.extract_from_file(test_file)
                os.remove(test_file)  # Cleanup
            except Exception as e:
                print(f"   ❌ Test processing failed: {e}")
                if os.path.exists(test_file):
                    os.remove(test_file)
                return
        else:
            # Mock processing
            result = {
                'text': test_content,
                'confidence': 95.0,
                'mock_extraction': True
            }
    
    extracted_text = result['text']
    confidence = result.get('confidence', 0)
    
    print(f"   📊 Extraction Stats:")
    print(f"      - Characters: {len(extracted_text)}")
    print(f"      - Words: {len(extracted_text.split())}")
    print(f"      - Confidence: {confidence}%")
    print(f"      - Is Mock: {result.get('mock_extraction', False)}")
    
    # Check for rent keywords
    rent_keywords = ['rent', 'monthly', '$', '£', 'payment']
    found_keywords = [kw for kw in rent_keywords if kw.lower() in extracted_text.lower()]
    print(f"      - Rent keywords found: {found_keywords}")
    
    print(f"\n   📝 First 200 characters of extracted text:")
    print(f"      '{extracted_text[:200]}...'")
    
    # Step 3: Vector store processing
    print("\n3. Vector Store Processing:")
    vector_store = LeaseVectorStore()
    vector_store.add_document(extracted_text, 'debug_document', {'source': 'debug'})
    
    print(f"   📊 Chunking Results:")
    print(f"      - Total chunks: {len(vector_store.chunks)}")
    
    for i, chunk in enumerate(vector_store.chunks):
        words = len(chunk['text'].split())
        has_rent = any(word in chunk['text'].lower() for word in ['rent', 'monthly'])
        has_money = any(symbol in chunk['text'] for symbol in ['$', '£'])
        
        print(f"      - Chunk {i}: {words} words, rent_terms={has_rent}, money_symbols={has_money}")
        print(f"        Preview: '{chunk['text'][:100]}...'")
    
    # Step 4: RAG Testing
    print("\n4. RAG Query Testing:")
    rag_assistant = LeaseRAGAssistant(vector_store)
    
    test_queries = [
        "What is the monthly rent?",
        "How much is the rent?",
        "What is the rent amount?",
        "Tell me about rent"
    ]
    
    for query in test_queries:
        print(f"\n   🔍 Testing: '{query}'")
        try:
            response = rag_assistant.query(query, k=3)
            print(f"      - Sources found: {len(response['sources'])}")
            print(f"      - Confidence: {response['confidence']:.3f}")
            print(f"      - Answer: '{response['answer'][:150]}...'")
            
            if response['sources']:
                for j, source in enumerate(response['sources'], 1):
                    score = source.get('score', 0)
                    preview = source['text'][:80].replace('\n', ' ')
                    print(f"        Source {j} (score {score:.3f}): '{preview}...'")
            else:
                print("        ❌ No sources found!")
                
        except Exception as e:
            print(f"      ❌ Query failed: {e}")
    
    # Step 5: Recommendations
    print("\n5. Diagnostic Summary:")
    print("   " + "=" * 40)
    
    if not found_keywords:
        print("   🚨 ISSUE: No rent-related keywords found in extracted text")
        print("   💡 SOLUTION: Check OCR quality or document format")
    elif len(vector_store.chunks) == 0:
        print("   🚨 ISSUE: No chunks created from document")
        print("   💡 SOLUTION: Check document length or chunking parameters")
    elif not any('rent' in chunk['text'].lower() for chunk in vector_store.chunks):
        print("   🚨 ISSUE: Rent information not preserved in chunks")
        print("   💡 SOLUTION: Adjust chunking strategy")
    else:
        print("   ✅ All components working correctly")
        print("   💡 If still having issues, check Streamlit session state")
    
    print("\n🔧 To run this diagnostic on your own file:")
    print("   python debug_document_processing.py /path/to/your/lease.pdf")

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else None
    debug_document_processing(file_path)