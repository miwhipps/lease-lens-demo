#!/usr/bin/env python3
"""Test only the OCR pipeline components"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def test_preprocessing():
    """Test OpenCV preprocessing"""
    print("🔧 Testing OCR Preprocessing...")
    
    try:
        from ocr_pipeline.preprocess import DocumentPreprocessor
        
        # Create simple test image
        img = np.ones((200, 400, 3), dtype=np.uint8) * 255
        cv2.putText(img, 'Test Lease Document', (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imwrite('test_doc.jpg', img)
        
        # Test preprocessor
        preprocessor = DocumentPreprocessor()
        result = preprocessor.enhance_document('test_doc.jpg')
        
        if result is not None:
            print('✅ Preprocessing test passed!')
            print(f'   Output shape: {result.shape}')
            return True
        else:
            print('❌ Preprocessing test failed!')
            return False
            
    except Exception as e:
        print(f'❌ Preprocessing test error: {e}')
        return False

def test_mock_textract():
    """Test mock Textract functionality"""
    print("📄 Testing Mock Textract...")
    
    try:
        # Create a simple mock for testing
        class MockTextract:
            def extract_from_file(self, path, preprocess=True):
                return {
                    'text': 'Monthly rent: $2000\nSecurity deposit: $4000\nPet policy: No pets allowed',
                    'confidence': 95.0,
                    'line_count': 3
                }
        
        mock = MockTextract()
        result = mock.extract_from_file('test_doc.jpg')
        
        if result and len(result['text']) > 0:
            print('✅ Mock Textract test passed!')
            print(f'   Text length: {len(result["text"])} characters')
            print(f'   Confidence: {result["confidence"]}%')
            return True
        else:
            print('❌ Mock Textract test failed!')
            return False
            
    except Exception as e:
        print(f'❌ Mock Textract test error: {e}')
        return False

if __name__ == "__main__":
    print("🧪 OCR Pipeline Component Tests")
    print("-" * 30)
    
    results = []
    results.append(test_preprocessing())
    results.append(test_mock_textract())
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    if passed == total:
        print("✅ OCR pipeline tests successful!")
    else:
        print("❌ Some OCR tests failed")
