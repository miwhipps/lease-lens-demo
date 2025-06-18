# NOTE: OpenCV functionality disabled for minimal deployment
# Original OpenCV-based preprocessing has been replaced with simple file handling
import os
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentPreprocessor:
    """Simplified document preprocessor without OpenCV dependencies"""

    def __init__(self):
        self.processed_files = []

    def enhance_document(self, file_path: str) -> Optional[str]:
        """Simplified document processing without OpenCV dependencies
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Path to the processed file (same as input for now)
        """
        logger.info(f"📄 Processing document: {file_path}")
        
        # Verify file exists
        if not os.path.exists(file_path):
            raise ValueError(f"Could not load file from {file_path}")
            
        # For minimal deployment, we simply validate the file and pass it through
        # AWS Textract will handle the actual image processing
        file_size = os.path.getsize(file_path)
        
        # Store processing info
        self.processed_files.append({
            "path": file_path,
            "size_bytes": file_size,
            "processed": True
        })
        
        logger.info(f"✅ Document validated - {file_size} bytes")
        logger.info("ℹ️  Using AWS Textract for image processing (no local preprocessing)")
        
        return file_path

    def get_processing_stats(self) -> dict:
        """Get statistics about processed files"""
        if not self.processed_files:
            return {"total_files": 0, "total_size_bytes": 0}
            
        return {
            "total_files": len(self.processed_files),
            "total_size_bytes": sum(f["size_bytes"] for f in self.processed_files),
            "files": self.processed_files
        }

    def save_processing_log(self, output_path: str):
        """Save processing log for demo purposes"""
        if not self.processed_files:
            logger.warning("No processed files to log")
            return
            
        try:
            os.makedirs(output_path, exist_ok=True)
            
            log_file = os.path.join(output_path, "processing_log.txt")
            with open(log_file, 'w') as f:
                f.write("Document Processing Log\n")
                f.write("=" * 30 + "\n\n")
                
                for i, file_data in enumerate(self.processed_files):
                    f.write(f"File {i+1}:\n")
                    f.write(f"  Path: {file_data['path']}\n")
                    f.write(f"  Size: {file_data['size_bytes']} bytes\n")
                    f.write(f"  Processed: {file_data['processed']}\n\n")
                    
            logger.info(f"Saved processing log to {log_file}")
            
        except Exception as e:
            logger.error(f"Failed to save processing log: {e}")


# Simple test function
def test_preprocessing():
    """Test the simplified preprocessing functionality"""
    print("🧪 Testing Document Preprocessor (No OpenCV)...")

    try:
        # Create a simple test file
        test_content = "Test Document\nMonthly Rent: $2,000\nSecurity Deposit: $4,000"
        test_file = "simple_test.txt"
        
        with open(test_file, 'w') as f:
            f.write(test_content)

        # Test preprocessor
        preprocessor = DocumentPreprocessor()
        result = preprocessor.enhance_document(test_file)

        if result is not None:
            print("✅ Preprocessing test passed!")
            print(f"   Output file: {result}")
            print(f"   File exists: {os.path.exists(result)}")
            
            # Get stats
            stats = preprocessor.get_processing_stats()
            print(f"   Processing stats: {stats}")
            
            # Clean up
            if os.path.exists(test_file):
                os.remove(test_file)

            return True
        else:
            print("❌ Preprocessing returned None")
            return False

    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_preprocessing()
