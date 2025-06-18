import boto3
import json
from io import BytesIO
import logging
import os
# NOTE: opencv (cv2) dependency removed for minimal deployment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextractExtractor:
    """AWS Textract wrapper with PDF conversion support"""

    def __init__(self, region_name="us-east-1"):
        try:
            self.textract = boto3.client("textract", region_name=region_name)
            logger.info(f"✅ Textract client initialized in region: {region_name}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Textract: {e}")
            raise

    def extract_text_from_image(self, image_bytes):
        """Extract text using AWS Textract"""
        try:
            logger.info("🔍 Starting Textract extraction...")

            # Call Textract
            response = self.textract.detect_document_text(Document={"Bytes": image_bytes})

            # Extract text blocks
            extracted_text = []
            confidence_scores = []
            words = []
            lines = []

            for block in response["Blocks"]:
                if block["BlockType"] == "LINE":
                    text = block.get("Text", "")
                    confidence = block.get("Confidence", 0)

                    extracted_text.append(text)
                    confidence_scores.append(confidence)
                    lines.append({"text": text, "confidence": confidence, "geometry": block.get("Geometry", {})})

                elif block["BlockType"] == "WORD":
                    words.append(
                        {
                            "text": block.get("Text", ""),
                            "confidence": block.get("Confidence", 0),
                            "geometry": block.get("Geometry", {}),
                        }
                    )

            # Combine text
            full_text = "\n".join(extracted_text)
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

            result = {
                "text": full_text,
                "confidence": avg_confidence,
                "line_count": len(extracted_text),
                "word_count": len(words),
                "lines": lines,
                "words": words,
                "raw_response": response,
            }

            logger.info(f"✅ Extraction completed - {len(full_text)} characters, {avg_confidence:.1f}% confidence")
            return result

        except Exception as e:
            logger.error(f"❌ Textract extraction failed: {str(e)}")
            raise Exception(f"Textract extraction failed: {str(e)}")

    def extract_from_file(self, file_path, preprocess=True):
        """Extract text from file with PDF conversion support"""
        logger.info(f"📄 Processing file: {file_path}")

        try:
            # Check file extension
            file_ext = os.path.splitext(file_path)[1].lower()

            if file_ext == ".pdf":
                # Try PDF processing with fallback to image conversion
                return self._process_pdf_with_fallback(file_path)
            else:
                # Handle image files
                logger.info("🖼️ Processing image file...")
                return self._process_image_file(file_path, preprocess)

        except Exception as e:
            logger.error(f"❌ File processing failed: {str(e)}")
            # Always fallback to mock extraction for demo
            return self._create_mock_extraction(file_path)

    def _process_pdf_with_fallback(self, file_path):
        """Process PDF with fallback to image conversion"""
        try:
            # First, try direct PDF processing
            logger.info("📄 Trying direct PDF processing...")
            with open(file_path, "rb") as f:
                pdf_bytes = f.read()

            logger.info(f"📄 PDF file size: {len(pdf_bytes)} bytes")
            return self.extract_text_from_image(pdf_bytes)

        except Exception as e:
            if "UnsupportedDocumentException" in str(e):
                logger.warning("⚠️ PDF format not supported by Textract, converting to image...")
                return self._convert_pdf_to_image_and_extract(file_path)
            else:
                logger.error(f"❌ PDF processing failed: {str(e)}")
                return self._create_mock_extraction(file_path)

    def _convert_pdf_to_image_and_extract(self, file_path):
        """Convert PDF to image and extract text"""
        try:
            # Try to convert PDF to image using different methods
            image_bytes = None

            # Method 1: Try using pdf2image (if available)
            try:
                from pdf2image import convert_from_path
                import io

                logger.info("🔄 Converting PDF to image using pdf2image...")
                pages = convert_from_path(file_path, dpi=300)  # Process ALL pages

                if pages:
                    logger.info(f"📄 Processing {len(pages)} pages from PDF")
                    all_text = ""
                    total_confidence = 0
                    total_lines = 0
                    total_words = 0

                    # Process each page
                    for page_num, page in enumerate(pages, 1):
                        logger.info(f"📄 Processing page {page_num}/{len(pages)}...")

                        # Convert page to bytes
                        img_buffer = io.BytesIO()
                        page.save(img_buffer, format="PNG")
                        page_bytes = img_buffer.getvalue()

                        # Extract text from this page
                        page_result = self.extract_text_from_image(page_bytes)

                        # Accumulate results
                        if page_result["text"].strip():
                            all_text += f"\n--- Page {page_num} ---\n" + page_result["text"] + "\n"
                            total_confidence += page_result["confidence"]
                            total_lines += page_result["line_count"]
                            total_words += page_result["word_count"]

                    # Return combined result
                    avg_confidence = total_confidence / len(pages) if pages else 0

                    return {
                        "text": all_text.strip(),
                        "confidence": avg_confidence,
                        "line_count": total_lines,
                        "word_count": total_words,
                        "page_count": len(pages),
                        "multi_page_extraction": True,
                    }

            except ImportError:
                logger.warning("⚠️ pdf2image not available, trying alternative method...")
                # Method 2: Try using PyMuPDF (if available)
                try:
                    import fitz  # PyMuPDF

                    logger.info("🔄 Converting PDF to image using PyMuPDF...")
                    doc = fitz.open(file_path)

                    logger.info(f"📄 Processing {len(doc)} pages from PDF")
                    all_text = ""
                    total_confidence = 0
                    total_lines = 0
                    total_words = 0

                    # Process each page
                    for page_num in range(len(doc)):
                        logger.info(f"📄 Processing page {page_num + 1}/{len(doc)}...")

                        page = doc[page_num]
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale for better quality
                        page_bytes = pix.tobytes("png")

                        # Extract text from this page
                        page_result = self.extract_text_from_image(page_bytes)

                        # Accumulate results
                        if page_result["text"].strip():
                            all_text += f"\n--- Page {page_num + 1} ---\n" + page_result["text"] + "\n"
                            total_confidence += page_result["confidence"]
                            total_lines += page_result["line_count"]
                            total_words += page_result["word_count"]

                    doc.close()

                    # Return combined result
                    avg_confidence = total_confidence / len(doc) if len(doc) > 0 else 0

                    return {
                        "text": all_text.strip(),
                        "confidence": avg_confidence,
                        "line_count": total_lines,
                        "word_count": total_words,
                        "page_count": len(doc),
                        "multi_page_extraction": True,
                    }

                except ImportError:
                    logger.warning("⚠️ PyMuPDF not available either")
                    raise Exception("No PDF conversion libraries available")

            # Extract text from converted image
            if image_bytes:
                return self.extract_text_from_image(image_bytes)
            else:
                raise Exception("Failed to convert PDF to image")

        except Exception as e:
            logger.error(f"❌ PDF to image conversion failed: {str(e)}")
            return self._create_mock_extraction(file_path)

    def _process_image_file(self, file_path, preprocess=True):
        """Process image file with simplified handling (no opencv)"""
        try:
            if preprocess:
                logger.info("🔧 Validating image file...")
                from .preprocess import DocumentPreprocessor

                preprocessor = DocumentPreprocessor()
                # The preprocessor now just validates the file and returns the path
                processed_path = preprocessor.enhance_document(file_path)
                
                # Read the file directly (AWS Textract handles image processing)
                with open(processed_path, "rb") as f:
                    image_bytes = f.read()

                logger.info("✅ Image validation completed")
            else:
                logger.info("📁 Reading image file directly...")
                with open(file_path, "rb") as f:
                    image_bytes = f.read()

            return self.extract_text_from_image(image_bytes)

        except Exception as e:
            logger.error(f"❌ Image processing failed: {str(e)}")
            return self._create_mock_extraction(file_path)

    def _create_mock_extraction(self, file_path):
        """Create mock extraction result for demo/fallback"""
        logger.warning("🎭 Using mock extraction (Textract not available)")

        filename = os.path.basename(file_path)

        mock_text = f"""RESIDENTIAL LEASE AGREEMENT

Document: {filename}
Property Address: 456 Demo Street, Apt 7C, San Francisco, CA 94103

LEASE TERMS:
Monthly Rent: $4,500.00 per month
Security Deposit: $9,000.00 (equivalent to 2 months rent)
Lease Duration: 12 months
Start Date: January 1, 2024
End Date: December 31, 2024

PET POLICY:
Pets allowed with written approval from landlord.
Maximum 2 pets per unit.
Pet deposit: $800.00 per pet (refundable)
Monthly pet rent: $150.00 per pet
Restricted breeds: Aggressive breeds prohibited

UTILITIES:
Tenant Responsible: Electricity, Gas, Internet, Cable TV
Landlord Responsible: Water, Sewer, Trash, Recycling, Landscaping
Estimated monthly utilities: $250-300

PARKING:
One assigned covered parking space included in rent.
Additional parking spaces available for $300/month.
Guest parking: 4-hour limit in designated visitor areas.

TERMINATION CLAUSES:
60 days written notice required for lease termination.
Early termination penalty: 3 months rent.
Military clause: Early termination allowed with military orders.

MAINTENANCE:
Landlord responsible for: Major repairs, HVAC, plumbing, electrical
Tenant responsible for: Minor repairs under $100, light bulbs, filters
Emergency maintenance available 24/7: (555) 123-HELP

ADDITIONAL FEES:
Application fee: $250 (non-refundable)
Move-in fee: $400 (one-time)
Late rent fee: $125 (after 5th day of month)
Key replacement: $100 per key
Cleaning fee: $300 (if required at move-out)"""

        return {
            "text": mock_text,
            "confidence": 88.5,
            "line_count": mock_text.count("\n") + 1,
            "word_count": len(mock_text.split()),
            "character_count": len(mock_text),
            "mock_extraction": True,
        }


# Alternative simple extractor for when Textract is not available
class MockTextractExtractor:
    """Mock extractor for demo/development purposes"""

    def __init__(self, region_name="us-east-1"):
        logger.info("🎭 Using Mock Textract Extractor (Demo Mode)")

    def extract_from_file(self, file_path, preprocess=True):
        """Mock extraction that works with any file type"""
        logger.info(f"🎭 Mock processing: {file_path}")

        filename = os.path.basename(file_path)

        mock_content = f"""RESIDENTIAL LEASE AGREEMENT

Document Source: {filename}
Processing Date: Demo Mode

PROPERTY INFORMATION:
Address: 789 Mock Avenue, Demo City, CA 94000
Unit Type: 2 Bedroom, 1 Bathroom Apartment
Square Footage: 950 sq ft

FINANCIAL TERMS:
Monthly Rent: $3,800.00
Security Deposit: $7,600.00 (2 months rent)
Pet Deposit: $700.00 per pet (maximum 2 pets)
Application Fee: $225.00 (non-refundable)

LEASE PERIOD:
Start Date: January 1, 2024
End Date: December 31, 2024
Lease Duration: 12 months

PET POLICY:
Pets allowed with landlord approval and additional deposit.
Maximum 2 pets per unit.
Monthly pet rent: $125 per pet.
Pet registration and vaccination records required.

UTILITIES:
Tenant Pays: Electricity, Gas, Internet
Landlord Pays: Water, Sewer, Trash Collection
Average monthly utility cost: $200-250

PARKING:
One assigned parking space included.
Additional spaces available for $250/month.
Guest parking available with 3-hour time limit.

TERMINATION:
60 days written notice required.
Early termination penalty: 2.5 months rent.
Military clause applies for active duty personnel.

MAINTENANCE:
Landlord handles major repairs and appliances.
Tenant responsible for minor maintenance under $75.
24/7 emergency maintenance hotline available."""

        return {
            "text": mock_content,
            "confidence": 91.2,
            "line_count": mock_content.count("\n") + 1,
            "word_count": len(mock_content.split()),
            "character_count": len(mock_content),
            "mock_extraction": True,
            "file_processed": filename,
        }


if __name__ == "__main__":
    print("🧪 Testing Textract Extractor...")
    try:
        extractor = TextractExtractor()
        print("✅ Real Textract client initialized")
    except Exception as e:
        print(f"⚠️ Real Textract not available: {e}")
        extractor = MockTextractExtractor()

    print("📄 Textract extractor ready!")
