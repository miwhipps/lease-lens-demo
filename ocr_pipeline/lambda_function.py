import json
import boto3
import base64
import logging
from typing import Dict, Any

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    AWS Lambda handler for OCR processing

    Expected event structure:
    {
        "body": {
            "image_base64": "base64_encoded_image_data",
            "options": {
                "feature_types": ["FORMS", "TABLES"] // optional
            }
        }
    }
    """

    try:
        # Parse input
        if isinstance(event.get("body"), str):
            body = json.loads(event["body"])
        else:
            body = event.get("body", {})

        # 🔍 DEBUGGING: Comprehensive input analysis
        logger.info(f"🔍 DEBUG: Processing OCR request")
        logger.info(f"🔍 DEBUG: Event keys: {list(event.keys())}")
        logger.info(f"🔍 DEBUG: Body keys: {list(body.keys())}")
        logger.info(f"🔍 DEBUG: Body type: {type(body)}")
        
        # Validate input
        if "image_base64" not in body:
            logger.error("🔍 DEBUG: Missing 'image_base64' in request body")
            return create_error_response(400, "Missing 'image_base64' in request body")

        # Decode image data with detailed logging
        try:
            base64_data = body["image_base64"]
            logger.info(f"🔍 DEBUG: Base64 data length: {len(base64_data)} characters")
            logger.info(f"🔍 DEBUG: Base64 preview: {base64_data[:50]}...")
            
            image_data = base64.b64decode(base64_data)
            logger.info(f"🔍 DEBUG: Decoded image data: {len(image_data)} bytes")
            
            # Check for valid image headers
            if image_data.startswith(b'\x89PNG'):
                logger.info("🔍 DEBUG: Detected PNG image format")
            elif image_data.startswith(b'\xff\xd8\xff'):
                logger.info("🔍 DEBUG: Detected JPEG image format")
            elif image_data.startswith(b'%PDF'):
                logger.info("🔍 DEBUG: Detected PDF format")
            else:
                logger.warning(f"🔍 DEBUG: Unknown file format, first 10 bytes: {image_data[:10]}")
                
        except Exception as e:
            logger.error(f"🔍 DEBUG: Base64 decode error: {str(e)}")
            return create_error_response(400, f"Invalid base64 image data: {str(e)}")

        # Get processing options with debugging
        options = body.get("options", {})
        feature_types = options.get("feature_types", [])
        logger.info(f"🔍 DEBUG: Processing options: {options}")
        logger.info(f"🔍 DEBUG: Feature types: {feature_types}")

        # Initialize Textract client
        logger.info("🔍 DEBUG: Initializing Textract client")
        textract = boto3.client("textract")
        logger.info("🔍 DEBUG: Textract client initialized successfully")

        # Choose Textract operation based on feature types
        if feature_types:
            # Use analyze_document for advanced features
            operation = "analyze_document"
            logger.info(f"🔍 DEBUG: Using {operation} with features: {feature_types}")
            response = textract.analyze_document(Document={"Bytes": image_data}, FeatureTypes=feature_types)
        else:
            # Use detect_document_text for basic OCR
            operation = "detect_document_text"
            logger.info(f"🔍 DEBUG: Using {operation} for basic text extraction")
            response = textract.detect_document_text(Document={"Bytes": image_data})

        logger.info(f"🔍 DEBUG: Textract {operation} completed successfully")
        logger.info(f"🔍 DEBUG: Response contains {len(response.get('Blocks', []))} blocks")

        # Process response
        logger.info("🔍 DEBUG: Processing Textract response")
        result = process_textract_response(response, operation)
        logger.info(f"🔍 DEBUG: Extracted {len(result.get('text', ''))} characters")
        logger.info(f"🔍 DEBUG: Confidence: {result.get('confidence', 0):.1f}%")

        return create_success_response(result)

    except Exception as e:
        logger.error(f"Lambda execution failed: {str(e)}")
        return create_error_response(500, f"Internal server error: {str(e)}")


def process_textract_response(response: Dict[str, Any], operation: str) -> Dict[str, Any]:
    """Process Textract response and extract relevant information"""

    blocks = response.get("Blocks", [])
    logger.info(f"🔍 DEBUG: Processing {len(blocks)} blocks from Textract response")
    
    # Count block types for debugging
    block_types = {}
    for block in blocks:
        block_type = block.get("BlockType", "UNKNOWN")
        block_types[block_type] = block_types.get(block_type, 0) + 1
    logger.info(f"🔍 DEBUG: Block types found: {block_types}")

    # Extract text content
    lines = []
    words = []
    confidence_scores = []

    # Extract forms and tables if available
    forms = []
    tables = []

    for block in blocks:
        block_type = block.get("BlockType")
        confidence = block.get("Confidence", 0)

        if block_type == "LINE":
            text = block.get("Text", "")
            lines.append(text)
            confidence_scores.append(confidence)

        elif block_type == "WORD":
            text = block.get("Text", "")
            words.append({"text": text, "confidence": confidence, "geometry": block.get("Geometry", {})})

        elif block_type == "KEY_VALUE_SET":
            # Process form fields
            entity_types = block.get("EntityTypes", [])
            if "KEY" in entity_types:
                forms.append({"type": "key", "text": block.get("Text", ""), "confidence": confidence, "id": block.get("Id")})
            elif "VALUE" in entity_types:
                forms.append({"type": "value", "text": block.get("Text", ""), "confidence": confidence, "id": block.get("Id")})

        elif block_type == "TABLE":
            tables.append({"confidence": confidence, "geometry": block.get("Geometry", {}), "id": block.get("Id")})

    # Combine extracted text
    full_text = "\n".join(lines)
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    
    # 🔍 DEBUGGING: Final extraction results
    logger.info(f"🔍 DEBUG: Text extraction summary:")
    logger.info(f"🔍 DEBUG:   Lines extracted: {len(lines)}")
    logger.info(f"🔍 DEBUG:   Words extracted: {len(words)}")
    logger.info(f"🔍 DEBUG:   Total characters: {len(full_text)}")
    logger.info(f"🔍 DEBUG:   Average confidence: {avg_confidence:.1f}%")
    logger.info(f"🔍 DEBUG:   Form fields found: {len(forms)}")
    logger.info(f"🔍 DEBUG:   Tables found: {len(tables)}")
    
    # Check for key lease terms in extracted text
    rent_indicators = ["rent", "monthly", "$", "payment", "lease"]
    found_indicators = [term for term in rent_indicators if term.lower() in full_text.lower()]
    logger.info(f"🔍 DEBUG:   Lease indicators found: {found_indicators}")
    
    if len(full_text) > 0:
        logger.info(f"🔍 DEBUG:   First 200 chars: {full_text[:200]}")
    else:
        logger.warning("🔍 DEBUG:   NO TEXT EXTRACTED - This is a problem!")

    # Prepare result
    result = {
        "text": full_text,
        "confidence": avg_confidence,
        "statistics": {
            "line_count": len(lines),
            "word_count": len(words),
            "character_count": len(full_text),
            "form_fields": len(forms),
            "tables": len(tables),
        },
        "operation": operation,
        "success": True,
        "debug_info": {
            "block_types": block_types,
            "lease_indicators": found_indicators,
            "extraction_quality": "good" if len(full_text) > 100 else "poor"
        }
    }

    # Add detailed information if requested
    if operation == "analyze_document":
        result["detailed"] = {"words": words, "forms": forms, "tables": tables}

    return result


def create_success_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a successful Lambda response"""
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
        },
        "body": json.dumps(data),
    }


def create_error_response(status_code: int, message: str) -> Dict[str, Any]:
    """Create an error Lambda response"""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
        },
        "body": json.dumps({"error": message, "success": False}),
    }


# Local testing function
def test_lambda_locally():
    """Test the Lambda function locally without opencv"""
    # Create test text content instead of image
    test_text_content = """
Monthly Rent: $2,000
Security Deposit: $4,000
Lease Duration: 12 months
Pet Policy: No pets allowed
"""
    
    # Create a simple mock image as base64 (minimal PNG header + data)
    # This is just for testing - in production, real images would be provided
    mock_png_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x00\x01\x00\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    image_base64 = base64.b64encode(mock_png_bytes).decode("utf-8")

    # Create test event
    test_event = {"body": {"image_base64": image_base64, "options": {}}}

    # Test the function
    try:
        result = lambda_handler(test_event, None)
        print("✅ Lambda test successful!")
        print(f"Status: {result['statusCode']}")

        if result["statusCode"] == 200:
            body = json.loads(result["body"])
            print(f"Text extracted: {body['text'][:100]}...")
            print(f"Confidence: {body['confidence']:.1f}%")
        else:
            print(f"Error: {result['body']}")

        return result

    except Exception as e:
        print(f"❌ Lambda test failed: {e}")
        return None


if __name__ == "__main__":
    print("🧪 Testing Lambda function locally...")
    test_lambda_locally()
