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

        logger.info(f"Processing OCR request with body keys: {list(body.keys())}")

        # Validate input
        if "image_base64" not in body:
            return create_error_response(400, "Missing 'image_base64' in request body")

        # Decode image data
        try:
            image_data = base64.b64decode(body["image_base64"])
            logger.info(f"Decoded image data: {len(image_data)} bytes")
        except Exception as e:
            return create_error_response(400, f"Invalid base64 image data: {str(e)}")

        # Get processing options
        options = body.get("options", {})
        feature_types = options.get("feature_types", [])

        # Initialize Textract client
        textract = boto3.client("textract")

        # Choose Textract operation based on feature types
        if feature_types:
            # Use analyze_document for advanced features
            response = textract.analyze_document(Document={"Bytes": image_data}, FeatureTypes=feature_types)
            operation = "analyze_document"
        else:
            # Use detect_document_text for basic OCR
            response = textract.detect_document_text(Document={"Bytes": image_data})
            operation = "detect_document_text"

        logger.info(f"Textract {operation} completed successfully")

        # Process response
        result = process_textract_response(response, operation)

        return create_success_response(result)

    except Exception as e:
        logger.error(f"Lambda execution failed: {str(e)}")
        return create_error_response(500, f"Internal server error: {str(e)}")


def process_textract_response(response: Dict[str, Any], operation: str) -> Dict[str, Any]:
    """Process Textract response and extract relevant information"""

    blocks = response.get("Blocks", [])

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
