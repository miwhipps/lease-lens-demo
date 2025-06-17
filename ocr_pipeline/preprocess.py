import cv2
import numpy as np
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentPreprocessor:
    """OpenCV-based document preprocessing for better OCR results"""

    def __init__(self):
        self.processed_images = []

    def enhance_document(self, image_path):
        """Apply OpenCV preprocessing to improve OCR accuracy"""
        logger.info(f"Processing image: {image_path}")

        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Store original for comparison
        original = img.copy()

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Deskew if needed (with proper error handling)
        try:
            deskewed = self._deskew(cleaned)
        except Exception as e:
            logger.warning(f"Deskewing failed, using cleaned image: {e}")
            deskewed = cleaned

        # Store results for comparison
        self.processed_images.append({"original": original, "processed": deskewed, "path": image_path})

        logger.info("✅ Image preprocessing completed")
        return deskewed

    def _deskew(self, image):
        """Simple deskewing using Hough transform with proper error handling"""
        try:
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

            if lines is not None and len(lines) > 0:
                angles = []

                # Handle different OpenCV versions - lines might be 2D or 3D array
                for line in lines[:10]:  # Use first 10 lines
                    if len(line.shape) == 1:
                        # OpenCV 4.x format: line is [rho, theta]
                        rho, theta = line[0], line[1]
                    else:
                        # OpenCV 3.x format: line is [[rho, theta]]
                        rho, theta = line[0][0], line[0][1]

                    angle = theta * 180 / np.pi
                    if abs(angle - 90) < 45:  # Near vertical lines
                        angles.append(angle - 90)

                if angles:
                    median_angle = np.median(angles)
                    if abs(median_angle) > 0.5:  # Only rotate if significant skew
                        (h, w) = image.shape[:2]
                        center = (w // 2, h // 2)
                        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                        return rotated

            return image

        except Exception as e:
            logger.warning(f"Deskewing failed: {e}")
            return image

    def save_comparison(self, output_path):
        """Save before/after comparison for demo purposes"""
        if not self.processed_images:
            logger.warning("No processed images to save")
            return

        try:
            import os

            os.makedirs(output_path, exist_ok=True)

            for i, img_data in enumerate(self.processed_images):
                # Create side-by-side comparison
                original = cv2.resize(img_data["original"], (400, 600))
                processed = cv2.resize(img_data["processed"], (400, 600))

                # Convert processed back to 3-channel for concatenation
                if len(processed.shape) == 2:
                    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

                comparison = np.hstack([original, processed])

                # Add labels
                cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(comparison, "Processed", (410, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                output_file = f"{output_path}/comparison_{i}.jpg"
                cv2.imwrite(output_file, comparison)
                logger.info(f"Saved comparison to {output_file}")

        except Exception as e:
            logger.error(f"Failed to save comparison: {e}")


# Simple test function
def test_preprocessing():
    """Test the preprocessing functionality"""
    print("🧪 Testing Document Preprocessor...")

    try:
        # Create a simple test image
        test_img = np.ones((200, 400, 3), dtype=np.uint8) * 255
        cv2.putText(test_img, "Test Document", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imwrite("simple_test.png", test_img)

        # Test preprocessor
        preprocessor = DocumentPreprocessor()
        result = preprocessor.enhance_document("simple_test.png")

        if result is not None:
            print("✅ Preprocessing test passed!")
            print(f"   Output shape: {result.shape}")
            print(f"   Output type: {result.dtype}")

            # Save result for inspection
            cv2.imwrite("preprocessed_result.png", result)
            print("   Saved result as preprocessed_result.png")

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
