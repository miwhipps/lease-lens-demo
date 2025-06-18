# 🔍 LeaseLens OCR Debugging Guide

## Overview
Comprehensive debugging has been added throughout the LeaseLens OCR processing pipeline to help diagnose extraction mismatches and processing issues.

## Debugging Features Added

### 1. File Upload Debugging (`streamlit_app.py`)
**Location:** File upload section after successful upload

**Features:**
- **File Information:** Name, size, type, dimensions (for images)
- **Image Preview:** Live preview of uploaded images (when PIL available)
- **File Format Detection:** Automatic detection of PDF vs image files
- **File Validation:** Checks for file existence and integrity

**Access:** Expandable "🔍 Debug: File Upload Details" section

### 2. OCR Processing Debugging (`streamlit_app.py`)
**Location:** During document processing workflow

**Features:**
- **Processing Parameters:** Shows chunk size, search K, preprocessing settings
- **OCR Engine Selection:** Real Textract vs Mock extractor with error details
- **Processing Steps:** Real-time progress with detailed status
- **Extraction Results:** Character/word counts, confidence scores, keyword detection
- **Content Analysis:** Checks for rent keywords, dollar signs, numbers
- **Text Preview:** First 300 characters of extracted text

**Access:** Live debug output during processing + persistent debug info

### 3. AWS Lambda Function Debugging (`ocr_pipeline/lambda_function.py`)
**Location:** Lambda handler and processing functions

**Features:**
- **Input Analysis:** Event structure, body keys, data types
- **Base64 Decoding:** Data length, format detection, validation
- **File Format Detection:** PNG, JPEG, PDF header analysis
- **Textract Processing:** Operation selection, block analysis, response details
- **Text Extraction:** Line/word counts, confidence analysis, content preview
- **Lease Term Detection:** Automatic search for rent-related keywords

**Access:** CloudWatch logs with 🔍 DEBUG prefixed messages

### 4. Textract Extractor Debugging (`ocr_pipeline/textract_extract.py`)
**Location:** File processing methods

**Features:**
- **File Validation:** Existence checks, size analysis, extension detection
- **Processing Method:** PDF vs image processing paths
- **Error Handling:** Detailed error messages with fallback information
- **Multi-page Processing:** Page count and processing status

**Access:** Application logs with detailed processing information

### 5. Comprehensive Debug Display (`streamlit_app.py`)
**Location:** Persistent debug panel after processing

**Features:**
- **📁 File Information:** Original + temp file details, processing time
- **🔧 OCR Engine Details:** Engine type, preprocessing status, OpenCV availability
- **📊 Extraction Results:** Characters, words, pages, confidence, word density
- **🔍 Content Analysis:** Keyword detection, content flags, quality indicators
- **🔬 AWS Textract Debug:** Raw Textract response analysis (when available)
- **📝 Text Preview:** Full extracted text with length information

**Access:** "🔍 COMPREHENSIVE DEBUG INFORMATION" expandable panel

## How to Use for Troubleshooting

### Step 1: Check File Upload
1. Upload your document
2. Expand "🔍 Debug: File Upload Details"
3. Verify:
   - File size is reasonable (not 0 bytes)
   - File type is correct (PDF/image)
   - Image preview displays correctly (for images)

### Step 2: Monitor OCR Processing
1. Click "🚀 Process Document"
2. Watch the real-time debug output for:
   - OCR engine selection (Real Textract vs Mock)
   - File processing steps
   - Character/word extraction counts
   - Keyword detection results

### Step 3: Analyze Results
1. After processing, expand "🔍 COMPREHENSIVE DEBUG INFORMATION"
2. Check:
   - **Extraction quality:** High character/word counts indicate good OCR
   - **Content flags:** Should see ✅ for $ signs and numbers in lease documents
   - **Keyword detection:** Should find rent-related terms
   - **Text preview:** Manually verify extracted content looks correct

### Step 4: Check Lambda Logs (if using real Textract)
1. Go to AWS CloudWatch
2. Find your Lambda function logs
3. Look for 🔍 DEBUG messages showing:
   - File format detection
   - Textract block analysis
   - Extraction quality metrics

## Common Issues and Solutions

### Issue: No text extracted (0 characters)
**Debug Steps:**
1. Check file upload section - is file size > 0?
2. Check OCR engine - is it using Mock vs Real Textract?
3. Check Lambda logs - any errors in Textract processing?
4. Check file format - is it a supported type?

### Issue: Wrong text extracted
**Debug Steps:**
1. Check image preview - does the preview look correct?
2. Check extraction preview - compare with expected content
3. Check preprocessing settings - try with/without OpenCV enhancement
4. Check confidence scores - low confidence indicates OCR issues

### Issue: Missing rent information
**Debug Steps:**
1. Check content analysis - are keywords being detected?
2. Check content flags - are $ signs and numbers present?
3. Check text preview - manually search for rent terms
4. Check chunking - rent info might be split across chunks

### Issue: Mock extractor being used instead of real Textract
**Debug Steps:**
1. Check OCR engine selection debug output
2. Check error details shown in debug panel
3. Verify AWS credentials are configured
4. Check Textract service availability in your region

## Debug Output Examples

### Successful Processing:
```
✅ Successfully initialized: AWS Textract
• Text extracted: 2,847 characters
• Word count: 423 words
• Confidence: 94.2%
• Contains 'rent': True
• Contains '$': True
✅ Rent keywords found: rent, monthly, $, payment
```

### Problem Detection:
```
❌ No text was extracted from the document!
⚠️ Fallback to: Mock OCR (Demo)
❌ No rent keywords found! This indicates a potential OCR problem.
❌ NO TEXT EXTRACTED! This is a critical problem.
```

## Performance Monitoring

The debug system also tracks:
- **Processing time:** End-to-end document processing duration
- **File sizes:** Original vs temporary file size comparison
- **Word density:** Characters per word ratio (quality indicator)
- **Chunk distribution:** How text is split for vector storage

This comprehensive debugging system will help identify exactly where OCR extraction issues occur and provide actionable information for resolution.