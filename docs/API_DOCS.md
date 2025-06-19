# LeaseLens API Documentation

## Overview

LeaseLens is a Streamlit web application that provides a user interface for lease document analysis. While it doesn't expose traditional REST APIs, it provides several programmatic interfaces through its Python modules and internal functions.

## Web Interface (Streamlit)

### Main Application URL
- **Local Development**: `http://localhost:8501`
- **Production**: `https://leaselens.streamlit.app` (when deployed)

### Application Modes

#### 1. Standard Mode
- Requires AWS Textract and Anthropic Claude API keys
- Full OCR and AI functionality
- Real document processing

#### 2. Demo Mode
- Works without API keys
- Uses mock data for demonstration
- All features functional with sample lease data

## Core API Interfaces

### 1. OCR Pipeline API

#### TextractExtractor Class

```python
from ocr_pipeline.textract_extract import TextractExtractor

# Initialize
extractor = TextractExtractor(region_name="us-east-1")

# Extract from file
result = extractor.extract_from_file(
    file_path: str,
    preprocess: bool = True
)
```

**Parameters**:
- `file_path` (str): Path to PDF or image file
- `preprocess` (bool): Enable preprocessing (default: True)

**Returns**:
```python
{
    "text": str,                    # Extracted text content
    "confidence": float,           # Average confidence (0-100)
    "line_count": int,            # Number of text lines
    "word_count": int,            # Number of words
    "character_count": int,       # Total characters
    "page_count": int,            # Number of pages (for PDFs)
    "mock_extraction": bool,      # True if mock data was used
    "multi_page_extraction": bool # True for multi-page PDFs
}
```

**Error Handling**:
- AWS service failures → Automatic fallback to mock extraction
- PDF conversion failures → Multiple conversion library fallbacks
- Invalid file formats → Descriptive error messages

#### DocumentPreprocessor Class

```python
from ocr_pipeline.preprocess import DocumentPreprocessor

# Initialize
preprocessor = DocumentPreprocessor()

# Enhance document
enhanced_path = preprocessor.enhance_document(file_path: str)

# Get processing statistics
stats = preprocessor.get_processing_stats()
```

### 2. Vector Store API

#### LeaseVectorStore Class

```python
from embeddings.vector_store import LeaseVectorStore

# Initialize
vector_store = LeaseVectorStore(model_name="simple_tfidf")

# Add document
vector_store.add_document(
    text: str,
    doc_id: str,
    source_info: Dict[str, Any] = None
)

# Search
results = vector_store.search(
    query: str,
    k: int = 5
)
```

**Search Results Format**:
```python
[
    {
        "text": str,              # Chunk text content
        "score": float,           # Similarity score (0-1)
        "chunk_id": str,          # Unique chunk identifier
        "doc_id": str,            # Source document ID
        "source_info": dict,      # Metadata
        "keywords": List[str]     # Extracted keywords
    }
]
```

**Search Features**:
- **Topic Boosting**: Enhanced scoring for financial, pet, utility queries
- **Keyword Extraction**: Lease-specific term detection
- **Fallback Search**: Multiple search strategies
- **Source Attribution**: Detailed provenance tracking

### 3. RAG Assistant API

#### LeaseRAGAssistant Class

```python
from ai_assistant.rag_chat import LeaseRAGAssistant

# Initialize
assistant = LeaseRAGAssistant(
    vector_store: LeaseVectorStore,
    anthropic_api_key: str = None  # Optional, uses env var
)

# Query document
response = assistant.query(
    question: str,
    k: int = 5  # Number of context chunks
)
```

**Query Response Format**:
```python
{
    "answer": str,                # Generated response
    "sources": List[Dict],        # Source chunks used
    "confidence": float,          # Response confidence (0-1)
    "processing_time": float,     # Time taken (seconds)
    "fallback_used": bool,        # True if Claude API unavailable
    "context_length": int         # Characters of context used
}
```

**Advanced Analysis Methods**:

```python
# Generate comprehensive lease summary
summary = assistant.get_lease_summary()
# Returns: Dict with categorized summary sections

# Analyze potential risks
risks = assistant.analyze_lease_risks()
# Returns: List of risk items with severity levels

# Extract key financial figures
figures = assistant.extract_key_figures()
# Returns: Dict with financial metrics and important dates
```

## Streamlit Function APIs

### Core Application Functions

#### Document Processing

```python
def process_document(
    uploaded_file,
    use_preprocessing: bool = True,
    chunk_size: int = 500,
    search_k: int = 5
) -> bool
```

**Purpose**: Process uploaded document through full pipeline

**Parameters**:
- `uploaded_file`: Streamlit UploadedFile object
- `use_preprocessing`: Enable document preprocessing
- `chunk_size`: Text chunk size for vector store
- `search_k`: Number of search results to retrieve

**Side Effects**:
- Updates `st.session_state.rag_assistant`
- Sets processing status flags
- Displays progress indicators

#### Chat Interface

```python
def handle_user_input(prompt: str) -> None
```

**Purpose**: Process user chat input and generate response

**Parameters**:
- `prompt`: User's question or input

**Side Effects**:
- Adds message to chat history
- Queries RAG assistant
- Updates UI with response

#### Sample Query Handling

```python
def handle_sample_query(query: str) -> None
```

**Purpose**: Process predefined sample questions

**Parameters**:
- `query`: Predefined question string

**Sample Queries Available**:
- "What is the monthly rent?"
- "What are the pet policies?"
- "Are there any break clauses?"
- "What is the security deposit?"
- "What utilities are included?"
- "What are the parking arrangements?"
- "Who handles maintenance?"
- "Can I sublet the property?"

### Advanced Feature Functions

#### Lease Analysis

```python
def generate_lease_summary() -> Dict[str, str]
def display_lease_summary() -> None

def generate_risk_analysis() -> List[Dict[str, Any]]
def display_risk_analysis() -> None

def extract_key_figures() -> Dict[str, Any]
def display_key_figures() -> None
```

**Lease Summary Categories**:
- **Financial Terms**: Rent, deposits, fees
- **Legal Provisions**: Termination, liability, compliance
- **Utilities & Services**: Included utilities, responsibilities
- **Property Details**: Address, features, restrictions

**Risk Analysis Levels**:
- **🔴 High Risk**: Significant concerns requiring attention
- **🟡 Medium Risk**: Moderate concerns to review
- **🟢 Low Risk**: Minor or standard provisions

**Key Figures Extracted**:
- Monthly rent amount
- Security deposit
- Total move-in cost
- Lease duration
- Important dates

### Debug and Status Functions

#### Document Processing Panel

```python
def display_document_processing_panel() -> None
```

**Purpose**: Show comprehensive processing status and metrics

**Information Displayed**:
- **File Information**: Name, size, type, pages
- **OCR Engine Details**: Service used, configuration
- **Extraction Results**: Text metrics, confidence scores
- **Content Analysis**: Keywords found, quality indicators
- **Vector Store Results**: Chunks created, search readiness

#### Debug Information

```python
def display_debug_info() -> None
```

**Purpose**: Show detailed technical information for troubleshooting

## Session State API

### Session State Structure

```python
st.session_state = {
    # Core application state
    "messages": List[Dict],           # Chat message history
    "rag_assistant": LeaseRAGAssistant,  # Initialized assistant
    "demo_mode": bool,                # Demo mode flag
    
    # Processing status
    "processing_status": {
        "file_uploaded": bool,
        "ocr_completed": bool,
        "embeddings_completed": bool,
        "rag_completed": bool
    },
    
    # Document statistics
    "document_stats": {
        "filename": str,
        "file_size": int,
        "character_count": int,
        "confidence_score": float,
        "processing_time": float
    },
    
    # Conversation analytics
    "conversation_stats": {
        "total_questions": int,
        "avg_confidence": float,
        "topics_discussed": List[str],
        "session_start": datetime
    },
    
    # Advanced features state
    "advanced_features": {
        "lease_summary": Dict,
        "risk_analysis": List[Dict],
        "key_figures": Dict
    },
    
    # Debug information
    "debug_info": Dict,              # Processing debug data
    "chunk_debug_info": Dict         # Vector store debug data
}
```

### Session State Management Functions

```python
def initialize_session_state() -> None
# Initialize all session state variables with defaults

def clean_source_display_text(text: str) -> str
# Clean extracted text for better display

def create_mock_extractor() -> MockTextractExtractor
# Create fallback mock extractor for demo mode
```

## Configuration API

### Environment Variables

```python
# AWS Configuration
AWS_ACCESS_KEY_ID          # AWS access key for Textract
AWS_SECRET_ACCESS_KEY      # AWS secret key
AWS_DEFAULT_REGION         # AWS region (default: us-east-1)

# Anthropic Configuration
ANTHROPIC_API_KEY          # Claude API key

# Application Settings
DEBUG                      # Enable debug mode (default: True)
STREAMLIT_PORT            # Port for Streamlit (default: 8501)
MAX_FILE_SIZE_MB          # Max upload size (default: 10)

# Vector Store Settings
CHUNK_SIZE                # Text chunk size (default: 500)
CHUNK_OVERLAP            # Chunk overlap (default: 50)
SEARCH_K                 # Search results count (default: 5)
```

### Runtime Configuration

```python
# Streamlit page configuration
st.set_page_config(
    page_title="LeaseLens - AI Lease Assistant",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Optional dependency flags
PIL_AVAILABLE: bool       # Pillow image processing
PANDAS_AVAILABLE: bool    # Pandas data manipulation
PLOTLY_AVAILABLE: bool    # Plotly visualizations
OPENCV_AVAILABLE: bool    # OpenCV preprocessing (disabled)
NUMPY_AVAILABLE: bool     # NumPy operations
```

## Error Handling Patterns

### Standard Error Response Format

```python
{
    "error": True,
    "error_type": str,        # "AWS_ERROR", "API_ERROR", "FILE_ERROR"
    "message": str,           # Human-readable error message
    "fallback_used": bool,    # Whether fallback was activated
    "details": dict           # Additional error context
}
```

### Common Error Scenarios

#### 1. AWS Service Errors
```python
# Textract unavailable → Mock extraction
# S3 access denied → Local file processing
# Rate limits → Retry with exponential backoff
```

#### 2. File Processing Errors
```python
# Invalid file format → User-friendly error message
# File too large → Size limit warning
# Corrupted PDF → Multiple conversion attempts
```

#### 3. API Integration Errors
```python
# Claude API down → Template-based responses
# Rate limits → Queue and retry system
# Authentication errors → Clear setup instructions
```

## Integration Examples

### Basic Document Processing

```python
import streamlit as st
from ocr_pipeline.textract_extract import TextractExtractor
from embeddings.vector_store import LeaseVectorStore
from ai_assistant.rag_chat import LeaseRAGAssistant

# Process document
extractor = TextractExtractor()
result = extractor.extract_from_file("lease.pdf")

# Create vector store
vector_store = LeaseVectorStore()
vector_store.add_document(result["text"], "doc_1")

# Initialize RAG assistant
assistant = LeaseRAGAssistant(vector_store)

# Query document
response = assistant.query("What is the monthly rent?")
print(response["answer"])
```

### Custom Analysis Pipeline

```python
# Custom lease analysis
def analyze_lease_document(file_path: str) -> Dict:
    # Extract text
    extractor = TextractExtractor()
    extraction = extractor.extract_from_file(file_path)
    
    # Create searchable index
    vector_store = LeaseVectorStore()
    vector_store.add_document(extraction["text"], "lease")
    
    # Initialize AI assistant
    assistant = LeaseRAGAssistant(vector_store)
    
    # Comprehensive analysis
    return {
        "summary": assistant.get_lease_summary(),
        "risks": assistant.analyze_lease_risks(),
        "figures": assistant.extract_key_figures(),
        "extraction_stats": extraction
    }
```

## Testing Interfaces

### Component Tests Available

```python
# Test OCR pipeline
python -m pytest tests/test_ocr.py

# Test vector store
python -m pytest tests/test_embeddings.py

# Test RAG assistant
python -m pytest tests/test_rag.py

# Test all components
python -m pytest tests/test_all_components.py
```

### Manual Testing Functions

```python
# Test document processing
from debug_document_processing import test_document_processing
test_document_processing("sample.pdf")

# Test vector store
from embeddings.vector_store import test_vector_store
test_vector_store()

# Test preprocessing
from ocr_pipeline.preprocess import test_preprocessing
test_preprocessing()
```

This API documentation reflects the actual implementation and interfaces available in the LeaseLens codebase, focusing on the Streamlit web interface and underlying Python modules that power the application.