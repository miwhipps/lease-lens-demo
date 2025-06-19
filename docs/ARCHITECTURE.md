# LeaseLens Architecture Documentation

## Overview

LeaseLens is a production-ready Streamlit web application that uses AWS Textract for OCR, Anthropic Claude for AI assistance, and a custom TF-IDF vector store for document search. The system processes lease documents and enables intelligent Q&A through a Retrieval-Augmented Generation (RAG) architecture.

## Technology Stack

### Core Dependencies (from requirements.txt)
- **Streamlit** - Web application framework
- **boto3** - AWS SDK for Textract OCR
- **anthropic** - Claude API client
- **python-dotenv** - Environment configuration
- **requests** - HTTP client
- **plotly** - Interactive visualizations
- **Pillow** - Image processing
- **PyMuPDF** - PDF processing

### Optional Dependencies (Graceful Fallbacks)
- **pandas** - Data manipulation (optional)
- **numpy** - Numerical operations (optional)
- **opencv-python** - Advanced image preprocessing (disabled for minimal deployment)

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   AWS Textract  │    │ Anthropic Claude │
│   Web App       │    │   OCR Service   │    │   AI Assistant   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Core Pipeline                            │
│  ┌───────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────┐ │
│  │   OCR     │  │ Text Chunking │  │ Vector Store │  │   RAG    │ │
│  │ Pipeline  │→ │  & Embedding  │→ │   (TF-IDF)   │→ │ Assistant │ │
│  └───────────┘  └──────────────┘  └─────────────┘  └──────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. Streamlit Application (`streamlit_app.py`)

**Main Entry Point**: Single-page Streamlit application with modular functions

**Key Functions**:
- `main()` - Application entry point and configuration
- `process_document()` - Core document processing pipeline
- `create_chat_interface()` - Interactive Q&A interface
- `show_welcome_screen()` - Feature overview and demo
- `display_document_processing_panel()` - Processing status dashboard

**Session State Management**:
```python
{
    "messages": [],  # Chat history
    "processing_status": {
        "file_uploaded": bool,
        "ocr_completed": bool,
        "embeddings_completed": bool,
        "rag_completed": bool
    },
    "rag_assistant": LeaseRAGAssistant,
    "document_stats": dict,
    "demo_mode": bool
}
```

### 2. OCR Pipeline (`ocr_pipeline/`)

#### TextractExtractor (`textract_extract.py`)
**Purpose**: AWS Textract integration with comprehensive fallbacks

**Key Methods**:
```python
class TextractExtractor:
    def extract_from_file(file_path, preprocess=True) -> dict
    def _process_pdf_with_fallback(file_path) -> dict
    def _process_pdf_with_pymupdf(file_path) -> dict
    def _convert_pdf_to_image_and_extract(file_path) -> dict
    def _create_mock_extraction(file_path) -> dict
```

**Processing Flow**:
1. **File Type Detection** - PDF vs Image
2. **PDF Processing** - Direct Textract or image conversion
3. **Multi-page Handling** - PyMuPDF page-by-page conversion
4. **Fallback System** - Mock extraction if AWS unavailable

**Output Format**:
```python
{
    "text": str,           # Extracted text
    "confidence": float,  # Average confidence (0-100)
    "line_count": int,    # Number of text lines
    "word_count": int,    # Number of words
    "page_count": int,    # Number of pages processed
    "mock_extraction": bool  # Whether mock data was used
}
```

#### DocumentPreprocessor (`preprocess.py`)
**Purpose**: File validation and preprocessing (simplified for minimal deployment)

**Key Methods**:
```python
class DocumentPreprocessor:
    def enhance_document(file_path: str) -> Optional[str]
    def get_processing_stats() -> dict
```

### 3. Vector Store (`embeddings/vector_store.py`)

#### LeaseVectorStore
**Purpose**: Custom TF-IDF based similarity search (no heavy ML dependencies)

**Key Methods**:
```python
class LeaseVectorStore:
    def chunk_text(text: str, chunk_size=200, overlap=30) -> List[str]
    def add_document(text: str, doc_id: str, source_info=None)
    def search(query: str, k=5) -> List[Dict]
    def get_document_stats() -> Dict
```

**Features**:
- **Intelligent Chunking** - Sentence-based with overlap
- **Keyword Extraction** - Lease-specific term detection
- **Topic Boosting** - Enhanced scoring for financial, pet, utility queries
- **Fallback Search** - Multiple search strategies for better recall

**Search Algorithm**:
1. Extract keywords from query and chunks
2. Calculate TF-IDF similarity scores
3. Apply topic-specific boosting (financial terms, pets, utilities)
4. Rank and return top-k results with source attribution

### 4. RAG Assistant (`ai_assistant/rag_chat.py`)

#### LeaseRAGAssistant
**Purpose**: Anthropic Claude integration with comprehensive fallbacks

**Key Methods**:
```python
class LeaseRAGAssistant:
    def query(question: str, k=5) -> Dict[str, Any]
    def get_lease_summary() -> Dict[str, str]
    def analyze_lease_risks() -> List[Dict[str, Any]]
    def extract_key_figures() -> Dict[str, Any]
```

**Query Processing Flow**:
1. **Context Retrieval** - Search vector store for relevant chunks
2. **Claude Integration** - Generate response with context
3. **Fallback System** - Template-based responses if Claude unavailable
4. **Source Attribution** - Include exact source citations

**Advanced Features**:
- **Lease Summary Generation** - Categorized summary with financial, legal, utility details
- **Risk Analysis** - Automated risk assessment with severity levels
- **Key Figure Extraction** - Financial metrics and important dates
- **Batch Processing** - Multiple queries with optimized API usage

## Data Flow

### Document Processing Pipeline

```
1. File Upload (Streamlit)
   ↓
2. File Validation (DocumentPreprocessor)
   ↓
3. OCR Processing (TextractExtractor)
   ├── PDF → PyMuPDF → Images → Textract
   ├── Images → Direct Textract
   └── Fallback → Mock Extraction
   ↓
4. Text Chunking (LeaseVectorStore)
   ├── Sentence-based splitting
   ├── Keyword extraction
   └── TF-IDF vectorization
   ↓
5. RAG Setup (LeaseRAGAssistant)
   ├── Claude API initialization
   ├── Context system prompts
   └── Fallback response system
   ↓
6. Interactive Q&A Ready
```

### Query Processing Pipeline

```
1. User Question Input
   ↓
2. Vector Search (LeaseVectorStore)
   ├── Keyword extraction
   ├── TF-IDF similarity
   ├── Topic boosting
   └── Rank top-k results
   ↓
3. Context Assembly
   ├── Combine relevant chunks
   ├── Source attribution
   └── Context window management
   ↓
4. Claude API Call (or Fallback)
   ├── System prompt + context + question
   ├── Response generation
   └── Error handling
   ↓
5. Response Display
   ├── Main answer
   ├── Source citations
   ├── Confidence score
   └── Chat history update
```

## Error Handling & Resilience

### Multi-Level Fallback System

1. **AWS Service Failures**
   - Primary: Real AWS Textract
   - Fallback: Mock extraction with realistic lease data

2. **PDF Processing Failures**
   - Primary: Direct PDF processing
   - Fallback 1: pdf2image conversion
   - Fallback 2: PyMuPDF conversion
   - Fallback 3: Mock extraction

3. **Claude API Failures**
   - Primary: Anthropic Claude API
   - Fallback: Template-based responses using extracted context

4. **Dependency Issues**
   - Optional imports with graceful degradation
   - Mock implementations for all major components
   - Comprehensive error logging

### Graceful Degradation Features

```python
# Example: Optional import pattern used throughout
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Usage with fallback
if PANDAS_AVAILABLE:
    df = pd.DataFrame(data)
    return df.to_dict()
else:
    return simple_dict_processing(data)
```

## Configuration Management

### Environment Variables (.env)
```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=YourAWSAccessKeyID
AWS_SECRET_ACCESS_KEY=YourAWSSecretAccessKey
AWS_DEFAULT_REGION=YourAWSDefaultRegion

# Anthropic Configuration
ANTHROPIC_API_KEY=YourAnthropicAPIKey

# Application Settings
DEBUG=True
STREAMLIT_PORT=8501
MAX_FILE_SIZE_MB=10

# Vector Store Settings
CHUNK_SIZE=500
CHUNK_OVERLAP=50
SEARCH_K=5
```

### Session State Configuration
- **Persistent Chat History** - Maintained across interactions
- **Processing Status** - Track pipeline completion
- **Demo Mode** - Works without API keys
- **Advanced Features** - Lease analysis tools

## Security Considerations

### API Key Management
- Environment variable configuration
- Graceful fallback when keys unavailable
- No hardcoded credentials

### File Processing Security
- File type validation
- Size limits (configurable)
- Temporary file cleanup
- Safe PDF processing with resource limits

### Data Privacy
- No persistent storage of uploaded documents
- Session-based processing only
- Automatic cleanup on session end

## Deployment Architecture

### Local Development
- **Entry Point**: `streamlit run streamlit_app.py`
- **Dependencies**: Minimal requirements.txt
- **Configuration**: .env file
- **Fallbacks**: Works offline with mock data

### Production Deployment
- **Platform**: Streamlit Cloud (primary)
- **CI/CD**: GitHub Actions workflow
- **Lambda Integration**: AWS Lambda for OCR processing
- **Monitoring**: Built-in health checks and status reporting

### Scalability Features
- **Stateless Design** - Each session independent
- **AWS Integration** - Scalable OCR through Textract
- **Minimal Dependencies** - Fast deployment and startup
- **Caching Support** - Vector store persistence

## Performance Characteristics

### Processing Speed
- **PDF Processing**: 2-10 seconds depending on pages
- **Text Chunking**: Sub-second for typical documents
- **Vector Search**: Sub-second for 100s of chunks
- **Claude Responses**: 2-5 seconds depending on complexity

### Memory Usage
- **Base Application**: ~50MB
- **Document Processing**: +10-50MB per document
- **Vector Storage**: ~1MB per 1000 chunks
- **Session State**: ~5-20MB per active session

### Limitations
- **File Size**: 10MB default limit (configurable)
- **Concurrent Users**: Limited by Streamlit Cloud resources
- **API Rate Limits**: Anthropic and AWS service limits apply
- **Storage**: No persistent document storage

## Extensibility Points

### Adding New OCR Providers
```python
# Follow the pattern in textract_extract.py
class NewOCRExtractor:
    def extract_from_file(self, file_path, preprocess=True):
        # Implement extraction logic
        return standardized_output_format
```

### Adding New Vector Store Backends
```python
# Follow the pattern in vector_store.py
class NewVectorStore:
    def search(self, query: str, k: int = 5):
        # Implement search logic
        return standardized_search_results
```

### Adding New AI Providers
```python
# Follow the pattern in rag_chat.py
class NewAIAssistant:
    def query(self, question: str, k: int = 5):
        # Implement AI integration
        return standardized_response_format
```

This architecture provides a robust, production-ready foundation for lease document analysis with comprehensive error handling, fallback systems, and extensibility for future enhancements.