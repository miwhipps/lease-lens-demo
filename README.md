# LeaseLens 🏠

**AI-Powered Lease Document Analysis**

LeaseLens is a production-ready Streamlit web application that uses AWS Textract for OCR and Anthropic Claude for intelligent lease document analysis. Upload any lease document and get instant answers about rent, policies, terms, and potential risks.

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-lease--lens--demo.onrender.com-blue)](https://lease-lens-demo.onrender.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## 🚀 Quick Start

### Try the Live Demo
**[🌐 lease-lens-demo.onrender.com](https://lease-lens-demo.onrender.com/)** - Works immediately with demo data, no setup required!

### Local Development
```bash
# Clone and setup
git clone <repository-url>
cd lease-lens-demo
cp .env.example .env

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run streamlit_app.py
```

## ✨ Features

### 🔍 **Intelligent Document Processing**
- **AWS Textract OCR** - Professional-grade text extraction from PDFs and images
- **Multi-page Support** - Handle complex lease documents with multiple pages
- **Automatic Fallbacks** - Works even without API keys using realistic demo data

### 🧠 **AI-Powered Analysis**
- **Anthropic Claude Integration** - Advanced natural language understanding
- **Contextual Q&A** - Ask questions in plain English about your lease
- **Smart Search** - TF-IDF vector search finds relevant information instantly

### 📊 **Advanced Lease Analysis**
- **📋 Lease Summary** - Categorized breakdown of financial, legal, and property details
- **⚠️ Risk Analysis** - Automated identification of potential concerns with severity levels
- **💰 Key Figures** - Extract important financial information and dates

### 🎨 **User Experience**
- **Clean Interface** - Intuitive Streamlit web application
- **Dark/Light Mode** - Theme support with accessibility features
- **Mobile Friendly** - Responsive design works on all devices
- **Real-time Processing** - Live status updates during document processing

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   AWS Textract  │    │ Anthropic Claude │
│   Web App       │    │   OCR Service   │    │   AI Assistant   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Core Pipeline                            │
│  ┌───────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────┐ │
│  │   OCR     │→ │ Text Chunking │→ │ Vector Store │→ │   RAG    │ │
│  │ Pipeline  │  │  & Embedding  │  │   (TF-IDF)   │  │ Assistant │ │
│  └───────────┘  └──────────────┘  └─────────────┘  └──────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components
- **OCR Pipeline** (`ocr_pipeline/`) - AWS Textract integration with fallbacks
- **Vector Store** (`embeddings/`) - Custom TF-IDF search engine
- **RAG Assistant** (`ai_assistant/`) - Claude integration with context management
- **Web Interface** (`streamlit_app.py`) - Complete Streamlit application

## 🛠️ Installation

### Requirements
- Python 3.9+
- AWS Account (optional - demo mode works without)
- Anthropic API Key (optional - fallback responses available)

### Dependencies
```bash
# Core production dependencies
streamlit>=1.28.0
boto3>=1.28.0
anthropic>=0.3.0
python-dotenv>=1.0.0
requests>=2.31.0
plotly>=5.15.0
Pillow>=10.0.0
PyMuPDF>=1.23.0

# Optional (graceful fallbacks if missing)
pandas>=2.0.0
numpy>=1.24.0
```

### Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys:
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_DEFAULT_REGION=us-east-1
ANTHROPIC_API_KEY=your_anthropic_key
```

### Advanced Setup Options
```bash
# Use the enhanced launcher
python run_app.py run --port 8502 --debug

# Check system status
python run_app.py status

# Run tests
python run_app.py test
```

## 📱 Usage

### Basic Usage
1. **Upload Document** - Drop PDF or image file in the upload area
2. **Wait for Processing** - OCR and vector indexing happen automatically
3. **Ask Questions** - Type natural language questions about your lease
4. **Get Advanced Analysis** - Use sidebar features for comprehensive analysis

### Sample Questions
- "What is the monthly rent?"
- "What are the pet policies?"
- "Are there any break clauses?"
- "What utilities are included?"
- "Who handles maintenance?"
- "Can I sublet the property?"

### Advanced Features
- **📋 Lease Summary** - Comprehensive categorized analysis
- **⚠️ Risk Analysis** - Potential concerns with severity ratings
- **💰 Key Figures** - Financial metrics and important dates
- **🔍 Document Processing Panel** - Detailed technical information

## 🔧 Configuration

### Environment Variables
```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key  
AWS_DEFAULT_REGION=us-east-1

# Anthropic Configuration
ANTHROPIC_API_KEY=your_anthropic_key

# Application Settings
DEBUG=True
MAX_FILE_SIZE_MB=10
STREAMLIT_PORT=8501

# Vector Store Settings
CHUNK_SIZE=500
CHUNK_OVERLAP=50
SEARCH_K=5
```

### Demo Mode
LeaseLens works perfectly without any API keys:
- **Mock OCR** - Realistic lease text extraction
- **Template Responses** - Contextual answers using extracted text
- **Full Functionality** - All features work with sample data

## 🧪 Testing

### Run Tests
```bash
# All tests
python run_app.py test

# Specific components
python -m pytest tests/test_ocr.py
python -m pytest tests/test_embeddings.py
python -m pytest tests/test_rag.py
```

### Manual Testing
```bash
# Test document processing
python debug_document_processing.py

# Test vector store
python -c "from embeddings.vector_store import test_vector_store; test_vector_store()"
```

## 🚢 Deployment

### Render (Production)
1. Connect repository to Render
2. Set environment variables in dashboard
3. Deploy from `main` branch
4. **Live URL**: [lease-lens-demo.onrender.com](https://lease-lens-demo.onrender.com/)

### Local Development
```bash
streamlit run streamlit_app.py
# OR
python run_app.py run
```

### Docker (Optional)
```bash
# Build image
docker build -t leaselens .

# Run container
docker run -p 8501:8501 --env-file .env leaselens
```

### AWS Lambda (OCR Backend)
- Automated deployment via GitHub Actions
- Function: `leaselens-ocr-processor`
- Runtime: Python 3.9, 1024MB memory, 15min timeout

## 📚 Documentation

- **[📖 ARCHITECTURE.md](./ARCHITECTURE.md)** - Complete system architecture and components
- **[🔧 API_DOCS.md](./API_DOCS.md)** - Detailed API documentation and interfaces  
- **[🚀 DEPLOYMENT.md](./DEPLOYMENT.md)** - Full deployment guide and CI/CD setup

## 🤝 Development

### Project Structure
```
lease-lens-demo/
├── streamlit_app.py          # Main Streamlit application
├── run_app.py               # Enhanced application launcher
├── requirements.txt         # Python dependencies
├── .env.example            # Environment template
├── styles.css              # UI styling and themes
├── ocr_pipeline/           # AWS Textract integration
├── embeddings/             # Vector search engine
├── ai_assistant/           # Claude RAG integration
├── tests/                  # Test suite
├── .github/workflows/      # CI/CD pipeline
└── docs/                   # Documentation
```

### Key Design Principles
- **Graceful Degradation** - Works without API keys
- **Comprehensive Fallbacks** - Multiple backup systems
- **Minimal Dependencies** - Fast deployment and startup
- **Production Ready** - Full CI/CD and monitoring

### Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 🔍 Troubleshooting

### Common Issues

**App shows demo mode / mock data**
- Check that `.env` file exists with correct API keys
- Verify AWS and Anthropic credentials are valid

**File upload fails**
- Check file size (default 10MB limit)
- Ensure file is PDF or image format
- Try different browser if issues persist

**Processing takes too long**
- Large documents may take 30+ seconds
- Multi-page PDFs require more processing time
- Check network connectivity to AWS/Anthropic

**Questions return generic responses**
- Document may not have been processed successfully
- Try re-uploading the document
- Check that text was extracted properly in processing panel

### Getting Help
- 📋 Check processing panel for detailed status
- 🔍 Use debug mode: `python run_app.py run --debug`
- 📊 Run system status: `python run_app.py status`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **AWS Textract** - Professional OCR capabilities
- **Anthropic Claude** - Advanced AI language understanding
- **Streamlit** - Excellent web application framework
- **PyMuPDF** - Reliable PDF processing
- **scikit-learn** - TF-IDF vectorization algorithms

## 📧 Contact

For questions, issues, or contributions:
- 🐛 **Bug Reports**: [GitHub Issues](../../issues)
- 💡 **Feature Requests**: [GitHub Discussions](../../discussions)
- 📖 **Documentation**: See `/docs` folder

---

**Built with ❤️ using Python, Streamlit, AWS, and Anthropic Claude**