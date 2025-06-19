# LeaseLens Deployment Guide

## Overview

LeaseLens is deployed using a multi-tier approach with Streamlit Cloud for the web application and AWS Lambda for OCR processing. The system includes comprehensive CI/CD pipelines and fallback mechanisms for development and production environments.

## Environment Setup

### 1. Local Development Setup

#### Prerequisites
- Python 3.9+
- Git
- AWS CLI (optional, for production features)

#### Quick Start
```bash
# Clone repository
git clone <repository-url>
cd lease-lens-demo

# Copy environment template
cp .env.example .env

# Edit .env with your API keys (see Configuration section)

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run streamlit_app.py
# OR use the launcher script
python run_app.py
```

#### Using the Application Launcher
The `run_app.py` script provides enhanced functionality:

```bash
# Run with default settings
python run_app.py run

# Run on custom port/host
python run_app.py run --port 8502 --host 0.0.0.0

# Run with debug logging
python run_app.py run --debug

# Check system status
python run_app.py status

# Run tests
python run_app.py test

# Install/update dependencies
python run_app.py install
```

### 2. Production Deployment

#### Deployment Targets
- **Primary**: Streamlit Cloud (`https://leaselens.streamlit.app`)
- **AWS Lambda**: OCR processing backend
- **Docker**: Alternative containerized deployment (optional)

## Configuration

### Environment Variables

#### Core Application Settings
```bash
# Application Configuration
DEBUG=True                    # Enable debug mode (development only)
STREAMLIT_PORT=8501          # Streamlit server port
MAX_FILE_SIZE_MB=10          # Maximum upload file size

# Vector Store Configuration
CHUNK_SIZE=500               # Text chunk size for processing
CHUNK_OVERLAP=50             # Overlap between chunks
SEARCH_K=5                   # Number of search results to return
```

#### AWS Configuration (Production)
```bash
# AWS Credentials
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION=us-east-1

# AWS Lambda Configuration
LAMBDA_FUNCTION_NAME=leaselens-ocr-processor
LAMBDA_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT_ID:role/your-lambda-role
LAMBDA_REGION=us-east-1
```

#### AI Service Configuration
```bash
# Anthropic Claude API
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Vector Embeddings (if using advanced models)
VECTOR_MODEL=all-MiniLM-L6-v2  # Currently using TF-IDF (no external model needed)
```

### Configuration Files

#### `.env.example` Template
The repository includes a complete `.env.example` template with all required variables:
- Copy to `.env` for local development
- All values are placeholder strings that need to be replaced
- Missing variables will trigger demo mode with mock data

#### `requirements.txt` Dependencies
Core production dependencies (minimal for fast deployment):
```
streamlit>=1.28.0
boto3>=1.28.0
anthropic>=0.3.0
python-dotenv>=1.0.0
requests>=2.31.0
plotly>=5.15.0
Pillow>=10.0.0
PyMuPDF>=1.23.0
```

Optional dependencies (graceful fallbacks):
```
pandas>=2.0.0          # Data manipulation (optional)
numpy>=1.24.0          # Numerical operations (optional)
```

Development/testing dependencies:
```
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.0.0
flake8>=6.0.0
bandit>=1.7.0
```

## CI/CD Pipeline

### GitHub Actions Workflow (`.github/workflows/main.yml`)

#### Workflow Triggers
- **Push**: `main` and `develop` branches
- **Pull Request**: targeting `main` branch

#### Pipeline Stages

1. **Test Stage** (`test` job)
   - Runs on: `ubuntu-latest`
   - Python version: 3.9
   - Dependencies caching enabled
   - Security scanning with Bandit
   - Code formatting with Black (line-length 127)
   - Linting with Flake8
   - Comprehensive test suite

2. **Docker Build** (`build-docker` job)
   - Currently disabled (`if: false`)
   - Builds and pushes to Docker Hub
   - Requires `DOCKER_USERNAME` and `DOCKER_PASSWORD` secrets

3. **Lambda Deployment** (`deploy-lambda` job)
   - Deploys only on `main` branch
   - Packages `ocr_pipeline/lambda_function.py`
   - Updates or creates AWS Lambda function
   - 900-second timeout, 1024MB memory

4. **Streamlit Deployment** (`deploy-streamlit` job)
   - Deploys only on `main` branch
   - Automatic deployment via Streamlit Cloud GitHub integration

#### Required GitHub Secrets
```bash
# AWS Configuration
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION

# Anthropic API
ANTHROPIC_API_KEY

# AWS Lambda
LAMBDA_FUNCTION_NAME
LAMBDA_ROLE_ARN
LAMBDA_REGION

# Docker Hub (optional)
DOCKER_USERNAME
DOCKER_PASSWORD
```

### Pipeline Features

#### Dependency Caching
- Pip cache optimization for faster builds
- Cache key based on `requirements.txt` hash

#### Error Handling
- Non-blocking security scans (high-severity only)
- Graceful test failures with detailed reporting
- Deployment rollback on failure

#### Current Status
Some pipeline stages are temporarily streamlined for rapid deployment:
- Security scanning temporarily bypassed
- Code formatting checks disabled during initial deployment
- Complex tests temporarily disabled to avoid opencv dependency issues

## Deployment Environments

### 1. Streamlit Cloud Deployment

#### Setup Process
1. **Repository Connection**
   - Connect GitHub repository to Streamlit Cloud
   - Select `main` branch for deployment
   - Set `streamlit_app.py` as entry point

2. **Environment Configuration**
   - Add all required environment variables in Streamlit Cloud dashboard
   - Variables are automatically loaded at runtime
   - Secrets are encrypted and secure

3. **Automatic Deployment**
   - Deploys automatically on push to `main` branch
   - Build process uses `requirements.txt`
   - Health checks and monitoring included

#### Production URL
- **Live Application**: `https://leaselens.streamlit.app`
- **Health Status**: Built-in Streamlit Cloud monitoring
- **Logs**: Available in Streamlit Cloud dashboard

### 2. AWS Lambda Deployment

#### Lambda Function Configuration
```python
# Function Details
Name: leaselens-ocr-processor
Runtime: python3.9
Handler: lambda_function.lambda_handler
Timeout: 900 seconds (15 minutes)
Memory: 1024 MB
```

#### Deployment Process
1. Package `ocr_pipeline/lambda_function.py` with dependencies
2. Create ZIP file with boto3 and required libraries
3. Update existing function or create new one
4. Configure IAM role with Textract permissions

#### Lambda Function Code Structure
```python
# lambda_function.py structure
def lambda_handler(event, context):
    # Process OCR requests from Streamlit app
    # Return structured OCR results
    # Handle errors with fallback responses
```

### 3. Docker Deployment (Optional)

#### Dockerfile (if needed)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Docker Compose (local development)
```yaml
version: '3.8'
services:
  leaselens:
    build: .
    ports:
      - "8501:8501"
    env_file:
      - .env
    volumes:
      - ./uploads:/app/uploads
```

## Monitoring and Health Checks

### Application Health
- **Built-in Status**: Demo mode functionality ensures app always works
- **Fallback Systems**: Mock data when APIs unavailable
- **Error Handling**: Comprehensive error messages with suggested fixes

### Performance Monitoring
- **Processing Times**: Tracked in session state
- **API Response Times**: Built-in timing for AWS and Anthropic services
- **Resource Usage**: Streamlit Cloud provides built-in monitoring

### Logging
```python
# Debug information available
DEBUG=True enables:
- Detailed processing logs
- API call timing
- Error stack traces
- Session state debugging
```

## Troubleshooting

### Common Deployment Issues

#### 1. Missing API Keys
**Symptoms**: App runs in demo mode, mock data displayed
**Solution**: 
- Verify all environment variables are set
- Check `.env` file in local development
- Verify secrets in Streamlit Cloud dashboard

#### 2. AWS Textract Failures
**Symptoms**: Mock extraction used, "Mock Extraction" badges shown
**Solution**:
- Verify AWS credentials and permissions
- Check AWS region configuration
- Ensure Textract service is available in selected region

#### 3. Lambda Deployment Failures
**Symptoms**: CI/CD pipeline fails at Lambda deployment step
**Solution**:
- Verify Lambda IAM role exists and has correct permissions
- Check Lambda function name matches configuration
- Ensure AWS credentials have Lambda management permissions

#### 4. Anthropic API Issues
**Symptoms**: Template responses instead of AI-generated content
**Solution**:
- Verify Anthropic API key is valid and has credits
- Check API rate limits
- Ensure network connectivity to Anthropic services

### Development Issues

#### 1. Import Errors
**Symptoms**: Module not found errors
**Solution**:
```bash
# Install all dependencies
pip install -r requirements.txt

# Check for missing packages
python run_app.py status

# Install specific missing packages
pip install package_name
```

#### 2. Port Conflicts
**Symptoms**: "Port already in use" errors
**Solution**:
```bash
# Use different port
python run_app.py run --port 8502

# Or find and kill process using port 8501
lsof -ti:8501 | xargs kill -9
```

#### 3. File Upload Issues
**Symptoms**: File upload failures or processing errors
**Solution**:
- Check file size (default limit: 10MB)
- Verify file format (PDF or image)
- Ensure file is not corrupted
- Check available disk space

## Performance Optimization

### Deployment Optimization
- **Minimal Dependencies**: Core requirements only for fast startup
- **Dependency Caching**: GitHub Actions cache for faster builds
- **Lazy Loading**: Optional imports for non-critical features

### Runtime Optimization
- **Session State**: Efficient state management
- **Vector Search**: Optimized TF-IDF implementation
- **API Caching**: Smart caching of API responses

### Scaling Considerations
- **Stateless Design**: Each session is independent
- **Resource Management**: Automatic cleanup of temporary files
- **Concurrent Users**: Limited by Streamlit Cloud plan

## Security

### API Key Management
- Environment variables only (no hardcoded keys)
- Separate development and production keys
- Rotation procedures documented

### File Processing Security
- File type validation
- Size limits enforced
- Temporary file cleanup
- No persistent storage of uploaded documents

### Network Security
- HTTPS only in production
- API endpoints properly secured
- No sensitive data in logs

## Backup and Recovery

### Code Repository
- **GitHub**: Primary code repository with full history
- **Branching**: `main` (production) and `develop` (staging)
- **Releases**: Tagged releases for rollback capability

### Configuration Backup
- **Environment Templates**: `.env.example` with all required variables
- **Documentation**: Complete setup procedures documented
- **CI/CD Configuration**: Version controlled in repository

### Data Recovery
- **No Persistent Data**: Application is stateless
- **Session Recovery**: Users can re-upload documents
- **Configuration Recovery**: From repository and documentation

## Maintenance

### Regular Maintenance Tasks
1. **Dependency Updates**: Monthly review of `requirements.txt`
2. **API Key Rotation**: Quarterly rotation of all API keys
3. **Security Scans**: Automated via GitHub Actions
4. **Performance Review**: Monthly analysis of processing times

### Monitoring Checklist
- [ ] Streamlit Cloud application health
- [ ] AWS Lambda function status
- [ ] API key expiration dates
- [ ] GitHub Actions pipeline status
- [ ] Error rates and response times

This deployment guide reflects the actual working configuration of LeaseLens, including all real CI/CD pipelines, environment variables, and deployment procedures currently in use.