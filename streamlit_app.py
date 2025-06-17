import streamlit as st
import tempfile
import os
import time
import json
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import cv2
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our components
try:
    from ocr_pipeline.textract_extract import TextractExtractor
    from ocr_pipeline.preprocess import DocumentPreprocessor
    from embeddings.vector_store import LeaseVectorStore
    from ai_assistant.rag_chat import LeaseRAGAssistant
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ Component import error: {e}")
    st.error("Please ensure all dependencies are installed and you're running from the project root.")
    COMPONENTS_AVAILABLE = False

def clean_source_display_text(text: str) -> str:
    """Clean source text for better display by removing page references and noise"""
    import re
    
    # Remove page references like "page 1 of 10", "Page 1/10", etc.
    text = re.sub(r'\bpage\s+\d+\s+of\s+\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bpage\s+\d+/\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d+\s+of\s+\d+\s+pages?\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bpg\.?\s+\d+\b', '', text, flags=re.IGNORECASE)
    
    # Remove excessive whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Clean up double punctuation left after page removal
    text = re.sub(r'[\.]{2,}', '.', text)  # Multiple periods to single
    text = re.sub(r'\.\s*\.', '.', text)  # Spaced double periods
    text = re.sub(r'\s*\.\s*\.', '.', text)  # Various double period patterns
    
    # Limit length for better display
    if len(text) > 500:
        text = text[:500] + "..."
    
    return text

def main():
    """Main Streamlit application"""
    
    # Configure page
    st.set_page_config(
        page_title="LeaseLens - AI Lease Assistant",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/yourusername/lease-lens-demo',
            'Report a bug': 'https://github.com/yourusername/lease-lens-demo/issues',
            'About': "LeaseLens - AI-Powered Lease Document Assistant"
        }
    )
    
    # Custom CSS styling
    apply_custom_styling()
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    create_header()
    
    # Main layout
    if not COMPONENTS_AVAILABLE:
        show_component_error()
        return
    
    # Sidebar
    create_sidebar()
    
    # Main content area
    col1, col2 = st.columns([2.5, 1.5])
    
    with col1:
        if st.session_state.get('rag_assistant'):
            # Check what to display in main area
            if st.session_state.get('show_lease_summary'):
                display_lease_summary()
                if st.button("💬 Return to Chat", type="secondary"):
                    st.session_state.show_lease_summary = False
                    st.rerun()
            elif st.session_state.get('show_risk_analysis'):
                display_risk_analysis()
                if st.button("💬 Return to Chat", type="secondary"):
                    st.session_state.show_risk_analysis = False
                    st.rerun()
            elif st.session_state.get('show_key_figures'):
                display_key_figures()
                if st.button("💬 Return to Chat", type="secondary"):
                    st.session_state.show_key_figures = False
                    st.rerun()
            else:
                create_chat_interface()
        else:
            show_welcome_screen()
    
    with col2:
        create_analytics_panel()
    
    # Footer
    create_footer()

def apply_custom_styling():
    """Apply custom CSS styling"""
    st.markdown("""
    <style>
    /* Main styling */
    .main-header {
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .feature-card {
        background: white;
        border: 1px solid #e1e5e9;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-success { background-color: #28a745; }
    .status-warning { background-color: #ffc107; }
    .status-danger { background-color: #dc3545; }
    .status-info { background-color: #17a2b8; }
    
    .chat-message {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 10px 10px 0;
    }
    
    .source-citation {
        background: #e9ecef;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        font-size: 0.9em;
    }
    
    .processing-animation {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            padding: 0.5rem;
            font-size: 1.2rem;
        }
        .feature-card {
            margin: 0.5rem 0;
            padding: 1rem;
        }
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    
    # Core application state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = {
            'file_uploaded': False,
            'ocr_completed': False,
            'embeddings_completed': False,
            'rag_completed': False
        }
    
    if 'document_stats' not in st.session_state:
        st.session_state.document_stats = {}
    
    if 'conversation_stats' not in st.session_state:
        st.session_state.conversation_stats = {
            'total_questions': 0,
            'avg_confidence': 0.0,
            'topics_discussed': [],
            'session_start': datetime.now()
        }
    
    if 'demo_mode' not in st.session_state:
        st.session_state.demo_mode = False
    
    if 'advanced_features' not in st.session_state:
        st.session_state.advanced_features = {
            'lease_summary': None,
            'risk_analysis': None,
            'key_figures': None
        }

def create_header():
    """Create the application header"""
    st.markdown("""
    <div class="main-header">
        <h1>🏠 LeaseLens - AI-Powered Lease Assistant</h1>
        <p>Upload your lease document and get instant AI-powered insights with source citations</p>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar():
    """Create the sidebar with upload and processing controls"""
    
    with st.sidebar:
        st.markdown("## 📄 Document Processing")
        
        # Demo mode toggle
        demo_mode = st.toggle(
            "🎮 Demo Mode", 
            value=st.session_state.demo_mode,
            help="Use sample data without uploading a document"
        )
        
        if demo_mode != st.session_state.demo_mode:
            st.session_state.demo_mode = demo_mode
            if demo_mode:
                load_demo_data()
                st.rerun()
        
        if not st.session_state.demo_mode:
            # File upload section
            uploaded_file = st.file_uploader(
                "Upload Lease Document",
                type=['pdf', 'png', 'jpg', 'jpeg'],
                help="Upload a lease document (PDF or image) for AI analysis"
            )
            
            if uploaded_file:
                st.success(f"✅ File uploaded: {uploaded_file.name}")
                st.session_state.processing_status['file_uploaded'] = True
                
                # Processing options
                with st.expander("⚙️ Processing Options", expanded=True):
                    use_preprocessing = st.checkbox(
                        "Apply OpenCV Enhancement", 
                        value=True,
                        help="Improve OCR accuracy with image preprocessing"
                    )
                    
                    chunk_size = st.slider(
                        "Text Chunk Size", 
                        min_value=200, 
                        max_value=800, 
                        value=500,
                        help="Size of text chunks for embeddings"
                    )
                    
                    search_k = st.slider(
                        "Number of Sources", 
                        min_value=3, 
                        max_value=10, 
                        value=5,
                        help="Number of relevant sources to retrieve"
                    )
                
                # Process button
                if st.button(
                    "🚀 Process Document", 
                    type="primary", 
                    use_container_width=True,
                    disabled=st.session_state.processing_status.get('rag_completed', False)
                ):
                    process_document(uploaded_file, use_preprocessing, chunk_size, search_k)
        
        # System status
        st.markdown("---")
        create_system_status()
        
        # Sample queries (if document is processed)
        if st.session_state.get('rag_assistant'):
            st.markdown("---")
            create_sample_queries()
        
        # Advanced features
        if st.session_state.get('rag_assistant'):
            st.markdown("---")
            create_advanced_features_sidebar()

def create_system_status():
    """Create system status indicators"""
    st.markdown("### 🔧 System Status")
    
    status_items = [
        ("File Upload", st.session_state.processing_status.get('file_uploaded', False)),
        ("OCR Processing", st.session_state.processing_status.get('ocr_completed', False)),
        ("Vector Embeddings", st.session_state.processing_status.get('embeddings_completed', False)),
        ("RAG Assistant", st.session_state.processing_status.get('rag_completed', False))
    ]
    
    for item, completed in status_items:
        status_class = "status-success" if completed else "status-danger"
        icon = "✅" if completed else "⏳"
        st.markdown(f'<span class="status-indicator {status_class}"></span>{icon} {item}', 
                   unsafe_allow_html=True)

def create_sample_queries():
    """Create sample query buttons"""
    st.markdown("### 💡 Try These Questions")
    
    sample_queries = [
        "What is the monthly rent?",
        "What are the pet policies?", 
        "Are there any break clauses?",
        "What is the security deposit?",
        "What utilities are included?",
        "What are the parking arrangements?",
        "Who handles maintenance?",
        "Can I sublet the property?"
    ]
    
    # Create columns for better layout
    for i in range(0, len(sample_queries), 2):
        col1, col2 = st.columns(2)
        
        with col1:
            if i < len(sample_queries):
                if st.button(
                    f"❓ {sample_queries[i]}", 
                    key=f"sample_{i}",
                    use_container_width=True,
                    help="Click to ask this question"
                ):
                    handle_sample_query(sample_queries[i])
        
        with col2:
            if i + 1 < len(sample_queries):
                if st.button(
                    f"❓ {sample_queries[i + 1]}", 
                    key=f"sample_{i + 1}",
                    use_container_width=True,
                    help="Click to ask this question"
                ):
                    handle_sample_query(sample_queries[i + 1])

def create_advanced_features_sidebar():
    """Create advanced features in sidebar"""
    st.markdown("### 🔬 Advanced Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📋 Lease Summary", use_container_width=True):
            generate_lease_summary()
    
    with col2:
        if st.button("⚠️ Risk Analysis", use_container_width=True):
            generate_risk_analysis()
    
    if st.button("💰 Key Figures", use_container_width=True):
        extract_key_figures()

def process_document(uploaded_file, use_preprocessing, chunk_size, search_k):
    """Process the uploaded document with progress tracking"""
    
    # Create progress tracking
    progress_container = st.container()
    
    with progress_container:
        # Progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        time_display = st.empty()
        
        start_time = time.time()
        
        try:
            # Step 1: Save uploaded file
            status_text.info("📁 Saving uploaded file...")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            progress_bar.progress(10)
            elapsed = time.time() - start_time
            time_display.text(f"⏱️ Elapsed: {elapsed:.1f}s")
            
            # Step 2: Initialize OCR components
            status_text.info("🔧 Initializing OCR components...")
            
            # Try to use real Textract, fall back to mock if needed
            try:
                extractor = TextractExtractor()
                ocr_type = "AWS Textract"
            except Exception as e:
                # Mock extractor for demo
                extractor = create_mock_extractor()
                ocr_type = "Mock OCR (Demo)"
                st.warning(f"⚠️ Using mock OCR: {str(e)[:100]}...")
            
            progress_bar.progress(25)
            elapsed = time.time() - start_time
            time_display.text(f"⏱️ Elapsed: {elapsed:.1f}s")
            
            # Step 3: Extract text
            status_text.info(f"🔍 Extracting text with {ocr_type}...")
            
            if hasattr(extractor, 'extract_from_file'):
                extraction_result = extractor.extract_from_file(tmp_path, preprocess=use_preprocessing)
            else:
                # Mock extraction
                extraction_result = extractor.extract_text()
            
            st.session_state.extraction_result = extraction_result
            st.session_state.processing_status['ocr_completed'] = True
            
            # Store debug info in session state for persistent display
            extracted_text = extraction_result.get('text', '')
            confidence = extraction_result.get('confidence', 0)
            
            # Check for rent keywords
            rent_keywords = ['rent', 'monthly', '$', '£', 'payment']
            found_keywords = [kw for kw in rent_keywords if kw.lower() in extracted_text.lower()]
            
            st.session_state.debug_info = {
                'ocr_engine': ocr_type,
                'confidence': confidence,
                'characters_extracted': len(extracted_text),
                'words_extracted': len(extracted_text.split()),
                'pages_processed': extraction_result.get('page_count', 1),
                'multi_page': extraction_result.get('multi_page_extraction', False),
                'rent_keywords_found': found_keywords,
                'extracted_text_preview': extracted_text[:300] + "..." if len(extracted_text) > 300 else extracted_text
            }
            
            progress_bar.progress(50)
            elapsed = time.time() - start_time
            time_display.text(f"⏱️ Elapsed: {elapsed:.1f}s")
            
            # Step 4: Create embeddings
            status_text.info("🧠 Creating vector embeddings...")
            
            vector_store = LeaseVectorStore()
            vector_store.add_document(
                extraction_result['text'],
                doc_id="uploaded_lease",
                source_info={
                    "filename": uploaded_file.name,
                    "upload_time": datetime.now().isoformat(),
                    "preprocessing_used": use_preprocessing
                }
            )
            
            st.session_state.vector_store = vector_store
            st.session_state.processing_status['embeddings_completed'] = True
            
            # Store vector store debug info in session state
            chunk_debug_info = []
            for i, chunk in enumerate(vector_store.chunks):
                words = len(chunk['text'].split())
                has_rent = any(word in chunk['text'].lower() for word in ['rent', 'monthly'])
                has_money = any(symbol in chunk['text'] for symbol in ['$', '£', '€', '¥', '£']) or any(word in chunk['text'].lower() for word in ['dollar', 'pound', 'euro', 'usd', 'gbp'])
                
                chunk_debug_info.append({
                    'chunk_id': i,
                    'words': words,
                    'has_rent': has_rent,
                    'has_money': has_money,
                    'preview': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
                })
            
            st.session_state.chunk_debug_info = {
                'total_chunks': len(vector_store.chunks),
                'chunks': chunk_debug_info
            }
            
            progress_bar.progress(75)
            elapsed = time.time() - start_time
            time_display.text(f"⏱️ Elapsed: {elapsed:.1f}s")
            
            # Step 5: Initialize RAG assistant
            status_text.info("🤖 Setting up AI assistant...")
            
            rag_assistant = LeaseRAGAssistant(vector_store)
            st.session_state.rag_assistant = rag_assistant
            st.session_state.processing_status['rag_completed'] = True
            
            progress_bar.progress(100)
            elapsed = time.time() - start_time
            time_display.text(f"⏱️ Total time: {elapsed:.1f}s")
            
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            
            # Update document stats
            st.session_state.document_stats = {
                'filename': uploaded_file.name,
                'file_size': len(uploaded_file.getvalue()),
                'character_count': len(extraction_result['text']),
                'confidence': extraction_result.get('confidence', 0),
                'line_count': extraction_result.get('line_count', 0),
                'chunk_count': len(vector_store.texts),
                'preprocessing_used': use_preprocessing,
                'processing_time': elapsed,
                'ocr_type': ocr_type
            }
            
            # Add welcome message
            welcome_message = generate_welcome_message(uploaded_file.name, extraction_result)
            st.session_state.messages = [{"role": "assistant", "content": welcome_message}]
            
            # Show success message
            status_text.success("✅ Document processed successfully!")
            st.success("🎉 Document processed successfully! You can now ask questions.")
            
            # Auto-rerun to show the chat interface
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error processing document: {str(e)}")
            progress_bar.progress(0)
            status_text.error("❌ Processing failed")
            
            # Clean up on error
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)

def create_mock_extractor():
    """Create a mock extractor for demo purposes"""
    class MockExtractor:
        def extract_from_file(self, file_path, preprocess=True):
            return {
                'text': """RESIDENTIAL LEASE AGREEMENT
                
Property Address: 123 Demo Street, Apt 5B, San Francisco, CA 94105

LEASE TERMS:
Monthly Rent: $4,200.00 per month
Security Deposit: $8,400.00 (equivalent to 2 months rent)
Lease Duration: 12 months
Start Date: January 1, 2024
End Date: December 31, 2024

PET POLICY:
Pets are allowed with written approval from landlord.
Pet deposit: $750.00 per pet (maximum 2 pets)
Monthly pet rent: $125.00 per pet
Restricted breeds: Pit bulls, Rottweilers, Dobermans

UTILITIES:
Tenant Responsible For: Electricity, Gas, Internet, Cable TV
Landlord Responsible For: Water, Sewer, Trash, Recycling, Landscaping
Estimated monthly utilities: $200-250

PARKING:
One assigned covered parking space included
Additional spaces available: $275/month
Guest parking: 3-hour limit in designated visitor areas

TERMINATION CLAUSES:
60 days written notice required for lease termination
Early termination penalty: 2.5 months rent
Military clause: Early termination allowed with PCS orders

MAINTENANCE:
Landlord handles: Major repairs, HVAC, plumbing, electrical, appliances
Tenant handles: Minor repairs under $100, light bulbs, air filters
Emergency maintenance available 24/7

ADDITIONAL FEES:
Application fee: $200 (non-refundable)
Move-in fee: $300 (one-time)
Late rent fee: $100 (after 5th day of month)
Key replacement: $75 per key""",
                'confidence': 94.5,
                'line_count': 35,
                'character_count': 1247
            }
        
        def extract_text(self):
            return self.extract_from_file("", False)
    
    return MockExtractor()

def generate_welcome_message(filename, extraction_result):
    """Generate a personalized welcome message"""
    char_count = len(extraction_result['text'])
    confidence = extraction_result.get('confidence', 0)
    
    return f"""Hello! I've successfully analyzed your lease document '{filename}'. 

📊 **Processing Summary:**
- Extracted {char_count:,} characters
- OCR confidence: {confidence:.1f}%
- Ready to answer your questions!

I can help you understand:
- Financial terms (rent, deposits, fees)
- Policies (pets, guests, parking)
- Rights and responsibilities
- Termination clauses
- And much more!

What would you like to know about your lease?"""

def load_demo_data():
    """Load demo data for demonstration"""
    st.info("🎮 Loading demo data...")
    
    # Create mock components
    vector_store = LeaseVectorStore()
    
    demo_lease_text = """RESIDENTIAL TENANCY AGREEMENT

Property: 456 Demo Avenue, Test City, CA 90210
Tenant: Demo User
Landlord: Demo Properties LLC

The monthly rent for this property is £1,500 per month.
Security deposit required is £3,000, equivalent to two months rent.

Pet Policy: Pets are allowed with written permission from landlord.
Pet deposit of £500 per pet is required.
Maximum of 2 pets allowed per unit.

Utilities: Tenant is responsible for electricity and gas.
Landlord pays for water, sewer, and rubbish collection.

Parking: One assigned parking space is included in rent.
Additional parking spaces available for £200 per month.

Termination: Either party may terminate lease with 60 days written notice.
Early termination penalty is two months rent.

Additional Terms:
Late payment fee: £75 after 5th day of month
Key replacement fee: £50 per key
Professional cleaning required at end of tenancy: £200"""
    
    vector_store.add_document(demo_lease_text, "demo_lease", {"filename": "demo_lease.pdf"})
    
    rag_assistant = LeaseRAGAssistant(vector_store)
    
    # Store in session state
    st.session_state.rag_assistant = rag_assistant
    st.session_state.vector_store = vector_store
    st.session_state.processing_status = {
        'file_uploaded': True,
        'ocr_completed': True,
        'embeddings_completed': True,
        'rag_completed': True
    }
    st.session_state.document_stats = {
        'filename': 'demo_lease.pdf',
        'character_count': len(demo_lease_text),
        'confidence': 95.0,
        'chunk_count': len(vector_store.texts),
        'ocr_type': 'Demo Mode'
    }
    
    # Add welcome message
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Welcome to the LeaseLens demo! I've loaded a sample lease document. Try asking questions like 'What is the monthly rent?' or 'Are pets allowed?'"
    }]

def display_debug_info():
    """Display persistent debug information"""
    if 'debug_info' in st.session_state or 'chunk_debug_info' in st.session_state:
        with st.expander("🔍 Document Processing Debug Info", expanded=False):
            if 'debug_info' in st.session_state:
                debug = st.session_state.debug_info
                st.write("### OCR Extraction Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("OCR Engine", debug['ocr_engine'])
                    st.metric("Confidence", f"{debug['confidence']}%")
                    st.metric("Characters", debug['characters_extracted'])
                
                with col2:
                    st.metric("Words", debug['words_extracted'])
                    st.metric("Pages Processed", debug.get('pages_processed', 1))
                    if debug.get('multi_page', False):
                        st.success("✅ Multi-page processing")
                    else:
                        st.info("ℹ️ Single page processing")
                    
                st.write("**Rent Keywords Found:**")
                if debug['rent_keywords_found']:
                    st.success(f"✅ {debug['rent_keywords_found']}")
                else:
                    st.error("❌ No rent keywords found!")
                
                st.write("**Extracted Text Preview:**")
                st.code(debug['extracted_text_preview'])
            
            if 'chunk_debug_info' in st.session_state:
                chunk_debug = st.session_state.chunk_debug_info
                st.write("### Vector Store Results")
                st.metric("Total Chunks", chunk_debug['total_chunks'])
                
                for chunk_info in chunk_debug['chunks']:
                    with st.container():
                        st.write(f"**Chunk {chunk_info['chunk_id']}:**")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Words", chunk_info['words'])
                        with col2:
                            status = "✅" if chunk_info['has_rent'] else "❌"
                            st.write(f"Rent terms: {status}")
                        with col3:
                            status = "✅" if chunk_info['has_money'] else "❌"
                            st.write(f"Money symbols: {status}")
                        
                        st.code(chunk_info['preview'])

def create_chat_interface():
    """Create the main chat interface"""
    st.markdown("## 💬 Chat with Your Lease")
    
    # Display debug info if available
    display_debug_info()
    
    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                if message["sources"]:
                    with st.expander("📚 View Sources", expanded=False):
                        for j, source in enumerate(message["sources"][:3]):
                            # Clean source text for display
                            clean_text = clean_source_display_text(source['text'])
                            doc_info = f" from {source.get('doc_id', 'document')}" if source.get('doc_id') else ""
                            st.markdown(f"**Source {j+1}**{doc_info} (Relevance: {source['score']:.1%})")
                            st.markdown(f"```\n{clean_text}\n```")
            
            # Show confidence score
            if message["role"] == "assistant" and "confidence" in message:
                confidence = message["confidence"]
                if confidence > 0:
                    color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                    st.markdown(f"**Confidence:** <span style='color: {color}'>{confidence:.1%}</span>", 
                               unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask about your lease...", key="main_chat_input"):
        handle_user_input(prompt)

def handle_user_input(prompt):
    """Handle user chat input"""
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Update conversation stats
    st.session_state.conversation_stats['total_questions'] += 1
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyzing your question..."):
            response = st.session_state.rag_assistant.query(prompt)
            
            # Display answer
            st.markdown(response['answer'])
            
            # Update conversation stats
            if response['confidence'] > 0:
                current_avg = st.session_state.conversation_stats['avg_confidence']
                total_questions = st.session_state.conversation_stats['total_questions']
                new_avg = ((current_avg * (total_questions - 1)) + response['confidence']) / total_questions
                st.session_state.conversation_stats['avg_confidence'] = new_avg
            
            # Store response with sources
            st.session_state.messages.append({
                "role": "assistant",
                "content": response['answer'],
                "sources": response['sources'],
                "confidence": response['confidence']
            })

def handle_sample_query(query):
    """Handle sample query button clicks"""
    if st.session_state.get('rag_assistant'):
        # Add to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Get response
        response = st.session_state.rag_assistant.query(query)
        st.session_state.messages.append({
            "role": "assistant",
            "content": response['answer'],
            "sources": response['sources'],
            "confidence": response['confidence']
        })
        
        # Update stats
        st.session_state.conversation_stats['total_questions'] += 1
        
        st.rerun()

def show_welcome_screen():
    """Show welcome screen when no document is processed"""
    st.markdown("## 👋 Welcome to LeaseLens!")
    
    # Feature overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🔍 Smart OCR</h4>
            <p>Advanced document processing with AWS Textract and OpenCV preprocessing for maximum accuracy.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>🧠 AI Understanding</h4>
            <p>Semantic search and natural language processing to understand your lease terms.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>📚 Source Citations</h4>
            <p>Every answer includes exact source references to prevent AI hallucination.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # How it works
    st.markdown("### 🚀 How LeaseLens Works")
    
    steps_col1, steps_col2 = st.columns(2)
    
    with steps_col1:
        st.markdown("""
        **1. Upload Your Document**
        - PDF or image formats supported
        - Automatic preprocessing for best results
        
        **2. AI Processing**
        - OCR extracts text with high accuracy
        - Creates semantic embeddings for search
        """)
    
    with steps_col2:
        st.markdown("""
        **3. Ask Questions**
        - Natural language queries
        - Get instant, accurate answers
        
        **4. Verify Sources**
        - See exact lease text supporting each answer
        - Confidence scores for transparency
        """)
    
    # Demo section
    st.markdown("---")
    st.markdown("### 🎬 Try LeaseLens Now")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🎮 Start Interactive Demo", type="primary", use_container_width=True):
            st.session_state.demo_mode = True
            load_demo_data()
            st.rerun()

def show_component_error():
    """Show error message when components are not available"""
    st.error("🚨 Component Import Error")
    st.markdown("""
    **LeaseLens components could not be loaded.** This usually happens when:
    
    1. **Dependencies are missing** - Run: `pip install -r requirements.txt`
    2. **Wrong directory** - Make sure you're in the project root
    3. **Python path issues** - Check your PYTHONPATH
    
    **Quick fix:**
    ```bash
    # Install dependencies
    pip install -r requirements.txt
    
    # Run from project root
    streamlit run streamlit_app.py
    ```
    """)

def create_analytics_panel():
    """Create the analytics and statistics panel"""
    st.markdown("## 📊 Document Analytics")
    
    if st.session_state.get('document_stats'):
        stats = st.session_state.document_stats
        
        # Document overview
        st.markdown("### 📄 Document Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Document", stats.get('filename', 'Unknown'))
            st.metric("OCR Confidence", f"{stats.get('confidence', 0):.1f}%")
            st.metric("Processing Time", f"{stats.get('processing_time', 0):.1f}s")
        
        with col2:
            st.metric("Characters", f"{stats.get('character_count', 0):,}")
            st.metric("Text Chunks", stats.get('chunk_count', 0))
            st.metric("OCR Engine", stats.get('ocr_type', 'Unknown'))
        
        # Confidence visualization
        if stats.get('confidence'):
            st.markdown("### 📈 Processing Quality")
            
            confidence = stats['confidence']
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "OCR Confidence %"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        # Conversation statistics
        if st.session_state.messages and len(st.session_state.messages) > 1:
            st.markdown("### 💬 Conversation Stats")
            
            conv_stats = st.session_state.conversation_stats
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Questions Asked", conv_stats['total_questions'])
            
            with col2:
                avg_conf = conv_stats.get('avg_confidence', 0)
                st.metric("Avg Confidence", f"{avg_conf:.1%}")
            
            # Session duration
            if 'session_start' in conv_stats:
                duration = datetime.now() - conv_stats['session_start']
                minutes = int(duration.total_seconds() / 60)
                st.metric("Session Duration", f"{minutes} min")
        
        # Advanced features results
        if st.session_state.advanced_features.get('lease_summary'):
            st.markdown("### 📋 Latest Analysis")
            
            # Safe length checks with proper defaults
            lease_summary = st.session_state.advanced_features.get('lease_summary')
            risk_analysis = st.session_state.advanced_features.get('risk_analysis')
            
            summary_count = len(lease_summary) if lease_summary else 0
            risk_count = len(risk_analysis) if risk_analysis else 0
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Summary Sections", summary_count)
            with col2:
                st.metric("Risks Identified", risk_count)
    
    else:
        # No document processed
        st.info("📋 Upload and process a document to see analytics.")
        
        # Show system capabilities
        st.markdown("### 🔧 System Capabilities")
        
        capabilities = {
            "OCR Engine": "AWS Textract + OpenCV",
            "Embeddings": "Sentence Transformers",
            "Vector DB": "FAISS",
            "LLM": "OpenAI GPT-3.5-Turbo",
            "Framework": "Streamlit + Python"
        }
        
        for capability, technology in capabilities.items():
            st.text(f"{capability}: {technology}")

def generate_lease_summary():
    """Generate lease summary and store for display"""
    if not st.session_state.get('rag_assistant'):
        st.error("❌ Please process a document first")
        return
    
    with st.spinner("📋 Generating comprehensive lease summary..."):
        try:
            summary = st.session_state.rag_assistant.get_lease_summary()
            st.session_state.advanced_features['lease_summary'] = summary
            st.session_state.show_lease_summary = True
            st.success("✅ Lease summary generated! Check the main area.")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Failed to generate summary: {e}")

def display_lease_summary():
    """Display the generated lease summary in main area"""
    summary = st.session_state.advanced_features.get('lease_summary')
    if not summary:
        return
    
    # Display in main area
    st.markdown("## 📋 Comprehensive Lease Summary")
    
    # Organize summary by categories
    financial_terms = {}
    policies = {}
    responsibilities = {}
    other_terms = {}
    
    for key, value in summary.items():
        key_lower = key.lower()
        if any(word in key_lower for word in ['rent', 'deposit', 'fee', 'cost', 'payment']):
            financial_terms[key] = value
        elif any(word in key_lower for word in ['pet', 'guest', 'noise', 'smoking', 'policies', 'sublet']):
            policies[key] = value
        elif any(word in key_lower for word in ['maintenance', 'repair', 'utility', 'utilities', 'responsibilities']):
            responsibilities[key] = value
        elif any(word in key_lower for word in ['parking', 'termination', 'lease term', 'duration']):
            other_terms[key] = value
        else:
            other_terms[key] = value
    
    # Display categorized summary
    col1, col2 = st.columns(2)
    
    with col1:
        if financial_terms:
            st.markdown("### 💰 Financial Terms")
            for key, value in financial_terms.items():
                st.markdown(f"**{key}**")
                st.write(value[:200] + "..." if len(value) > 200 else value)
                st.markdown("---")
        
        if responsibilities:
            st.markdown("### 🔧 Responsibilities")
            for key, value in responsibilities.items():
                st.markdown(f"**{key}**")
                st.write(value[:200] + "..." if len(value) > 200 else value)
                st.markdown("---")
    
    with col2:
        if policies:
            st.markdown("### 📋 Policies")
            for key, value in policies.items():
                st.markdown(f"**{key}**")
                st.write(value[:200] + "..." if len(value) > 200 else value)
                st.markdown("---")
        
        if other_terms:
            st.markdown("### 📄 Other Terms")
            for key, value in other_terms.items():
                st.markdown(f"**{key}**")
                st.write(value[:200] + "..." if len(value) > 200 else value)
                st.markdown("---")
    
    st.success("✅ Lease summary displayed!")

def generate_risk_analysis():
    """Generate risk analysis and store for display"""
    if not st.session_state.get('rag_assistant'):
        st.error("❌ Please process a document first")
        return
    
    with st.spinner("⚠️ Analyzing lease risks and red flags..."):
        try:
            risks = st.session_state.rag_assistant.analyze_lease_risks()
            st.session_state.advanced_features['risk_analysis'] = risks
            st.session_state.show_risk_analysis = True
            st.success("✅ Risk analysis generated! Check the main area.")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Failed to analyze risks: {e}")

def display_risk_analysis():
    """Display the generated risk analysis in main area"""
    risks = st.session_state.advanced_features.get('risk_analysis')
    if not risks:
        return
    
    # Display risk analysis
    st.markdown("## ⚠️ Lease Risk Analysis")
    
    if not risks:
        st.info("✅ No significant risks identified in this lease.")
        return
    
    # Categorize risks by level
    high_risks = [r for r in risks if r.get('risk_level') == 'HIGH']
    medium_risks = [r for r in risks if r.get('risk_level') == 'MEDIUM']
    low_risks = [r for r in risks if r.get('risk_level') == 'LOW']
    
    # Risk overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🔴 High Risk", len(high_risks))
    with col2:
        st.metric("🟡 Medium Risk", len(medium_risks))
    with col3:
        st.metric("🟢 Low Risk", len(low_risks))
    
    # Display risks by category
    for risk_level, risk_list, color in [
        ("HIGH", high_risks, "🔴"),
        ("MEDIUM", medium_risks, "🟡"),
        ("LOW", low_risks, "🟢")
    ]:
        if risk_list:
            st.markdown(f"### {color} {risk_level} Risk Items")
            
            for risk in risk_list:
                with st.expander(f"{risk['category']} (Confidence: {risk['confidence']:.1%})"):
                    st.write(risk['description'])
                    
                    if risk.get('sources'):
                        st.markdown("**Sources:**")
                        for i, source in enumerate(risk['sources'][:2], 1):
                            clean_text = clean_source_display_text(source['text'])[:100] + "..."
                            st.markdown(f"Source {i}: `{clean_text}`")
    
    st.success("✅ Risk analysis completed!")

def extract_key_figures():
    """Extract key figures and store for display"""
    if not st.session_state.get('rag_assistant'):
        st.error("❌ Please process a document first")
        return
    
    with st.spinner("💰 Extracting key financial figures..."):
        try:
            figures = st.session_state.rag_assistant.extract_key_figures()
            st.session_state.advanced_features['key_figures'] = figures
            st.session_state.show_key_figures = True
            st.success("✅ Key figures extracted! Check the main area.")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Failed to extract key figures: {e}")

def display_key_figures():
    """Display the extracted key figures in main area"""
    figures = st.session_state.advanced_features.get('key_figures')
    if not figures:
        return
    
    # Display key figures
    st.markdown("## 💰 Key Financial Figures")
    
    if figures.get('financial'):
        st.markdown("### 💵 Financial Breakdown")
        
        # Create a summary table
        financial_data = []
        for category, data in figures['financial'].items():
            amounts = data.get('amounts', [])
            confidence = data.get('confidence', 0)
            
            financial_data.append({
                'Category': category.title(),
                'Amount': amounts[0] if amounts else 'Not specified',
                'Confidence': f"{confidence:.1%}",
                'Additional Amounts': ', '.join(amounts[1:]) if len(amounts) > 1 else 'None'
            })
        
        if financial_data:
            df = pd.DataFrame(financial_data)
            st.dataframe(df, use_container_width=True)
    
    # Cost estimates
    if figures.get('summary'):
        st.markdown("### 📊 Cost Estimates")
        
        col1, col2 = st.columns(2)
        
        with col1:
            monthly_cost = figures['summary'].get('estimated_monthly_cost', 'Unable to calculate')
            st.metric("Estimated Monthly Cost", monthly_cost)
        
        with col2:
            move_in_cost = figures['summary'].get('move_in_cost', 'Unable to calculate')
            st.metric("Estimated Move-in Cost", move_in_cost)
    
    # Date information
    if figures.get('dates'):
        st.markdown("### 📅 Important Dates")
        
        for category, data in figures['dates'].items():
            st.markdown(f"**{category.title()}:** {data['raw_answer'][:100]}...")
    
    st.success("✅ Key figures extracted successfully!")

def create_footer():
    """Create application footer"""
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🏠 LeaseLens**  
        AI-Powered Lease Assistant
        """)
    
    with col2:
        st.markdown("""
        **🔗 Links**  
        [GitHub Repository](https://github.com/yourusername/lease-lens-demo)  
        [Documentation](https://github.com/yourusername/lease-lens-demo/wiki)
        """)
    
    with col3:
        st.markdown(f"""
        **📊 Session Info**  
        Started: {st.session_state.conversation_stats.get('session_start', datetime.now()).strftime('%H:%M')}  
        Questions: {st.session_state.conversation_stats.get('total_questions', 0)}
        """)

# Additional utility functions for enhanced features

def export_conversation():
    """Export conversation history"""
    if not st.session_state.messages:
        st.warning("No conversation to export")
        return
    
    # Prepare conversation data
    conversation_data = {
        'document': st.session_state.document_stats.get('filename', 'Unknown'),
        'timestamp': datetime.now().isoformat(),
        'messages': st.session_state.messages,
        'stats': st.session_state.conversation_stats
    }
    
    # Convert to JSON
    json_data = json.dumps(conversation_data, indent=2, ensure_ascii=False)
    
    # Provide download
    st.download_button(
        label="📥 Download Conversation",
        data=json_data,
        file_name=f"leaselens_conversation_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json"
    )

if __name__ == "__main__":
    main()
