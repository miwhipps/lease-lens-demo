import streamlit as st
import tempfile
import os
import time
import json
from datetime import datetime
from dotenv import load_dotenv

# Optional imports with graceful fallbacks
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    cv2 = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

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
    text = re.sub(r"\bpage\s+\d+\s+of\s+\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bpage\s+\d+/\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d+\s+of\s+\d+\s+pages?\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bpg\.?\s+\d+\b", "", text, flags=re.IGNORECASE)

    # Remove excessive whitespace and normalize
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    # Clean up double punctuation left after page removal
    text = re.sub(r"[\.]{2,}", ".", text)  # Multiple periods to single
    text = re.sub(r"\.\s*\.", ".", text)  # Spaced double periods
    text = re.sub(r"\s*\.\s*\.", ".", text)  # Various double period patterns

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
            "Get Help": "https://github.com/yourusername/lease-lens-demo",
            "Report a bug": "https://github.com/yourusername/lease-lens-demo/issues",
            "About": "LeaseLens - AI-Powered Lease Document Assistant",
        },
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
        if st.session_state.get("rag_assistant"):
            # Check what to display in main area
            if st.session_state.get("show_lease_summary"):
                display_lease_summary()
                if st.button("💬 Return to Chat", type="secondary"):
                    st.session_state.show_lease_summary = False
                    st.rerun()
            elif st.session_state.get("show_risk_analysis"):
                display_risk_analysis()
                if st.button("💬 Return to Chat", type="secondary"):
                    st.session_state.show_risk_analysis = False
                    st.rerun()
            elif st.session_state.get("show_key_figures"):
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
    """Load external CSS stylesheet with theme-aware styling"""
    # Load the external CSS file
    css_path = os.path.join(os.path.dirname(__file__), 'styles.css')
    try:
        with open(css_path, 'r') as f:
            css_content = f.read()
        
        st.markdown(f"""
        <style>
        {css_content}
        
        /* Hide Streamlit default elements */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        .stDeployButton {{visibility: hidden;}}
        </style>
        """, unsafe_allow_html=True)
        
    except FileNotFoundError:
        st.warning("⚠️ External CSS file not found, using minimal styling")
        st.markdown("""
        <style>
        /* Minimal fallback styling */
        .feature-card {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(225, 229, 233, 0.8);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables"""

    # Core application state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "processing_status" not in st.session_state:
        st.session_state.processing_status = {
            "file_uploaded": False,
            "ocr_completed": False,
            "embeddings_completed": False,
            "rag_completed": False,
        }

    if "document_stats" not in st.session_state:
        st.session_state.document_stats = {}

    if "conversation_stats" not in st.session_state:
        st.session_state.conversation_stats = {
            "total_questions": 0,
            "avg_confidence": 0.0,
            "topics_discussed": [],
            "session_start": datetime.now(),
        }

    if "demo_mode" not in st.session_state:
        st.session_state.demo_mode = False

    if "advanced_features" not in st.session_state:
        st.session_state.advanced_features = {"lease_summary": None, "risk_analysis": None, "key_figures": None}


def create_header():
    """Create the application header"""
    st.markdown(
        """
    <div class="main-header">
        <h1>🏠 LeaseLens - AI-Powered Lease Assistant</h1>
        <p>Upload your lease document and get instant AI-powered insights with source citations</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def create_sidebar():
    """Create the sidebar with upload and processing controls"""

    with st.sidebar:
        st.markdown("## 📄 Document Processing")

        # Demo mode toggle
        demo_mode = st.toggle(
            "🎮 Demo Mode", value=st.session_state.demo_mode, help="Use sample data without uploading a document"
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
                type=["pdf", "png", "jpg", "jpeg"],
                help="Upload a lease document (PDF or image) for AI analysis",
            )

            if uploaded_file:
                st.success(f"✅ File uploaded: {uploaded_file.name}")
                st.session_state.processing_status["file_uploaded"] = True
                
                # 🔍 DEBUGGING: Comprehensive file info
                with st.expander("🔍 Debug: File Upload Details", expanded=False):
                    st.write("**📁 File Information:**")
                    st.write(f"• Name: `{uploaded_file.name}`")
                    st.write(f"• Size: `{uploaded_file.size:,} bytes` ({uploaded_file.size / 1024:.1f} KB)")
                    st.write(f"• Type: `{uploaded_file.type}`")
                    
                    # Show file preview if possible
                    if uploaded_file.type.startswith('image/') and PIL_AVAILABLE and Image is not None:
                        st.write("**🖼️ Image Preview:**")
                        try:
                            image = Image.open(uploaded_file)
                            st.image(image, caption=f'Preview: {uploaded_file.name}', use_column_width=True)
                            st.write(f"• Image dimensions: {image.size[0]} x {image.size[1]} pixels")
                            st.write(f"• Image mode: {image.mode}")
                        except Exception as e:
                            st.error(f"Could not display image preview: {e}")
                    elif uploaded_file.type == 'application/pdf':
                        st.write("**📄 PDF File Detected**")
                        st.info("PDF preview not available - will be processed by OCR engine")
                    else:
                        st.warning(f"Unknown file type: {uploaded_file.type}")
                    
                    # Reset file pointer after preview
                    uploaded_file.seek(0)

                # Processing options
                with st.expander("⚙️ Processing Options", expanded=True):
                    use_preprocessing = st.checkbox(
                        "Apply OpenCV Enhancement", value=True, help="Improve OCR accuracy with image preprocessing"
                    )

                    chunk_size = st.slider(
                        "Text Chunk Size", min_value=200, max_value=800, value=500, help="Size of text chunks for embeddings"
                    )

                    search_k = st.slider(
                        "Number of Sources", min_value=3, max_value=10, value=5, help="Number of relevant sources to retrieve"
                    )

                # Process button
                if st.button(
                    "🚀 Process Document",
                    type="primary",
                    use_container_width=True,
                    disabled=st.session_state.processing_status.get("rag_completed", False),
                ):
                    process_document(uploaded_file, use_preprocessing, chunk_size, search_k)

        # System status
        st.markdown("---")
        create_system_status()

        # Sample queries (if document is processed)
        if st.session_state.get("rag_assistant"):
            st.markdown("---")
            create_sample_queries()

        # Advanced features
        if st.session_state.get("rag_assistant"):
            st.markdown("---")
            create_advanced_features_sidebar()


def create_system_status():
    """Create system status indicators"""
    st.markdown("### 🔧 System Status")

    status_items = [
        ("File Upload", st.session_state.processing_status.get("file_uploaded", False)),
        ("OCR Processing", st.session_state.processing_status.get("ocr_completed", False)),
        ("Vector Embeddings", st.session_state.processing_status.get("embeddings_completed", False)),
        ("RAG Assistant", st.session_state.processing_status.get("rag_completed", False)),
    ]

    for item, completed in status_items:
        status_class = "status-success" if completed else "status-danger"
        icon = "✅" if completed else "⏳"
        st.markdown(f'<span class="status-indicator {status_class}"></span>{icon} {item}', unsafe_allow_html=True)


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
        "Can I sublet the property?",
    ]

    # Create columns for better layout
    for i in range(0, len(sample_queries), 2):
        col1, col2 = st.columns(2)

        with col1:
            if i < len(sample_queries):
                if st.button(
                    f"❓ {sample_queries[i]}", key=f"sample_{i}", use_container_width=True, help="Click to ask this question"
                ):
                    handle_sample_query(sample_queries[i])

        with col2:
            if i + 1 < len(sample_queries):
                if st.button(
                    f"❓ {sample_queries[i + 1]}",
                    key=f"sample_{i + 1}",
                    use_container_width=True,
                    help="Click to ask this question",
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
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                file_content = uploaded_file.getvalue()
                tmp_file.write(file_content)
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

            progress_bar.progress(25)
            elapsed = time.time() - start_time
            time_display.text(f"⏱️ Elapsed: {elapsed:.1f}s")

            # Step 3: Extract text
            status_text.info(f"🔍 Extracting text with {ocr_type}...")
            
            if hasattr(extractor, "extract_from_file"):
                extraction_result = extractor.extract_from_file(tmp_path, preprocess=use_preprocessing)
            else:
                extraction_result = extractor.extract_text()

            st.session_state.extraction_result = extraction_result
            st.session_state.processing_status["ocr_completed"] = True

            # Store debug info in session state for persistent display
            extracted_text = extraction_result.get("text", "")
            confidence = extraction_result.get("confidence", 0)

            # Check for rent keywords
            rent_keywords = ["rent", "monthly", "$", "£", "payment"]
            found_keywords = [kw for kw in rent_keywords if kw.lower() in extracted_text.lower()]

            # Enhanced debug info for persistent display
            st.session_state.debug_info = {
                "processing_time": time.time() - start_time,
                "file_info": {
                    "name": uploaded_file.name,
                    "size_bytes": uploaded_file.size,
                    "type": uploaded_file.type,
                    "temp_path": tmp_path,
                    "temp_size": os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
                },
                "ocr_engine": ocr_type,
                "preprocessing": {
                    "enabled": use_preprocessing,
                    "opencv_available": OPENCV_AVAILABLE,
                },
                "extraction_results": {
                    "confidence": confidence,
                    "characters_extracted": len(extracted_text),
                    "words_extracted": len(extracted_text.split()),
                    "pages_processed": extraction_result.get("page_count", 1),
                    "multi_page": extraction_result.get("multi_page_extraction", False),
                    "is_mock": extraction_result.get("mock_extraction", False),
                    "textract_debug": extraction_result.get("debug_info", {}),
                },
                "content_analysis": {
                    "rent_keywords_found": found_keywords,
                    "has_dollar_signs": "$" in extracted_text,
                    "has_numbers": any(char.isdigit() for char in extracted_text),
                    "word_density": len(extracted_text.split()) / len(extracted_text) if len(extracted_text) > 0 else 0,
                },
                "extracted_text_preview": extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
                "full_text_length": len(extracted_text)
            }

            progress_bar.progress(50)
            elapsed = time.time() - start_time
            time_display.text(f"⏱️ Elapsed: {elapsed:.1f}s")

            # Step 4: Create embeddings
            status_text.info("🧠 Creating vector embeddings...")

            vector_store = LeaseVectorStore()
            vector_store.add_document(
                extraction_result["text"],
                doc_id="uploaded_lease",
                source_info={
                    "filename": uploaded_file.name,
                    "upload_time": datetime.now().isoformat(),
                    "preprocessing_used": use_preprocessing,
                },
            )

            st.session_state.vector_store = vector_store
            st.session_state.processing_status["embeddings_completed"] = True

            # Store vector store debug info in session state
            chunk_debug_info = []
            for i, chunk in enumerate(vector_store.chunks):
                words = len(chunk["text"].split())
                has_rent = any(word in chunk["text"].lower() for word in ["rent", "monthly"])
                has_money = any(symbol in chunk["text"] for symbol in ["$", "£", "€", "¥", "£"]) or any(
                    word in chunk["text"].lower() for word in ["dollar", "pound", "euro", "usd", "gbp"]
                )

                chunk_debug_info.append(
                    {
                        "chunk_id": i,
                        "words": words,
                        "has_rent": has_rent,
                        "has_money": has_money,
                        "preview": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                    }
                )

            st.session_state.chunk_debug_info = {"total_chunks": len(vector_store.chunks), "chunks": chunk_debug_info}

            progress_bar.progress(75)
            elapsed = time.time() - start_time
            time_display.text(f"⏱️ Elapsed: {elapsed:.1f}s")

            # Step 5: Initialize RAG assistant
            status_text.info("🤖 Setting up AI assistant...")

            rag_assistant = LeaseRAGAssistant(vector_store)
            st.session_state.rag_assistant = rag_assistant
            st.session_state.processing_status["rag_completed"] = True

            progress_bar.progress(100)
            elapsed = time.time() - start_time
            time_display.text(f"⏱️ Total time: {elapsed:.1f}s")

            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

            # Update document stats
            st.session_state.document_stats = {
                "filename": uploaded_file.name,
                "file_size": len(uploaded_file.getvalue()),
                "character_count": len(extraction_result["text"]),
                "confidence": extraction_result.get("confidence", 0),
                "line_count": extraction_result.get("line_count", 0),
                "chunk_count": len(vector_store.texts),
                "preprocessing_used": use_preprocessing,
                "processing_time": elapsed,
                "ocr_type": ocr_type,
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
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)


def create_mock_extractor():
    """Create a mock extractor for demo purposes"""

    class MockExtractor:
        def extract_from_file(self, file_path, preprocess=True):
            return {
                "text": """RESIDENTIAL LEASE AGREEMENT
                
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
                "confidence": 94.5,
                "line_count": 35,
                "character_count": 1247,
            }

        def extract_text(self):
            return self.extract_from_file("", False)

    return MockExtractor()


def generate_welcome_message(filename, extraction_result):
    """Generate a personalized welcome message"""
    char_count = len(extraction_result["text"])
    confidence = extraction_result.get("confidence", 0)

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
        "file_uploaded": True,
        "ocr_completed": True,
        "embeddings_completed": True,
        "rag_completed": True,
    }
    st.session_state.document_stats = {
        "filename": "demo_lease.pdf",
        "character_count": len(demo_lease_text),
        "confidence": 95.0,
        "chunk_count": len(vector_store.texts),
        "ocr_type": "Demo Mode",
    }

    # Add welcome message
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Welcome to the LeaseLens demo! I've loaded a sample lease document. Try asking questions like 'What is the monthly rent?' or 'Are pets allowed?'",
        }
    ]


def display_document_processing_panel():
    """Display a professional document processing summary panel"""
    if not ("debug_info" in st.session_state or "chunk_debug_info" in st.session_state):
        return
    
    st.markdown("---")
    st.subheader('📊 Document Processing Summary')
    
    debug = st.session_state.get("debug_info", {})
    chunk_debug = st.session_state.get("chunk_debug_info", {})
    
    # File Information Section
    with st.expander('📁 File Information', expanded=True):
        file_info = debug.get("file_info", {})
        extraction = debug.get("extraction_results", {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('File Name', file_info.get('name', 'Unknown'))
            st.metric('File Size', f"{file_info.get('size_bytes', 0):,} bytes")
        with col2:
            st.metric('File Type', file_info.get('type', 'Unknown'))
            st.metric('Pages Processed', extraction.get('pages_processed', 1))
        with col3:
            upload_time = debug.get('processing_time', 0)
            st.metric('Processing Time', f"{upload_time:.2f}s")
            success = extraction.get('characters_extracted', 0) > 0
            st.metric('Status', '✅ Processed' if success else '❌ Failed')
    
    # OCR Engine Details Section
    with st.expander('🔍 OCR Engine Details'):
        preprocessing = debug.get("preprocessing", {})
        col1, col2 = st.columns(2)
        with col1:
            engine = debug.get("ocr_engine", "Unknown")
            is_mock = extraction.get("is_mock", False)
            engine_display = f"{engine} {'(Demo Mode)' if is_mock else '(Production)'}"
            st.write(f'**Engine:** {engine_display}')
            st.write(f'**Region:** us-east-1')
            st.write(f'**Processing Method:** {"Mock Extraction" if is_mock else "AWS Textract"}')
        with col2:
            st.write(f'**Preprocessing:** {"✅ Enabled" if preprocessing.get("enabled") else "❌ Disabled"}')
            st.write(f'**OpenCV Available:** {"✅ Yes" if preprocessing.get("opencv_available") else "❌ No"}')
            confidence = extraction.get("confidence", 0)
            if confidence >= 90:
                conf_status = "🟢 Excellent"
            elif confidence >= 75:
                conf_status = "🟡 Good"
            else:
                conf_status = "🔴 Poor"
            st.write(f'**Quality:** {conf_status} ({confidence:.1f}%)')
    
    # Extraction Results Section  
    with st.expander('📄 Extraction Results'):
        if extraction.get('characters_extracted', 0) > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric('Characters Extracted', f"{extraction.get('characters_extracted', 0):,}")
                st.metric('Confidence Score', f"{extraction.get('confidence', 0):.1f}%")
            with col2:
                st.metric('Lines Detected', extraction.get('line_count', 0))
                st.metric('Words Extracted', f"{extraction.get('words_extracted', 0):,}")
            with col3:
                st.metric('Pages Processed', extraction.get('pages_processed', 1))
                multi_page = extraction.get('multi_page', False)
                st.metric('Multi-page', '✅ Yes' if multi_page else '❌ No')
                
            # Content Quality Indicators
            st.markdown("#### 📈 Content Quality Indicators")
            content = debug.get("content_analysis", {})
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                keywords = content.get("rent_keywords_found", [])
                if keywords:
                    st.success(f"✅ Keywords: {', '.join(keywords[:3])}")
                else:
                    st.error("❌ No rent keywords found")
            
            with col2:
                if content.get("has_dollar_signs", False):
                    st.success("✅ Currency symbols detected")
                else:
                    st.warning("⚠️ No currency symbols")
            
            with col3:
                if content.get("has_numbers", False):
                    st.success("✅ Numbers detected")
                else:
                    st.warning("⚠️ No numbers detected")
                    
            with col4:
                word_density = content.get("word_density", 0)
                if word_density > 0.05:
                    st.success(f"✅ Good text density")
                else:
                    st.warning(f"⚠️ Low text density")
        else:
            st.error('❌ No text was extracted from the document')
            st.write("**Possible causes:**")
            st.write("• Document may be an image without text")
            st.write("• OCR service may not be available") 
            st.write("• File may be corrupted or unreadable")
    
    # Extracted Text Preview Section
    with st.expander('📝 Extracted Text Preview'):
        extracted_text = debug.get('extracted_text_preview', '')
        full_length = debug.get('full_text_length', 0)
        
        if extracted_text:
            # Show preview in a text area
            preview_length = min(500, len(extracted_text))
            st.text_area(
                f'First {preview_length} characters:',
                value=extracted_text[:preview_length] + ('...' if full_length > preview_length else ''),
                height=150,
                disabled=True,
                key="text_preview"
            )
            
            # Text statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f'**Total Characters:** {full_length:,}')
            with col2:
                word_count = extraction.get('words_extracted', 0)
                st.write(f'**Word Count:** {word_count:,}')
            with col3:
                reading_time = max(1, word_count // 200)
                st.write(f'**Est. Reading Time:** {reading_time} min')
                
            # Quality assessment
            if full_length > 1000:
                st.success("✅ Document appears to have substantial content")
            elif full_length > 100:
                st.info("ℹ️ Document has moderate content")
            else:
                st.warning("⚠️ Document has minimal content - check OCR quality")
        else:
            st.warning('⚠️ No text preview available')
    
    # Vector Store Results Section
    with st.expander('🗂️ Vector Store Results'):
        if chunk_debug.get('total_chunks', 0) > 0:
            col1, col2 = st.columns(2)
            with col1:
                st.metric('Chunks Created', chunk_debug.get('total_chunks', 0))
                st.metric('Embedding Model', 'TF-IDF + Sentence Transformers')
                st.metric('Chunk Size', 'Variable (sentence-based)')
            with col2:
                st.metric('Total Documents', 1)
                st.metric('Vector Store Type', 'Simple TF-IDF')
                st.metric('Search Ready', '✅ Yes')
                
            # Show chunk quality analysis
            st.markdown("#### 📊 Chunk Quality Analysis")
            chunks = chunk_debug.get('chunks', [])
            if chunks:
                rent_chunks = sum(1 for chunk in chunks if chunk.get('has_rent', False))
                money_chunks = sum(1 for chunk in chunks if chunk.get('has_money', False))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric('Chunks with Rent Terms', rent_chunks)
                with col2:
                    st.metric('Chunks with Money Info', money_chunks)
                with col3:
                    quality_score = (rent_chunks + money_chunks) / len(chunks) * 100
                    st.metric('Content Quality', f"{quality_score:.1f}%")
                    
                # Show sample chunks
                st.markdown("#### 📝 Sample Chunks")
                for i, chunk in enumerate(chunks[:3]):
                    with st.container():
                        chunk_quality = []
                        if chunk.get('has_rent'): chunk_quality.append("🏠 Rent")
                        if chunk.get('has_money'): chunk_quality.append("💰 Money")
                        quality_str = " | ".join(chunk_quality) if chunk_quality else "📄 General"
                        
                        st.write(f"**Chunk {i+1}** ({chunk.get('words', 0)} words) - {quality_str}")
                        st.code(chunk.get('preview', '')[:150] + ("..." if len(chunk.get('preview', '')) > 150 else ""))
        else:
            st.error('❌ Vector store processing failed')
            st.write("**Possible causes:**")
            st.write("• No text was extracted from the document")
            st.write("• Text processing pipeline encountered an error")


def display_debug_info():
    """Display comprehensive debug information"""
    if "debug_info" in st.session_state or "chunk_debug_info" in st.session_state:
        with st.expander("🔍 COMPREHENSIVE DEBUG INFORMATION", expanded=False):
            if "debug_info" in st.session_state:
                debug = st.session_state.debug_info
                
                # File Information Section
                st.write("### 📁 File Information")
                file_info = debug.get("file_info", {})
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Original File:**")
                    st.write(f"• Name: `{file_info.get('name', 'Unknown')}`")
                    st.write(f"• Size: `{file_info.get('size_bytes', 0):,} bytes`")
                    st.write(f"• Type: `{file_info.get('type', 'Unknown')}`")
                
                with col2:
                    st.write(f"**Temp File:**")
                    st.write(f"• Path: `{file_info.get('temp_path', 'Unknown')}`")
                    st.write(f"• Size: `{file_info.get('temp_size', 0):,} bytes`")
                    st.write(f"• Processing time: `{debug.get('processing_time', 0):.2f}s`")

                # OCR Engine Information
                st.write("### 🔧 OCR Engine Details")
                preprocessing = debug.get("preprocessing", {})
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("OCR Engine", debug.get("ocr_engine", "Unknown"))
                    st.write(f"• Preprocessing enabled: `{preprocessing.get('enabled', False)}`")
                    st.write(f"• OpenCV available: `{preprocessing.get('opencv_available', False)}`")
                
                with col2:
                    extraction = debug.get("extraction_results", {})
                    st.metric("Confidence", f"{extraction.get('confidence', 0):.1f}%")
                    st.write(f"• Is mock extraction: `{extraction.get('is_mock', False)}`")
                    st.write(f"• Multi-page: `{extraction.get('multi_page', False)}`")

                # Extraction Results
                st.write("### 📊 Extraction Results")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Characters", f"{extraction.get('characters_extracted', 0):,}")
                    st.metric("Words", f"{extraction.get('words_extracted', 0):,}")
                
                with col2:
                    st.metric("Pages", extraction.get('pages_processed', 1))
                    content = debug.get("content_analysis", {})
                    word_density = content.get('word_density', 0)
                    st.metric("Word Density", f"{word_density:.3f}")
                
                with col3:
                    st.write("**Content Flags:**")
                    if content.get('has_dollar_signs', False):
                        st.success("✅ Has $ signs")
                    else:
                        st.error("❌ No $ signs")
                    
                    if content.get('has_numbers', False):
                        st.success("✅ Has numbers")
                    else:
                        st.error("❌ No numbers")

                # Keywords Analysis
                st.write("### 🔍 Content Analysis")
                keywords = content.get("rent_keywords_found", [])
                if keywords:
                    st.success(f"✅ **Rent keywords found:** {', '.join(keywords)}")
                else:
                    st.error("❌ **No rent keywords found!** This indicates a potential OCR problem.")

                # Textract Debug Info (if available)
                textract_debug = extraction.get("textract_debug", {})
                if textract_debug:
                    st.write("### 🔬 AWS Textract Debug")
                    st.json(textract_debug)

                # Text Preview
                st.write("### 📝 Extracted Text Preview")
                preview_text = debug.get("extracted_text_preview", "")
                if preview_text:
                    st.code(preview_text, language="text")
                    st.write(f"**Full text length:** {debug.get('full_text_length', 0):,} characters")
                else:
                    st.error("❌ NO TEXT EXTRACTED! This is a critical problem.")

            if "chunk_debug_info" in st.session_state:
                chunk_debug = st.session_state.chunk_debug_info
                st.write("### Vector Store Results")
                st.metric("Total Chunks", chunk_debug["total_chunks"])

                for chunk_info in chunk_debug["chunks"]:
                    with st.container():
                        st.write(f"**Chunk {chunk_info['chunk_id']}:**")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Words", chunk_info["words"])
                        with col2:
                            status = "✅" if chunk_info["has_rent"] else "❌"
                            st.write(f"Rent terms: {status}")
                        with col3:
                            status = "✅" if chunk_info["has_money"] else "❌"
                            st.write(f"Money symbols: {status}")

                        st.code(chunk_info["preview"])


def create_chat_interface():
    """Create the main chat interface"""
    st.markdown("## 💬 Chat with Your Lease")

    # Display professional document processing panel
    display_document_processing_panel()

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
                            clean_text = clean_source_display_text(source["text"])
                            doc_info = f" from {source.get('doc_id', 'document')}" if source.get("doc_id") else ""
                            st.markdown(f"**Source {j+1}**{doc_info} (Relevance: {source['score']:.1%})")
                            st.markdown(f"```\n{clean_text}\n```")

            # Show confidence score
            if message["role"] == "assistant" and "confidence" in message:
                confidence = message["confidence"]
                if confidence > 0:
                    color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                    st.markdown(
                        f"**Confidence:** <span style='color: {color}'>{confidence:.1%}</span>", unsafe_allow_html=True
                    )

    # Chat input
    if prompt := st.chat_input("Ask about your lease...", key="main_chat_input"):
        handle_user_input(prompt)


def handle_user_input(prompt):
    """Handle user chat input"""
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Update conversation stats
    st.session_state.conversation_stats["total_questions"] += 1

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyzing your question..."):
            response = st.session_state.rag_assistant.query(prompt)

            # Display answer
            st.markdown(response["answer"])

            # Update conversation stats
            if response["confidence"] > 0:
                current_avg = st.session_state.conversation_stats["avg_confidence"]
                total_questions = st.session_state.conversation_stats["total_questions"]
                new_avg = ((current_avg * (total_questions - 1)) + response["confidence"]) / total_questions
                st.session_state.conversation_stats["avg_confidence"] = new_avg

            # Store response with sources
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response["sources"],
                    "confidence": response["confidence"],
                }
            )


def handle_sample_query(query):
    """Handle sample query button clicks"""
    if st.session_state.get("rag_assistant"):
        # Add to chat history
        st.session_state.messages.append({"role": "user", "content": query})

        # Get response
        response = st.session_state.rag_assistant.query(query)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response["answer"],
                "sources": response["sources"],
                "confidence": response["confidence"],
            }
        )

        # Update stats
        st.session_state.conversation_stats["total_questions"] += 1

        st.rerun()


def show_welcome_screen():
    """Show welcome screen when no document is processed"""
    st.markdown("## 👋 Welcome to LeaseLens!")

    # Feature overview
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        <div class="feature-card">
            <h4>🔍 Smart OCR</h4>
            <p>Advanced document processing with AWS Textract and OpenCV preprocessing for maximum accuracy.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="feature-card">
            <h4>🧠 AI Understanding</h4>
            <p>Semantic search and natural language processing to understand your lease terms.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="feature-card">
            <h4>📚 Source Citations</h4>
            <p>Every answer includes exact source references to prevent AI hallucination.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # How it works
    st.markdown("### 🚀 How LeaseLens Works")

    steps_col1, steps_col2 = st.columns(2)

    with steps_col1:
        st.markdown(
            """
        **1. Upload Your Document**
        - PDF or image formats supported
        - Automatic preprocessing for best results
        
        **2. AI Processing**
        - OCR extracts text with high accuracy
        - Creates semantic embeddings for search
        """
        )

    with steps_col2:
        st.markdown(
            """
        **3. Ask Questions**
        - Natural language queries
        - Get instant, accurate answers
        
        **4. Verify Sources**
        - See exact lease text supporting each answer
        - Confidence scores for transparency
        """
        )

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
    st.markdown(
        """
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
    """
    )


def create_analytics_panel():
    """Create the analytics and statistics panel"""
    st.markdown("## 📊 Document Analytics")

    if st.session_state.get("document_stats"):
        stats = st.session_state.document_stats

        # Document overview
        st.markdown("### 📄 Document Overview")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Document", stats.get("filename", "Unknown"))
            st.metric("OCR Confidence", f"{stats.get('confidence', 0):.1f}%")
            st.metric("Processing Time", f"{stats.get('processing_time', 0):.1f}s")

        with col2:
            st.metric("Characters", f"{stats.get('character_count', 0):,}")
            st.metric("Text Chunks", stats.get("chunk_count", 0))
            st.metric("OCR Engine", stats.get("ocr_type", "Unknown"))

        # Confidence visualization
        if stats.get("confidence"):
            st.markdown("### 📈 Processing Quality")

            confidence = stats["confidence"]
            if PLOTLY_AVAILABLE and go is not None:
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=confidence,
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": "OCR Confidence %"},
                        gauge={
                            "axis": {"range": [None, 100]},
                            "bar": {"color": "darkblue"},
                            "steps": [
                                {"range": [0, 60], "color": "lightgray"},
                                {"range": [60, 80], "color": "yellow"},
                                {"range": [80, 100], "color": "green"},
                            ],
                            "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 90},
                        },
                    )
                )
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback to simple metric display without plotly
                st.metric("OCR Confidence", f"{confidence:.1f}%")
                if confidence >= 90:
                    st.success("🟢 Excellent quality")
                elif confidence >= 80:
                    st.info("🟡 Good quality")
                elif confidence >= 60:
                    st.warning("🟠 Fair quality")
                else:
                    st.error("🔴 Poor quality")

        # Conversation statistics
        if st.session_state.messages and len(st.session_state.messages) > 1:
            st.markdown("### 💬 Conversation Stats")

            conv_stats = st.session_state.conversation_stats

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Questions Asked", conv_stats["total_questions"])

            with col2:
                avg_conf = conv_stats.get("avg_confidence", 0)
                st.metric("Avg Confidence", f"{avg_conf:.1%}")

            # Session duration
            if "session_start" in conv_stats:
                duration = datetime.now() - conv_stats["session_start"]
                minutes = int(duration.total_seconds() / 60)
                st.metric("Session Duration", f"{minutes} min")

        # Advanced features results
        if st.session_state.advanced_features.get("lease_summary"):
            st.markdown("### 📋 Latest Analysis")

            # Safe length checks with proper defaults
            lease_summary = st.session_state.advanced_features.get("lease_summary")
            risk_analysis = st.session_state.advanced_features.get("risk_analysis")

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
            "LLM": "Anthropic Claude",
            "Framework": "Streamlit + Python",
        }

        for capability, technology in capabilities.items():
            st.text(f"{capability}: {technology}")


def generate_lease_summary():
    """Generate lease summary and store for display"""
    if not st.session_state.get("rag_assistant"):
        st.error("❌ Please process a document first")
        return

    with st.spinner("📋 Generating comprehensive lease summary..."):
        try:
            summary = st.session_state.rag_assistant.get_lease_summary()
            st.session_state.advanced_features["lease_summary"] = summary
            st.session_state.show_lease_summary = True
            st.success("✅ Lease summary generated! Check the main area.")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Failed to generate summary: {e}")


def display_lease_summary():
    """Display the generated lease summary in main area"""
    summary = st.session_state.advanced_features.get("lease_summary")
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
        if any(word in key_lower for word in ["rent", "deposit", "fee", "cost", "payment"]):
            financial_terms[key] = value
        elif any(word in key_lower for word in ["pet", "guest", "noise", "smoking", "policies", "sublet"]):
            policies[key] = value
        elif any(word in key_lower for word in ["maintenance", "repair", "utility", "utilities", "responsibilities"]):
            responsibilities[key] = value
        elif any(word in key_lower for word in ["parking", "termination", "lease term", "duration"]):
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
    if not st.session_state.get("rag_assistant"):
        st.error("❌ Please process a document first")
        return

    with st.spinner("⚠️ Analyzing lease risks and red flags..."):
        try:
            risks = st.session_state.rag_assistant.analyze_lease_risks()
            st.session_state.advanced_features["risk_analysis"] = risks
            st.session_state.show_risk_analysis = True
            st.success("✅ Risk analysis generated! Check the main area.")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Failed to analyze risks: {e}")


def display_risk_analysis():
    """Display the generated risk analysis in main area"""
    risks = st.session_state.advanced_features.get("risk_analysis")
    if not risks:
        return

    # Display risk analysis
    st.markdown("## ⚠️ Lease Risk Analysis")

    if not risks:
        st.info("✅ No significant risks identified in this lease.")
        return

    # Categorize risks by level
    high_risks = [r for r in risks if r.get("risk_level") == "HIGH"]
    medium_risks = [r for r in risks if r.get("risk_level") == "MEDIUM"]
    low_risks = [r for r in risks if r.get("risk_level") == "LOW"]

    # Risk overview
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🔴 High Risk", len(high_risks))
    with col2:
        st.metric("🟡 Medium Risk", len(medium_risks))
    with col3:
        st.metric("🟢 Low Risk", len(low_risks))

    # Display risks by category
    for risk_level, risk_list, color in [("HIGH", high_risks, "🔴"), ("MEDIUM", medium_risks, "🟡"), ("LOW", low_risks, "🟢")]:
        if risk_list:
            st.markdown(f"### {color} {risk_level} Risk Items")

            for risk in risk_list:
                with st.expander(f"{risk['category']} (Confidence: {risk['confidence']:.1%})"):
                    st.write(risk["description"])

                    if risk.get("sources"):
                        st.markdown("**Sources:**")
                        for i, source in enumerate(risk["sources"][:2], 1):
                            clean_text = clean_source_display_text(source["text"])[:100] + "..."
                            st.markdown(f"Source {i}: `{clean_text}`")

    st.success("✅ Risk analysis completed!")


def extract_key_figures():
    """Extract key figures and store for display"""
    if not st.session_state.get("rag_assistant"):
        st.error("❌ Please process a document first")
        return

    with st.spinner("💰 Extracting key financial figures..."):
        try:
            figures = st.session_state.rag_assistant.extract_key_figures()
            st.session_state.advanced_features["key_figures"] = figures
            st.session_state.show_key_figures = True
            st.success("✅ Key figures extracted! Check the main area.")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Failed to extract key figures: {e}")


def display_key_figures():
    """Display the extracted key figures in main area"""
    figures = st.session_state.advanced_features.get("key_figures")
    if not figures:
        return

    # Display key figures
    st.markdown("## 💰 Key Financial Figures")

    if figures.get("financial"):
        st.markdown("### 💵 Financial Breakdown")

        # Create a summary table
        financial_data = []
        for category, data in figures["financial"].items():
            amounts = data.get("amounts", [])
            confidence = data.get("confidence", 0)

            financial_data.append(
                {
                    "Category": category.title(),
                    "Amount": amounts[0] if amounts else "Not specified",
                    "Confidence": f"{confidence:.1%}",
                    "Additional Amounts": ", ".join(amounts[1:]) if len(amounts) > 1 else "None",
                }
            )

        if financial_data:
            if PANDAS_AVAILABLE and pd is not None:
                df = pd.DataFrame(financial_data)
                st.dataframe(df, use_container_width=True)
            else:
                # Fallback to basic table display without pandas
                st.markdown("#### Financial Terms")
                for item in financial_data:
                    st.markdown(f"**{item['Term']}:** {item['Amount']}")
                    if item['Notes']:
                        st.markdown(f"  *{item['Notes']}*")

    # Cost estimates
    if figures.get("summary"):
        st.markdown("### 📊 Cost Estimates")

        col1, col2 = st.columns(2)

        with col1:
            monthly_cost = figures["summary"].get("estimated_monthly_cost", "Unable to calculate")
            st.metric("Estimated Monthly Cost", monthly_cost)

        with col2:
            move_in_cost = figures["summary"].get("move_in_cost", "Unable to calculate")
            st.metric("Estimated Move-in Cost", move_in_cost)

    # Date information
    if figures.get("dates"):
        st.markdown("### 📅 Important Dates")

        for category, data in figures["dates"].items():
            st.markdown(f"**{category.title()}:** {data['raw_answer'][:100]}...")

    st.success("✅ Key figures extracted successfully!")


def create_footer():
    """Create application footer"""
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        **🏠 LeaseLens**  
        AI-Powered Lease Assistant
        """
        )

    with col2:
        st.markdown(
            """
        **🔗 Links**  
        [GitHub Repository](https://github.com/yourusername/lease-lens-demo)  
        [Documentation](https://github.com/yourusername/lease-lens-demo/wiki)
        """
        )

    with col3:
        st.markdown(
            f"""
        **📊 Session Info**  
        Started: {st.session_state.conversation_stats.get('session_start', datetime.now()).strftime('%H:%M')}  
        Questions: {st.session_state.conversation_stats.get('total_questions', 0)}
        """
        )


# Additional utility functions for enhanced features


def export_conversation():
    """Export conversation history"""
    if not st.session_state.messages:
        st.warning("No conversation to export")
        return

    # Prepare conversation data
    conversation_data = {
        "document": st.session_state.document_stats.get("filename", "Unknown"),
        "timestamp": datetime.now().isoformat(),
        "messages": st.session_state.messages,
        "stats": st.session_state.conversation_stats,
    }

    # Convert to JSON
    json_data = json.dumps(conversation_data, indent=2, ensure_ascii=False)

    # Provide download
    st.download_button(
        label="📥 Download Conversation",
        data=json_data,
        file_name=f"leaselens_conversation_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
