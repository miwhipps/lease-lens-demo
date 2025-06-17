#!/usr/bin/env python3
"""
Safe startup script for LeaseLens that avoids multiprocessing issues
"""

import os
import sys

# Set environment variables to avoid multiprocessing issues
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Disable multiprocessing in libraries
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

def main():
    print("🏠 Starting LeaseLens (Safe Mode)")
    print("=" * 40)
    
    # Test components first
    print("🧪 Testing components...")
    
    try:
        from embeddings.vector_store import LeaseVectorStore
        vs = LeaseVectorStore()
        print("✅ Vector store: OK")
    except Exception as e:
        print(f"❌ Vector store: {e}")
        return False
    
    try:
        from ai_assistant.rag_chat import LeaseRAGAssistant
        print("✅ RAG assistant: OK")
    except Exception as e:
        print(f"❌ RAG assistant: {e}")
        return False
    
    # Start Streamlit
    print("\n🚀 Starting Streamlit app...")
    import subprocess
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py'])

if __name__ == "__main__":
    main()
