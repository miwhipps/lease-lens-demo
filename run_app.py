#!/usr/bin/env python3
"""
LeaseLens Application Launcher
Provides multiple ways to run the application with proper error handling
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path


def check_dependencies():
    """Check if all required dependencies are available"""
    required_packages = ["streamlit", "opencv-python", "sentence-transformers", "faiss-cpu", "plotly", "pandas"]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n🔧 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False

    print("✅ All dependencies are available")
    return True


def check_environment():
    """Check environment configuration"""
    env_file = Path(".env")

    if not env_file.exists():
        print("⚠️ .env file not found")
        print("   Creating .env file from template...")

        env_template = Path(".env.example")
        if env_template.exists():
            import shutil

            shutil.copy(env_template, env_file)
            print("✅ Created .env file from template")
            print("   Please add your API keys to .env")
        else:
            print("❌ No .env.example template found")
            return False

    return True


def run_streamlit(args):
    """Run the Streamlit application"""

    # Check dependencies
    if not check_dependencies():
        return False

    # Check environment
    if not check_environment():
        return False

    # Build streamlit command
    cmd = ["streamlit", "run", "streamlit_app.py"]

    if args.port:
        cmd.extend(["--server.port", str(args.port)])

    if args.host:
        cmd.extend(["--server.address", args.host])

    if args.debug:
        cmd.append("--logger.level=debug")

    print(f"🚀 Starting LeaseLens...")
    print(f"   Command: {' '.join(cmd)}")
    print(f"   URL: http://{args.host or 'localhost'}:{args.port or 8501}")
    print()

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start application: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        return True

    return True


def run_tests():
    """Run application tests"""
    print("🧪 Running LeaseLens tests...")

    test_files = ["test_all_components.py", "test_ocr_only.py", "test_vector_only.py", "test_rag_only.py"]

    available_tests = [f for f in test_files if Path(f).exists()]

    if not available_tests:
        print("❌ No test files found")
        return False

    print(f"Found {len(available_tests)} test files")

    for test_file in available_tests:
        print(f"\n📋 Running {test_file}...")
        try:
            result = subprocess.run([sys.executable, test_file], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {test_file} passed")
            else:
                print(f"❌ {test_file} failed:")
                print(result.stderr)
        except Exception as e:
            print(f"❌ Error running {test_file}: {e}")

    return True


def show_status():
    """Show application status"""
    print("📊 LeaseLens Status Report")
    print("=" * 30)

    # Check files
    required_files = [
        "streamlit_app.py",
        "requirements.txt",
        "ocr_pipeline/preprocess.py",
        "embeddings/vector_store.py",
        "ai_assistant/rag_chat.py",
    ]

    print("\n📁 Required Files:")
    for file_path in required_files:
        exists = Path(file_path).exists()
        status = "✅" if exists else "❌"
        print(f"   {status} {file_path}")

    # Check dependencies
    print("\n📦 Dependencies:")
    check_dependencies()

    # Check environment
    print("\n🔧 Environment:")
    check_environment()

    # Show system info
    print(f"\n💻 System Info:")
    print(f"   Python: {sys.version}")
    print(f"   Working Directory: {os.getcwd()}")
    print(f"   Platform: {sys.platform}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="LeaseLens Application Launcher")

    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=["run", "test", "status", "install"],
        help="Command to execute (default: run)",
    )

    parser.add_argument("--port", type=int, default=8501, help="Port to run the application (default: 8501)")

    parser.add_argument("--host", default="localhost", help="Host to bind the application (default: localhost)")

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.command == "run":
        success = run_streamlit(args)
    elif args.command == "test":
        success = run_tests()
    elif args.command == "status":
        show_status()
        success = True
    elif args.command == "install":
        print("🔧 Installing dependencies...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
            print("✅ Dependencies installed successfully")
            success = True
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            success = False
    else:
        parser.print_help()
        success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
