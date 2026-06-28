"""
HuggingFace Spaces entry point.
Starts FastAPI in a background thread, then runs Streamlit.
"""
import subprocess
import threading
import time
import sys
import os

# Set environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def start_fastapi():
    """Start FastAPI server in background thread."""
    import uvicorn
    print("Starting FastAPI on port 8000...")
    uvicorn.run(
        "api.rag_api:app",
        host="0.0.0.0",
        port=8000,
        log_level="error"  # Reduce noise in HF logs
    )


def wait_for_api(max_retries=30):
    """Wait until FastAPI is ready before Streamlit starts."""
    import requests
    for i in range(max_retries):
        try:
            r = requests.get("http://localhost:8000/health", timeout=2)
            if r.ok:
                print(f"FastAPI ready after {i+1} attempts")
                return True
        except Exception:
            pass
        time.sleep(2)
    print("WARNING: FastAPI did not start in time")
    return False


if __name__ == "__main__":
    # Start FastAPI in background thread
    api_thread = threading.Thread(target=start_fastapi, daemon=True)
    api_thread.start()

    # Wait for API to be ready
    wait_for_api()

    # Start Streamlit
    print("Starting Streamlit...")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "ui/streamlit_app.py",
        "--server.port=7860",        # HF Spaces uses port 7860
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false"
    ])