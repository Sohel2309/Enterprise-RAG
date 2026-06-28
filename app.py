import subprocess
import threading
import time
import sys
import os

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def start_fastapi():
    """Start FastAPI in background."""
    import uvicorn
    uvicorn.run(
        "api.rag_api:app",
        host="0.0.0.0",
        port=8000,
        log_level="error"
    )

def wait_for_api(max_retries=30):
    """Wait for API to be ready."""
    import requests
    for i in range(max_retries):
        try:
            r = requests.get("http://localhost:8000/health", timeout=2)
            if r.ok:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False

if __name__ == "__main__":
    # Start API in background thread
    api_thread = threading.Thread(target=start_fastapi, daemon=True)
    api_thread.start()
    
    # Wait for API
    wait_for_api()
    
    # Start Streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "ui/streamlit_app.py",
        "--server.port=7860",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false"
    ])