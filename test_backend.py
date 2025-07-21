import sys
import subprocess
import time
import requests
from pathlib import Path

def run_command(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)}")
    return subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

def test_backend():
    print("🚀 Testing backend setup...")
    
    # Start the backend server
    backend_dir = Path(__file__).parent / "backend"
    server = run_command(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=backend_dir
    )
    
    try:
        # Give the server some time to start
        print("⏳ Waiting for server to start...")
        time.sleep(5)
        
        # Test the root endpoint
        print("🔍 Testing API endpoints...")
        response = requests.get("http://localhost:8000/")
        print(f"Root endpoint status: {response.status_code}")
        
        # Test the docs endpoint
        response = requests.get("http://localhost:8000/docs")
        print(f"Docs endpoint status: {response.status_code}")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
    finally:
        # Clean up
        print("🛑 Stopping server...")
        server.terminate()
        server.wait()
        print("✅ Testing complete!")

if __name__ == "__main__":
    test_backend()
