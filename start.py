#!/usr/bin/env python3
"""
One Ring AI Platform - Startup Script

This script automates the setup and launch of the One Ring AI platform.
It will:
1. Create a Python virtual environment
2. Install all required dependencies
3. Initialize the application
4. Launch the FastAPI server with Uvicorn

Usage:
    python start.py [--no-venv] [--port PORT] [--host HOST]
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from typing import Optional, Tuple

# Project configuration
PROJECT_DIR = Path(__file__).parent.absolute()
VENV_DIR = PROJECT_DIR / ".venv"
REQUIREMENTS_FILES = ["requirements.txt", "requirements-docs.txt"]
PYTHON_EXECUTABLE = "python" if platform.system() == "Windows" else "python3"
PIP_EXECUTABLE = "pip" if platform.system() == "Windows" else "pip3"

# Colors for console output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header():
    """Print the application header."""
    print(f"{Colors.HEADER}{'='*60}")
    print(f"{'One Ring AI Platform'.center(60)}")
    print(f"{'='*60}{Colors.ENDC}\n")

def print_step(step: str):
    """Print a step message."""
    print(f"{Colors.OKBLUE}[*] {step}...{Colors.ENDC}")

def print_success(message: str):
    """Print a success message."""
    print(f"{Colors.OKGREEN}[+] {message}{Colors.ENDC}")

def print_warning(message: str):
    """Print a warning message."""
    print(f"{Colors.WARNING}[!] {message}{Colors.ENDC}")

def print_error(message: str, exit_code: int = 1):
    """Print an error message and exit."""
    print(f"{Colors.FAIL}[!] ERROR: {message}{Colors.ENDC}", file=sys.stderr)
    sys.exit(exit_code)

def run_command(command: str, cwd: Optional[Path] = None, shell: bool = False) -> Tuple[int, str]:
    """Run a shell command and return the exit code and output."""
    try:
        # Always use shell=True on Windows to handle paths with spaces correctly
        use_shell = shell or platform.system() == "Windows"
        if use_shell:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                check=False
            )
        else:
            result = subprocess.run(
                command.split(),
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                check=False
            )
        return result.returncode, result.stdout.strip() + "\n" + result.stderr.strip()
    except Exception as e:
        return 1, str(e)

def check_python_version():
    """Check if the Python version is supported."""
    print_step("Checking Python version")
    if sys.version_info < (3, 9):
        print_error("Python 3.9 or higher is required")
    print_success(f"Using Python {sys.version.split()[0]}")

def create_virtualenv(force_recreate=False):
    """Create a Python virtual environment.
    
    Args:
        force_recreate: If True, delete the existing environment if it exists.
    """
    print_step(f"Creating virtual environment at {VENV_DIR}")
    
    # Check if we need to recreate the environment
    if VENV_DIR.exists() and force_recreate:
        print_warning(f"Removing existing virtual environment at {VENV_DIR}")
        try:
            shutil.rmtree(VENV_DIR)
            print_success("Existing environment removed")
        except Exception as e:
            print_error(f"Failed to remove existing environment: {e}")
    elif VENV_DIR.exists():
        # Check if the environment appears to be valid
        pip_path = VENV_DIR / "Scripts" / "pip.exe" if platform.system() == "Windows" else VENV_DIR / "bin" / "pip"
        if not pip_path.exists():
            print_warning(f"Virtual environment at {VENV_DIR} appears to be corrupted (pip not found)")
            try:
                shutil.rmtree(VENV_DIR)
                print_success("Corrupted environment removed")
            except Exception as e:
                print_error(f"Failed to remove corrupted environment: {e}")
        else:
            print_warning(f"Virtual environment already exists at {VENV_DIR}")
            return
    
    try:
        venv.create(VENV_DIR, with_pip=True)
        print_success("Virtual environment created")
    except Exception as e:
        print_error(f"Failed to create virtual environment: {e}")

def install_dependencies():
    """Install Python dependencies."""
    print_step("Installing dependencies")
    
    # Get the correct pip executable
    pip = VENV_DIR / "Scripts" / "pip"
    if platform.system() != "Windows":
        pip = VENV_DIR / "bin" / "pip"
    
    # Check if pip exists
    if not pip.exists() and platform.system() == "Windows":
        pip = VENV_DIR / "Scripts" / "pip.exe"
        if not pip.exists():
            print_error(f"Pip executable not found at {pip}. Virtual environment may be corrupted.")
    
    # Upgrade pip first
    print_step("Upgrading pip")
    returncode, output = run_command(f"\"{pip}\" install --upgrade pip", shell=True)
    if returncode != 0:
        print_warning(f"Failed to upgrade pip: {output}")
    else:
        print_success("Successfully upgraded pip")
    
    # Install requirements files
    for req_file in REQUIREMENTS_FILES:
        req_path = PROJECT_DIR / req_file
        if req_path.exists():
            print_step(f"Installing from {req_file}")
            returncode, output = run_command(f"\"{pip}\" install -r \"{req_path}\"", shell=True)
            if returncode != 0:
                print_warning(f"Failed to install from {req_file}: {output}")
            else:
                print_success(f"Successfully installed from {req_file}")
    
    # Install core requirements directly
    print_step("Installing core requirements directly")
    returncode, output = run_command(
        f"\"{pip}\" install fastapi uvicorn python-dotenv pydantic loguru", 
        shell=True
    )
    if returncode != 0:
        print_warning(f"Failed to install core requirements: {output}")
    else:
        print_success("Successfully installed core requirements")
    
    # Install the package in development mode (only if setup.py exists)
    setup_file = PROJECT_DIR / "setup.py"
    if setup_file.exists():
        print_step("Installing One Ring package in development mode")
        returncode, output = run_command(f"\"{pip}\" install -e .", shell=True, cwd=PROJECT_DIR)
        if returncode != 0:
            print_warning(f"Failed to install One Ring package: {output}")
        else:
            print_success("Successfully installed One Ring package")
    else:
        print_warning("No setup.py found, skipping package installation")
        
    # Verify uvicorn is installed
    print_step("Verifying uvicorn installation")
    returncode, output = run_command(f"\"{pip}\" show uvicorn", shell=True)
    if returncode != 0:
        print_warning("Uvicorn not found, installing it explicitly")
        returncode, output = run_command(f"\"{pip}\" install uvicorn", shell=True)
        if returncode != 0:
            print_error(f"Failed to install uvicorn: {output}")
        else:
            print_success("Successfully installed uvicorn")
    else:
        print_success("Uvicorn is installed")

def check_ollama():
    """Check if Ollama is installed and running."""
    print_step("Checking Ollama installation")
    
    # Try to get Ollama version
    returncode, output = run_command("ollama --version")
    if returncode != 0:
        print_warning(
            "Ollama is not installed or not in PATH. "
            "Please install Ollama from https://ollama.ai/"
        )
    else:
        print_success(f"Ollama is installed: {output}")
        
        # Check if Ollama server is running
        try:
            import requests
            response = requests.get("http://localhost:11434/api/version", timeout=5)
            if response.status_code == 200:
                print_success("Ollama server is running")
            else:
                print_warning(f"Ollama server returned status code {response.status_code}")
        except Exception as e:
            print_warning(f"Ollama server is not running: {e}")
            print("  You can start the Ollama server with: ollama serve")

def start_application(host: str, port: int):
    """Start the FastAPI application with Uvicorn."""
    print_step("Starting One Ring AI Platform")
    
    # Get the correct Python executable
    python = VENV_DIR / "Scripts" / "python"
    if platform.system() != "Windows":
        python = VENV_DIR / "bin" / "python"
    
    # Set environment variables
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_DIR)
    
    # Check if uvicorn is installed in the virtual environment
    uvicorn_path = VENV_DIR / "Scripts" / "uvicorn"
    if platform.system() != "Windows":
        uvicorn_path = VENV_DIR / "bin" / "uvicorn"
    
    if not uvicorn_path.exists():
        uvicorn_path = VENV_DIR / "Scripts" / "uvicorn.exe"
        if not uvicorn_path.exists():
            print_error("Uvicorn not found in the virtual environment. Please install it with 'pip install uvicorn'.")
    
    print(f"\n{Colors.HEADER}{' Starting One Ring AI Platform ':=^60}{Colors.ENDC}")
    print(f"{Colors.BOLD}Server URL:{Colors.ENDC} http://{host}:{port}")
    print(f"{Colors.BOLD}API Docs:{Colors.ENDC} http://{host}:{port}/docs")
    print(f"{Colors.BOLD}Exit:{Colors.ENDC} Press Ctrl+C to stop the server\n")
    
    try:
        # Start the FastAPI server using the module approach
        cmd = f"\"{python}\" -m uvicorn one_ring.api.app:create_app --host {host} --port {port} --reload"
        print(f"Running command: {cmd}")
        process = subprocess.Popen(
            cmd,
            shell=True,
            cwd=str(PROJECT_DIR),
            env=env
        )
        process.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print_error(f"Failed to start application: {e}")

def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Start the One Ring AI Platform")
    parser.add_argument(
        "--no-venv",
        action="store_true",
        help="Use the system Python instead of creating a virtual environment"
    )
    parser.add_argument(
        "--recreate-venv",
        action="store_true",
        help="Force recreation of the virtual environment"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)"
    )
    parser.add_argument(
        "--install-only",
        action="store_true",
        help="Only install dependencies, don't start the server"
    )
    args = parser.parse_args()
    
    # Print header
    print_header()
    
    # Check Python version
    check_python_version()
    
    # Create virtual environment if not disabled
    if not args.no_venv:
        create_virtualenv(force_recreate=args.recreate_venv)
        # Update Python and pip executables to use the virtual environment
        if VENV_DIR.exists():
            if platform.system() == "Windows":
                global PYTHON_EXECUTABLE, PIP_EXECUTABLE
                PYTHON_EXECUTABLE = str(VENV_DIR / "Scripts" / "python.exe")
                PIP_EXECUTABLE = str(VENV_DIR / "Scripts" / "pip.exe")
            else:
                PYTHON_EXECUTABLE = str(VENV_DIR / "bin" / "python3")
                PIP_EXECUTABLE = str(VENV_DIR / "bin" / "pip3")
    
    # Install dependencies
    install_dependencies()
    
    # Check Ollama
    check_ollama()
    
    # Start the application if not in install-only mode
    if not args.install_only:
        start_application(args.host, args.port)

if __name__ == "__main__":
    main()
