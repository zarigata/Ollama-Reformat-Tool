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
import time
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
        if shell:
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
        return result.returncode, result.stdout.strip()
    except Exception as e:
        return 1, str(e)

def check_python_version():
    """Check if the Python version is supported.
    
    The One Ring AI Platform is optimized for Python 3.11.
    """
    print_step("Checking Python version")
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    # Enforce Python 3.11 specifically
    if version.major != 3 or version.minor != 11:
        print_warning(f"Python 3.11.x is required, but you are using {version_str}")
        print_warning("Please install Python 3.11 and try again")
        print_warning("You can download it from https://www.python.org/downloads/release/python-3118/")
        sys.exit(1)
    
    print_success(f"Using Python {version_str}")

def create_virtualenv(force=False):
    """Create a Python virtual environment.
    
    Args:
        force: If True, recreate the virtual environment even if it exists.
    """
    print_step(f"Setting up virtual environment at {VENV_DIR}")
    
    # Check if virtual environment exists and is valid
    venv_valid = False
    pyvenv_cfg = VENV_DIR / "pyvenv.cfg"
    python_exe = VENV_DIR / "Scripts" / "python.exe" if platform.system() == "Windows" else VENV_DIR / "bin" / "python"
    pip_exe = VENV_DIR / "Scripts" / "pip.exe" if platform.system() == "Windows" else VENV_DIR / "bin" / "pip"
    
    if VENV_DIR.exists() and pyvenv_cfg.exists() and python_exe.exists() and pip_exe.exists() and not force:
        try:
            # Test if the Python in the venv actually works
            result = subprocess.run(
                [str(python_exe), "-c", "print('Virtual environment is working')"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0 and "working" in result.stdout:
                print_success("Existing virtual environment is valid")
                venv_valid = True
            else:
                print_warning("Existing virtual environment appears to be corrupted")
        except Exception as e:
            print_warning(f"Error testing virtual environment: {e}")
    
    # Remove corrupted or forced virtual environment
    if VENV_DIR.exists() and (not venv_valid or force):
        print_step("Removing existing virtual environment")
        try:
            if platform.system() == "Windows":
                # On Windows, some files might be locked, use robocopy to empty the directory
                # Create empty temp dir
                temp_dir = PROJECT_DIR / ".venv_temp"
                temp_dir.mkdir(exist_ok=True)
                
                # Use robocopy to mirror empty dir to venv (effectively deleting contents)
                subprocess.run(
                    ["robocopy", str(temp_dir), str(VENV_DIR), "/MIR"], 
                    capture_output=True,
                    check=False
                )
                
                # Remove temp dir and any remaining venv dir
                shutil.rmtree(temp_dir, ignore_errors=True)
                shutil.rmtree(VENV_DIR, ignore_errors=True)
            else:
                # On Unix, we can usually just remove the directory
                shutil.rmtree(VENV_DIR, ignore_errors=True)
                
            print_success("Removed existing virtual environment")
        except Exception as e:
            print_warning(f"Error removing virtual environment: {e}")
            print_warning("Continuing anyway, but installation may fail")
    
    # Create new virtual environment if needed
    if not venv_valid:
        print_step("Creating new virtual environment")
        try:
            venv.create(VENV_DIR, with_pip=True)
            print_success("Virtual environment created successfully")
        except Exception as e:
            print_error(f"Failed to create virtual environment: {e}")

def install_dependencies():
    """Install Python dependencies."""
    print_step("Installing dependencies")
    
    # Get the correct pip executable
    pip_exe = VENV_DIR / "Scripts" / "pip.exe" if platform.system() == "Windows" else VENV_DIR / "bin" / "pip"
    python_exe = VENV_DIR / "Scripts" / "python.exe" if platform.system() == "Windows" else VENV_DIR / "bin" / "python"
    
    # Check if pip exists and is functional
    if not pip_exe.exists():
        print_warning(f"Pip not found at {pip_exe}. Installing pip...")
        try:
            # First try using the bootstrap method which is more reliable
            get_pip_script = PROJECT_DIR / "get-pip.py"
            
            # Download get-pip.py if it doesn't exist
            if not get_pip_script.exists():
                print_step("Downloading get-pip.py")
                try:
                    import requests
                    response = requests.get("https://bootstrap.pypa.io/get-pip.py")
                    with open(get_pip_script, "wb") as f:
                        f.write(response.content)
                    print_success("Downloaded get-pip.py")
                except ImportError:
                    # If requests is not available, use built-in urllib
                    from urllib.request import urlopen
                    with urlopen("https://bootstrap.pypa.io/get-pip.py") as response, open(get_pip_script, "wb") as f:
                        f.write(response.read())
                    print_success("Downloaded get-pip.py")
                except Exception as e:
                    print_warning(f"Error downloading get-pip.py: {e}")
                    print_warning("Will try ensurepip instead")
            
            # Use get-pip.py if it exists
            if get_pip_script.exists():
                returncode, output = run_command(f"{python_exe} {get_pip_script}")
                if returncode == 0:
                    print_success("Pip installed successfully using get-pip.py")
                    # Refresh pip executable path
                    pip_command = f"{pip_exe}"
                else:
                    print_warning(f"Failed to install pip using get-pip.py: {output}")
                    print_warning("Will try ensurepip instead")
                    # Try to use ensurepip as fallback
                    returncode, output = run_command(f"{python_exe} -m ensurepip --upgrade")
                    if returncode != 0:
                        print_warning(f"Failed to install pip with ensurepip: {output}")
                        print_warning("Will attempt to use 'python -m pip' instead")
                        pip_command = f"{python_exe} -m pip"
                    else:
                        print_success("Pip installed successfully with ensurepip")
                        pip_command = f"{pip_exe}"
            else:
                # Fallback to ensurepip
                returncode, output = run_command(f"{python_exe} -m ensurepip --upgrade")
                if returncode != 0:
                    print_warning(f"Failed to install pip with ensurepip: {output}")
                    print_warning("Will attempt to use 'python -m pip' instead")
                    pip_command = f"{python_exe} -m pip"
                else:
                    print_success("Pip installed successfully with ensurepip")
                    pip_command = f"{pip_exe}"
        except Exception as e:
            print_warning(f"Error installing pip: {e}")
            pip_command = f"{python_exe} -m pip"
    else:
        pip_command = f"{pip_exe}"
    
    # Upgrade pip first
    print_step("Upgrading pip")
    returncode, output = run_command(f"{pip_command} install --upgrade pip")
    if returncode != 0:
        print_warning(f"Failed to upgrade pip: {output}")
    
    # Install required packages for the setup
    print_step("Installing base packages")
    base_packages = ["wheel", "setuptools", "requests", "uvicorn", "fastapi", "pydantic", "loguru"]
    returncode, output = run_command(f"{pip_command} install {' '.join(base_packages)}")
    if returncode != 0:
        print_warning(f"Failed to install base packages: {output}")
    else:
        print_success("Base packages installed successfully")
    
    # Install requirements files
    for req_file in REQUIREMENTS_FILES:
        req_path = PROJECT_DIR / req_file
        if req_path.exists():
            print_step(f"Installing from {req_file}")
            returncode, output = run_command(f"{pip_command} install -r {req_path}")
            if returncode != 0:
                print_warning(f"Failed to install from {req_file}: {output}")
            else:
                print_success(f"Successfully installed from {req_file}")
        else:
            print_warning(f"Requirements file {req_file} not found")
            # Create an empty requirements file if it doesn't exist
            try:
                with open(req_path, "w") as f:
                    f.write("# Requirements file\n# Add your dependencies here\n")
                print_success(f"Created empty {req_file}")
            except Exception as e:
                print_warning(f"Failed to create {req_file}: {e}")
    
    # Create setup.py if it doesn't exist
    setup_py = PROJECT_DIR / "setup.py"
    if not setup_py.exists():
        print_warning("setup.py not found, creating a minimal one")
        try:
            with open(setup_py, "w") as f:
                f.write("""\
from setuptools import setup, find_packages

setup(
    name="one_ring",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic",
        "loguru",
    ],
)
""")
            print_success("Created minimal setup.py")
        except Exception as e:
            print_warning(f"Failed to create setup.py: {e}")
    
    # Install the package in development mode
    print_step("Installing One Ring package in development mode")
    returncode, output = run_command(f"{pip_command} install -e .")
    if returncode != 0:
        print_warning(f"Failed to install One Ring package: {output}")
    else:
        print_success("Successfully installed One Ring package")

def ensure_ollama_running():
    """Check if Ollama is installed and running. If installed but not running, start it.
    
    In the realms of Middle Earth, the great forge must be lit before the One Ring can be crafted.
    """
    print_step("Checking Ollama installation and server status")
    
    # Check if ollama is installed
    returncode, output = run_command("ollama --version")
    if returncode != 0:
        print_error("Ollama is not installed. Please install it from https://ollama.com/download", exit_code=1)
        return False
    else:
        print_success(f"Ollama is installed: {output.strip()}")
    
    # Check if ollama server is running
    returncode, output = run_command("ollama list")
    if returncode != 0:
        print_warning("Ollama server is not running. Attempting to start it automatically...")
        
        # Start Ollama server based on OS
        if platform.system() == "Windows":
            # On Windows, start Ollama in a new detached process
            try:
                # Using subprocess.Popen to start a detached process
                startup_info = subprocess.STARTUPINFO()
                startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startup_info.wShowWindow = subprocess.SW_HIDE
                
                # Start ollama serve in a separate process
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                    startupinfo=startup_info,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
                
                # Wait for the server to start up
                print_step("Waiting for Ollama server to start...")
                max_retries = 5
                for i in range(max_retries):
                    time.sleep(2)  # Give it some time to start
                    returncode, output = run_command("ollama list")
                    if returncode == 0:
                        print_success("Ollama server started successfully")
                        return True
                    if i < max_retries - 1:
                        print_step(f"Still waiting... ({i+1}/{max_retries})")
                
                print_error("Failed to start Ollama server automatically. Please start it manually with 'ollama serve' in a separate terminal")
                return False
                
            except Exception as e:
                print_error(f"Error starting Ollama server: {e}")
                print_warning("Please start it manually with 'ollama serve' in a separate terminal")
                return False
        else:
            # On Unix systems
            print_warning("On Unix systems, please start Ollama server manually with 'ollama serve' in a separate terminal")
            print_error("Ollama server is not running", exit_code=1)
            return False
    else:
        print_success("Ollama server is running")
        return True

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
    
    # Set environment variables
    os.environ["PYTHONPATH"] = str(PROJECT_DIR)
    
    # Check if the one_ring package is installed
    try:
        import one_ring
    except ImportError:
        print_warning("One Ring package not installed. Installing it now...")
        
        # Use the same pip command as in install_dependencies
        pip_exe = VENV_DIR / "Scripts" / "pip.exe" if platform.system() == "Windows" else VENV_DIR / "bin" / "pip"
        python_exe = VENV_DIR / "Scripts" / "python.exe" if platform.system() == "Windows" else VENV_DIR / "bin" / "python"
        
        if pip_exe.exists():
            pip_command = f"{pip_exe}"
        else:
            pip_command = f"{python_exe} -m pip"
        
        returncode, output = run_command(f"{pip_command} install -e .")
        if returncode != 0:
            print_error(f"Failed to install One Ring package: {output}")
        else:
            print_success("Successfully installed One Ring package")
    
    # Check if uvicorn is available
    try:
        subprocess.check_output([sys.executable, "-m", "pip", "show", "uvicorn"])
        uvicorn_installed = True
    except subprocess.CalledProcessError:
        uvicorn_installed = False
    
    if not uvicorn_installed:
        print_warning("Uvicorn not found. Installing it now...")
        
        # Use the same pip command as before
        pip_exe = VENV_DIR / "Scripts" / "pip.exe" if platform.system() == "Windows" else VENV_DIR / "bin" / "pip"
        python_exe = VENV_DIR / "Scripts" / "python.exe" if platform.system() == "Windows" else VENV_DIR / "bin" / "python"
        
        if pip_exe.exists():
            pip_command = f"{pip_exe}"
        else:
            pip_command = f"{python_exe} -m pip"
        
        returncode, output = run_command(f"{pip_command} install uvicorn")
        if returncode != 0:
            print_warning(f"Failed to install uvicorn with pip: {output}")
            print_step("Trying alternative installation method")
            returncode, output = run_command(f"{sys.executable} -m pip install uvicorn")
            if returncode != 0:
                print_error(f"Failed to install uvicorn: {output}")
        else:
            print_success("Successfully installed uvicorn")
    
    # Set environment variables
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_DIR)
    
    print(f"\n{Colors.HEADER}{' Starting One Ring AI Platform ':=^60}{Colors.ENDC}")
    print(f"{Colors.BOLD}Server URL:{Colors.ENDC} http://{host}:{port}")
    print(f"{Colors.BOLD}API Docs:{Colors.ENDC} http://{host}:{port}/docs")
    print(f"{Colors.BOLD}Exit:{Colors.ENDC} Press Ctrl+C to stop the server\n")
    
    # Make sure the one_ring directory exists
    one_ring_dir = PROJECT_DIR / "one_ring"
    if not one_ring_dir.exists() or not (one_ring_dir / "__init__.py").exists():
        print_warning("one_ring package directory not found or incomplete")
        print_warning("Creating minimal package structure...")
        create_minimal_package_structure()
    
    try:
        # Start the FastAPI server
        subprocess.run(
            [
                str(python_exe), "-m", "uvicorn",
                "one_ring.api.app:create_app",
                "--host", host,
                "--port", str(port),
                "--reload"
            ],
            cwd=PROJECT_DIR,
            env=env
        )
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print_error(f"Failed to start application: {e}")

def create_minimal_package_structure():
    """Create a minimal package structure for the application."""
    print_step("Creating minimal package structure")
    
    # Directory structure to create
    directories = [
        "one_ring",
        "one_ring/api",
        "one_ring/api/v1",
        "one_ring/api/v1/endpoints",
        "one_ring/core",
        "one_ring/services",
        "one_ring/data",
        "one_ring/training",
        "one_ring/utils",
    ]
    
    # Create directories
    for directory in directories:
        dir_path = PROJECT_DIR / directory
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print_success(f"Created directory: {directory}")
            except Exception as e:
                print_warning(f"Failed to create directory {directory}: {e}")
    
    # Define and create files individually to avoid syntax issues with triple quotes
    init_content = (
        "# One Ring AI Platform - A powerful tool for fine-tuning language models.\n\n"
        "__version__ = \"0.1.0\""
    )
    
    empty_init = ""
    
    app_content = (
        "from fastapi import FastAPI, Request, Response\n"
        "from fastapi.middleware.cors import CORSMiddleware\n"
        "from fastapi.responses import JSONResponse\n\n"
        "def create_app():\n"
        "    \"\"\"Create and configure the FastAPI application.\"\"\"\n"
        "    app = FastAPI(\n"
        "        title=\"One Ring AI Platform\",\n"
        "        description=\"A powerful tool for fine-tuning language models.\",\n"
        "        version=\"0.1.0\",\n"
        "    )\n\n"
        "    # Configure CORS\n"
        "    app.add_middleware(\n"
        "        CORSMiddleware,\n"
        "        allow_origins=[\"*\"],\n"
        "        allow_credentials=True,\n"
        "        allow_methods=[\"*\"],\n"
        "        allow_headers=[\"*\"],\n"
        "    )\n\n"
        "    # Health check endpoint\n"
        "    @app.get(\"/health\")\n"
        "    def health_check():\n"
        "        return {\"status\": \"ok\"}\n\n"
        "    # Root endpoint\n"
        "    @app.get(\"/\")\n"
        "    def root():\n"
        "        return {\"message\": \"Welcome to the One Ring AI Platform\"}\n\n"
        "    return app"
    )
    
    config_content = (
        "from pathlib import Path\n"
        "from typing import List, Optional\n\n"
        "from pydantic import BaseSettings\n\n"
        "class Settings(BaseSettings):\n"
        "    \"\"\"Application settings.\"\"\"\n"
        "    # API settings\n"
        "    API_HOST: str = \"127.0.0.1\"\n"
        "    API_PORT: int = 8000\n\n"
        "    # Directory settings\n"
        "    DATA_DIR: Path = Path(\"data\")\n"
        "    MODEL_SAVE_DIR: Path = Path(\"models\")\n\n"
        "    class Config:\n"
        "        env_prefix = \"ONE_RING_\"\n\n"
        "# Create global settings instance\n"
        "settings = Settings()"
    )
    
    # Map file paths to content
    file_contents = {
        "one_ring/__init__.py": init_content,
        "one_ring/api/__init__.py": empty_init,
        "one_ring/api/v1/__init__.py": empty_init,
        "one_ring/api/v1/endpoints/__init__.py": empty_init,
        "one_ring/core/__init__.py": empty_init,
        "one_ring/services/__init__.py": empty_init,
        "one_ring/data/__init__.py": empty_init,
        "one_ring/training/__init__.py": empty_init,
        "one_ring/utils/__init__.py": empty_init,
        "one_ring/api/app.py": app_content,
        "one_ring/core/config.py": config_content,
    }
    
    # Create files
    for file_path, content in file_contents.items():
        path = PROJECT_DIR / file_path
        if not path.exists():
            try:
                with open(path, "w") as f:
                    f.write(content)
                print_success(f"Created file: {file_path}")
            except Exception as e:
                print_warning(f"Failed to create file {file_path}: {e}")


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
        "--force-venv",
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
        "--debug",
        action="store_true",
        help="Enable debug mode with more verbose output"
    )
    parser.add_argument(
        "--no-ollama-check",
        action="store_true",
        help="Skip checking and starting Ollama server"
    )
    args = parser.parse_args()
    
    # Print header
    print_header()
    
    # Check Python version (strictly enforce Python 3.11)
    check_python_version()
    
    # Create virtual environment if not disabled
    if not args.no_venv:
        create_virtualenv(force=args.force_venv)
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
    
    # Create package structure if needed
    one_ring_dir = PROJECT_DIR / "one_ring"
    if not one_ring_dir.exists() or not (one_ring_dir / "__init__.py").exists():
        create_minimal_package_structure()
    
    # Ensure Ollama is running (unless explicitly skipped)
    if not args.no_ollama_check:
        ensure_ollama_running()
    
    # Start application
    start_application(args.host, args.port)

if __name__ == "__main__":
    main()
