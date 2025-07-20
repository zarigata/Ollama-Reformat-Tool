import os
import sys
import subprocess
import signal
import time
from pathlib import Path
from typing import List, Optional
import webbrowser

class ProcessManager:
    def __init__(self):
        self.processes = []
        self.root_dir = Path(__file__).parent.absolute()
        self.backend_dir = self.root_dir / 'backend'
        self.frontend_dir = self.root_dir / 'frontend'
        
        # Ensure required directories exist
        (self.backend_dir / 'uploads').mkdir(exist_ok=True)
        (self.backend_dir / 'models').mkdir(exist_ok=True)
        
        # Environment variables
        self.env = os.environ.copy()
        self.env['PYTHONUNBUFFERED'] = '1'
        
    def run_command(self, cmd: List[str], cwd: Optional[Path] = None, shell: bool = False) -> subprocess.Popen:
        """Run a command and return the process object."""
        cwd = cwd or self.root_dir
        print(f"🚀 Starting: {' '.join(cmd)} in {cwd}")
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            shell=shell,
            env=self.env,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
        )
        self.processes.append(process)
        return process
    
    def setup_virtualenv(self) -> bool:
        """Set up Python virtual environment if it doesn't exist."""
        venv_dir = self.backend_dir / 'venv'
        if not venv_dir.exists():
            print("🔧 Setting up Python virtual environment...")
            try:
                # Create virtual environment
                self.run_command([sys.executable, '-m', 'venv', 'venv'], cwd=self.backend_dir).wait()
                
                # Get the correct pip path based on platform
                if sys.platform == 'win32':
                    pip_path = venv_dir / 'Scripts' / 'pip.exe'
                    python_path = venv_dir / 'Scripts' / 'python.exe'
                else:
                    pip_path = venv_dir / 'bin' / 'pip'
                    python_path = venv_dir / 'bin' / 'python'
                
                # Ensure pip is up to date
                self.run_command([str(python_path), '-m', 'pip', 'install', '--upgrade', 'pip']).wait()
                
                # Install requirements
                requirements_file = self.backend_dir / 'requirements.txt'
                if requirements_file.exists():
                    self.run_command([str(pip_path), 'install', '-r', str(requirements_file)]).wait()
                else:
                    print(f"⚠️ requirements.txt not found at {requirements_file}")
                
                return True
            except Exception as e:
                import traceback
                print(f"❌ Failed to set up virtual environment: {e}")
                print(traceback.format_exc())
                return False
        return True
    
    def start_backend(self) -> Optional[subprocess.Popen]:
        """Start the FastAPI backend server."""
        print("🚀 Starting backend server...")
        
        # Determine the correct Python path based on platform
        if sys.platform == 'win32':
            python_path = self.backend_dir / 'venv' / 'Scripts' / 'python.exe'
        else:
            python_path = self.backend_dir / 'venv' / 'bin' / 'python'
        
        # Convert to string for subprocess
        python_exec = str(python_path.resolve())
        
        if not python_path.exists():
            print(f"❌ Python executable not found at: {python_exec}")
            print("Please make sure the virtual environment was created successfully.")
            return None
        
        print(f"✅ Using Python at: {python_exec}")
        return self.run_command(
            [python_exec, '-m', 'uvicorn', 'main:app', '--host', '0.0.0.0', '--port', '8000'],
            cwd=self.backend_dir
        )
    
    def start_frontend(self) -> Optional[subprocess.Popen]:
        """Start the Next.js frontend development server."""
        print("🚀 Starting frontend server...")
        # Install frontend dependencies if node_modules doesn't exist
        if not (self.frontend_dir / 'node_modules').exists():
            print("🔧 Installing frontend dependencies...")
            self.run_command(['npm', 'install'], cwd=self.frontend_dir).wait()
        
        return self.run_command(['npm', 'run', 'dev'], cwd=self.frontend_dir)
    
    def open_browser(self):
        """Open the browser to the frontend."""
        print("🌐 Opening browser to http://localhost:3000")
        time.sleep(5)  # Give the servers time to start
        webbrowser.open('http://localhost:3000')
    
    def cleanup(self):
        """Clean up all running processes."""
        print("\n🛑 Cleaning up processes...")
        for process in self.processes:
            try:
                if sys.platform == 'win32':
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    process.terminate()
                process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    process.kill()
                except:
                    pass
        self.processes = []
    
    def run(self):
        """Run the application."""
        try:
            # Setup
            if not self.setup_virtualenv():
                return 1
            
            # Start services
            backend = self.start_backend()
            if not backend:
                return 1
                
            frontend = self.start_frontend()
            if not frontend:
                return 1
            
            # Open browser
            self.open_browser()
            
            # Keep running until interrupted
            print("\n✅ Application is running! Press Ctrl+C to stop.")
            print("Backend: http://localhost:8000")
            print("Frontend: http://localhost:3000")
            print("\n📝 OpenAPI documentation: http://localhost:8000/docs")
            
            # Wait for processes to complete
            for process in self.processes:
                process.wait()
                
        except KeyboardInterrupt:
            print("\n🛑 Received keyboard interrupt. Shutting down...")
            return 0
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            return 1
        finally:
            self.cleanup()
        
        return 0

if __name__ == "__main__":
    manager = ProcessManager()
    sys.exit(manager.run())
