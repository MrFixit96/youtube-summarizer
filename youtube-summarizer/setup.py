#!/usr/bin/env python3
"""
Setup script for YouTube Summarizer.
This script:
1. Installs required Python packages using UV (preferred) or pip
2. Checks for and installs FFmpeg if needed
3. Adds FFmpeg to PATH
4. Sets up environment configuration
"""

import os
import sys
import platform
import subprocess
import shutil
import zipfile
import tempfile
from pathlib import Path
import urllib.request

def is_admin():
    """Check if script is running with admin/root privileges."""
    try:
        if platform.system() == 'Windows':
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        else:  # Unix-based systems
            return os.geteuid() == 0
    except:
        return False

def print_colored(text, color="green"):
    """Print colored text."""
    colors = {
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "blue": "\033[94m",
        "reset": "\033[0m"
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}")

def run_command(command, shell=False, check=True, env=None, silent=False):
    """Run a shell command and return output."""
    if not silent:
        print_colored(f"Running: {command if isinstance(command, str) else ' '.join(command)}", "blue")
    
    try:
        result = subprocess.run(
            command,
            shell=shell,
            check=check,
            text=True,
            capture_output=True,
            env=env
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        if not silent:
            print_colored(f"Command failed with error: {e}", "red")
            print_colored(f"Error output: {e.stderr}", "red")
        raise

def check_python_version():
    """Check if Python version is 3.8 or higher."""
    if sys.version_info < (3, 8):
        print_colored("Python 3.8 or higher is required.", "red")
        sys.exit(1)
    else:
        print_colored(f"Python version: {sys.version.split()[0]} (OK)", "green")

def check_uv():
    """Check if UV package manager is installed."""
    try:
        subprocess.check_call(["uv", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print_colored("UV package manager is installed (OK)", "green")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_colored("UV package manager not found", "yellow")
        return False

def check_pip():
    """Check if pip is available."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print_colored("Pip is available (OK)", "green")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_colored("Pip not found", "yellow")
        return False

def install_uv_with_winget():
    """Install UV package manager using winget on Windows."""
    if platform.system() != 'Windows':
        print_colored("Winget is only available on Windows. Cannot install UV with winget.", "yellow")
        return False
    
    print_colored("Installing UV package manager using winget...", "blue")
    try:
        run_command(["winget", "install", "astral-sh.uv"], check=False)
        # Check if UV was installed successfully
        if check_uv():
            print_colored("UV installed successfully with winget (OK)", "green")
            return True
        else:
            print_colored("UV installation with winget may have failed. Will try alternative methods.", "yellow")
            return False
    except subprocess.CalledProcessError:
        print_colored("Failed to install UV with winget.", "yellow")
        return False

def install_uv():
    """Install UV package manager."""
    # First try with winget on Windows
    if platform.system() == 'Windows':
        if install_uv_with_winget():
            return True
    
    # Fallback to pip installation
    print_colored("Installing UV package manager using pip...", "blue")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "uv"], stdout=subprocess.PIPE)
        print_colored("UV installed successfully with pip (OK)", "green")
        return True
    except subprocess.CalledProcessError:
        print_colored("Failed to install UV. Will use pip instead.", "yellow")
        return False

def setup_virtual_env(python_version=None):
    """Set up a virtual environment using UV or venv."""
    print_colored("Setting up virtual environment...", "blue")
    
    venv_dir = ".venv"
    
    # Check if virtual environment already exists
    if os.path.exists(venv_dir):
        print_colored(f"Virtual environment already exists at {venv_dir} (OK)", "green")
        return True
    
    # Try to use UV for virtual environment creation
    if check_uv():
        try:
            python_arg = f"--python={python_version}" if python_version else ""
            command = f"uv venv {python_arg} {venv_dir}"
            run_command(command, shell=True)
            print_colored(f"Virtual environment created using UV at {venv_dir} (OK)", "green")
            
            # Make sure pip is available in the virtual environment
            if platform.system() == "Windows":
                pip_command = [os.path.join(venv_dir, "Scripts", "python.exe"), "-m", "ensurepip", "--upgrade"]
            else:
                pip_command = [os.path.join(venv_dir, "bin", "python"), "-m", "ensurepip", "--upgrade"]
            
            try:
                print_colored("Ensuring pip is available in the virtual environment...", "blue")
                run_command(pip_command)
                print_colored("Pip is now available in the virtual environment (OK)", "green")
            except subprocess.CalledProcessError as e:
                print_colored(f"Warning: Failed to ensure pip is available: {e}", "yellow")
                print_colored("Some installation methods may not work.", "yellow")
            
            return True
        except subprocess.CalledProcessError:
            print_colored("Failed to create virtual environment with UV.", "yellow")
    
    # Fallback to venv
    print_colored("Falling back to standard venv module...", "yellow")
    try:
        run_command([sys.executable, "-m", "venv", venv_dir])
        print_colored(f"Virtual environment created using venv at {venv_dir} (OK)", "green")
        return True
    except subprocess.CalledProcessError:
        print_colored("Failed to create virtual environment.", "red")
        return False

def install_dependencies(venv_dir=".venv"):
    """Install required Python packages."""
    print_colored("Installing Python dependencies...", "blue")
    
    # Get path to the virtual environment's Python executable and binaries
    if platform.system() == "Windows":
        python_exec = os.path.join(venv_dir, "Scripts", "python.exe")
        pip_exec = os.path.join(venv_dir, "Scripts", "pip.exe")
        venv_bin_dir = os.path.join(venv_dir, "Scripts")
    else:
        python_exec = os.path.join(venv_dir, "bin", "python")
        pip_exec = os.path.join(venv_dir, "bin", "pip")
        venv_bin_dir = os.path.join(venv_dir, "bin")
    
    # Check if the virtual environment exists
    if not os.path.exists(python_exec):
        print_colored(f"Virtual environment not found at {venv_dir}", "red")
        return False
    
    # Check if we have global UV available
    global_uv_available = check_uv()
    
    if global_uv_available:
        # Use the global UV to install dependencies directly
        print_colored("Using global UV for faster package installation (OK)", "green")
        try:
            # Use global UV but specify the virtual environment
            install_command = ["uv", "pip", "install", "--python", python_exec, "-r", "requirements.txt"]
            run_command(install_command)
            print_colored("Python dependencies installed successfully with UV (OK)", "green")
            return True
        except subprocess.CalledProcessError as e:
            print_colored(f"UV installation failed: {e}", "yellow")
            print_colored("Falling back to alternative methods", "yellow")
    
    # If global UV didn't work, try using the virtual environment's python to install 
    try:
        # Try installing with the virtual env's python
        print_colored("Using virtualenv's python to install dependencies", "blue")
        install_command = [python_exec, "-m", "pip", "install", "-r", "requirements.txt"]
        run_command(install_command)
        print_colored("Python dependencies installed successfully (OK)", "green")
        return True
    except subprocess.CalledProcessError as e:
        print_colored(f"Failed to install dependencies using virtualenv python: {e}", "red")
        return False

def install_dependencies_with_uv(venv_dir=".venv"):
    """Install required Python packages using UV."""
    print_colored("Installing Python dependencies with UV...", "blue")
    
    # Get path to the virtual environment's Python executable
    if platform.system() == "Windows":
        python_exec = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        python_exec = os.path.join(venv_dir, "bin", "python")
    
    # Check if the virtual environment exists
    if not os.path.exists(python_exec):
        print_colored(f"Virtual environment not found at {venv_dir}", "red")
        return False
    
    try:
        # Use UV directly (it's faster and handles dependencies better)
        install_command = ["uv", "pip", "install", "--python", python_exec, "-r", "requirements.txt"]
        run_command(install_command)
        print_colored("Python dependencies installed successfully with UV (OK)", "green")
        return True
    except subprocess.CalledProcessError as e:
        print_colored(f"UV installation failed: {e}", "yellow")
        print_colored("Falling back to regular pip installation...", "yellow")
        return install_dependencies(venv_dir)

def check_ffmpeg():
    """Check if FFmpeg is installed and accessible."""
    try:
        output = subprocess.check_output(["ffmpeg", "-version"], stderr=subprocess.STDOUT, text=True)
        print_colored(f"FFmpeg is installed (OK) ({output.split()[2]})", "green")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_colored("FFmpeg not found in PATH", "yellow")
        return False

def find_ffmpeg_in_winget():
    """Find FFmpeg installed via winget."""
    try:
        output = run_command(["winget", "list", "--name", "ffmpeg"], silent=True)
        if "FFmpeg" in output:
            print_colored("FFmpeg is installed via winget", "green")
            # Try to find the exact path
            try:
                # Get user's AppData Local path
                localappdata = os.environ.get("LOCALAPPDATA")
                winget_packages = os.path.join(localappdata, "Microsoft", "WinGet", "Packages")
                
                for root, dirs, files in os.walk(winget_packages):
                    if "ffmpeg.exe" in files:
                        ffmpeg_path = os.path.join(root, "ffmpeg.exe")
                        print_colored(f"Found FFmpeg at: {ffmpeg_path}", "green")
                        return ffmpeg_path
            except Exception as e:
                print_colored(f"Error finding FFmpeg in winget: {e}", "yellow")
            
            return "winget"  # Return a marker that it's installed but we couldn't find the path
        return None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def download_ffmpeg_windows():
    """Download FFmpeg for Windows."""
    print_colored("Downloading FFmpeg...", "blue")
    ffmpeg_url = "https://github.com/GyanD/codexffmpeg/releases/download/7.1.1/ffmpeg-7.1.1-full_build.zip"
    temp_dir = tempfile.gettempdir()
    zip_path = os.path.join(temp_dir, "ffmpeg.zip")
    
    try:
        # Download the zip file
        print_colored(f"Downloading from {ffmpeg_url}...", "blue")
        urllib.request.urlretrieve(ffmpeg_url, zip_path)
        
        # Create target directory in the app folder
        ffmpeg_dir = Path.home() / ".youtube_summarizer" / "ffmpeg"
        ffmpeg_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract the zip file
        print_colored("Extracting FFmpeg...", "blue")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # The extraction creates a directory like ffmpeg-7.1.1-full_build
        extracted_dir = next(Path(temp_dir).glob("ffmpeg-*-full_build"))
        
        # Move the bin directory to our target location
        bin_path = extracted_dir / "bin"
        for file in bin_path.glob("*"):
            shutil.copy(file, ffmpeg_dir)
        
        # Clean up
        shutil.rmtree(extracted_dir)
        os.remove(zip_path)
        
        print_colored(f"FFmpeg installed to {ffmpeg_dir}", "green")
        return str(ffmpeg_dir)
    except Exception as e:
        print_colored(f"Failed to download and extract FFmpeg: {e}", "red")
        return None

def install_ffmpeg_linux():
    """Install FFmpeg on Linux."""
    print_colored("Installing FFmpeg using apt...", "blue")
    try:
        if not is_admin():
            print_colored("Administrator privileges required to install FFmpeg.", "red")
            print_colored("Please run: sudo apt-get install ffmpeg", "yellow")
            return None
        
        run_command(["apt-get", "update"])
        run_command(["apt-get", "install", "-y", "ffmpeg"])
        print_colored("FFmpeg installed successfully (OK)", "green")
        return True
    except subprocess.CalledProcessError:
        print_colored("Failed to install FFmpeg.", "red")
        return None

def install_ffmpeg_mac():
    """Install FFmpeg on macOS."""
    print_colored("Installing FFmpeg using brew...", "blue")
    try:
        run_command(["brew", "install", "ffmpeg"])
        print_colored("FFmpeg installed successfully (OK)", "green")
        return True
    except subprocess.CalledProcessError:
        print_colored("Failed to install FFmpeg.", "red")
        print_colored("Please install Homebrew (https://brew.sh/) and run: brew install ffmpeg", "yellow")
        return None

def install_ffmpeg():
    """Install FFmpeg based on the operating system."""
    system = platform.system()
    
    if system == "Windows":
        # First check if it's installed via winget
        ffmpeg_path = find_ffmpeg_in_winget()
        if ffmpeg_path:
            if ffmpeg_path != "winget":  # We found the actual path
                return ffmpeg_path
            else:
                print_colored("FFmpeg is installed via winget but path couldn't be determined.", "yellow")
                print_colored("You may need to manually add it to your PATH", "yellow")
                return None
        
        # If not, download and install it
        return download_ffmpeg_windows()
    elif system == "Linux":
        return install_ffmpeg_linux() and shutil.which("ffmpeg")
    elif system == "Darwin":  # macOS
        return install_ffmpeg_mac() and shutil.which("ffmpeg")
    else:
        print_colored(f"Unsupported operating system: {system}", "red")
        return None

def update_path_env_var(new_path):
    """Update PATH environment variable."""
    if not new_path or not os.path.exists(new_path):
        return False
    
    system = platform.system()
    
    # First update the path for the current process
    if os.path.isfile(new_path):
        # If new_path is a file (like ffmpeg.exe), use its directory
        new_path = os.path.dirname(new_path)
    
    print_colored(f"Adding {new_path} to current process PATH", "blue")
    os.environ["PATH"] = f"{new_path}{os.pathsep}{os.environ.get('PATH', '')}"
    
    # For demonstration, try to run ffmpeg after updating PATH
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        print_colored("FFmpeg is now accessible in the current process (OK)", "green")
    except Exception:
        print_colored("FFmpeg still not accessible in current process", "yellow")
        
    # Now update for persistence
    if system == "Windows":
        # For current user (persistent)
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment", 0, winreg.KEY_ALL_ACCESS) as key:
                current_path, _ = winreg.QueryValueEx(key, "Path")
                if new_path not in current_path:
                    new_path_value = f"{current_path};{new_path}"
                    winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new_path_value)
                    print_colored(f"Added {new_path} to user PATH environment variable (OK)", "green")
                    # Notify the system about the change
                    subprocess.run(["setx", "PATH", new_path_value])
                    return True
                else:
                    print_colored(f"{new_path} is already in PATH (OK)", "green")
                    return True
        except Exception as e:
            print_colored(f"Failed to update PATH environment variable: {e}", "red")
            print_colored(f"Please manually add {new_path} to your PATH environment variable", "yellow")
            return False
    else:
        # For Unix-based systems, update .bashrc or .bash_profile
        shell_profile = os.path.expanduser("~/.bashrc")
        if system == "Darwin" and os.path.exists(os.path.expanduser("~/.bash_profile")):
            shell_profile = os.path.expanduser("~/.bash_profile")
        
        try:
            with open(shell_profile, "r") as f:
                content = f.read()
            
            if f"export PATH={new_path}:$PATH" not in content:
                with open(shell_profile, "a") as f:
                    f.write(f"\n# Added by YouTube Summarizer setup\nexport PATH={new_path}:$PATH\n")
                
                print_colored(f"Added {new_path} to PATH in {shell_profile} (OK)", "green")
                print_colored(f"Please run: source {shell_profile} or restart your terminal", "yellow")
                return True
            else:
                print_colored(f"{new_path} is already in PATH (OK)", "green")
                return True
        except Exception as e:
            print_colored(f"Failed to update PATH in {shell_profile}: {e}", "red")
            print_colored(f"Please manually add the following line to {shell_profile}:", "yellow")
            print_colored(f"export PATH={new_path}:$PATH", "yellow")
            return False

def create_env_file():
    """Create a .env file with necessary configuration."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    
    if os.path.exists(env_path):
        print_colored(".env file already exists (OK)", "green")
        return
    
    system = platform.system()
    ffmpeg_path = ""
    
    if system == "Windows":
        if hasattr(sys, "app_ffmpeg_path") and sys.app_ffmpeg_path:
            ffmpeg_path = sys.app_ffmpeg_path
        else:
            # Try to find FFmpeg in PATH
            ffmpeg_binary = shutil.which("ffmpeg")
            if ffmpeg_binary:
                ffmpeg_path = os.path.dirname(ffmpeg_binary)
    
    env_content = f"""# YouTube Summarizer Configuration
# Created by setup.py

# Model settings
WHISPER_MODEL=large-v3  # tiny, base, small, medium, large, large-v2, large-v3
LOCAL_MODEL_NAME=google/flan-t5-large

# Directory settings
TEMP_DIR=temp
OUTPUT_DIR=output

# FFmpeg settings
FFMPEG_PATH={ffmpeg_path}
"""
    
    with open(env_path, "w") as f:
        f.write(env_content)
    
    print_colored(f".env file created at {env_path} (OK)", "green")

def main():
    """Main setup function."""
    print_colored("Setting up YouTube Summarizer...", "blue")
    
    # Store the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Step 1: Check Python version
    check_python_version()
    
    # Step 2: Check and install UV package manager
    uv_installed = check_uv()
    if not uv_installed:
        print_colored("UV package manager not found. Installing with winget...", "blue")
        uv_installed = install_uv_with_winget()
        if not uv_installed and check_pip():
            print_colored("Trying to install UV via pip as fallback...", "blue")
            uv_installed = install_uv()
    
    # Step 3: Set up virtual environment
    python_version = "3.12"  # Target specific Python version
    setup_virtual_env(python_version)
    
    # Step 4: Check for FFmpeg before installing Python dependencies
    # as some Python packages might need FFmpeg during installation
    if not check_ffmpeg():
        print_colored("FFmpeg is not in PATH. Installing...", "blue")
        ffmpeg_path = install_ffmpeg()
        if ffmpeg_path:
            # Store the path for later
            sys.app_ffmpeg_path = ffmpeg_path
            # Update PATH
            update_path_env_var(ffmpeg_path)
            # Check again
            if check_ffmpeg():
                print_colored("FFmpeg is now properly installed and in PATH (OK)", "green")
            else:
                print_colored("FFmpeg is installed but still not in PATH.", "yellow")
                print_colored(f"Please manually add {ffmpeg_path} to your PATH environment variable", "yellow")
    
    # Step 5: Install dependencies using UV or pip
    if uv_installed:
        install_dependencies_with_uv()
    else:
        install_dependencies()
    
    # Step 6: Create .env file
    create_env_file()
    
    print_colored("\nSetup completed successfully!", "green")
    print_colored("You can now run the YouTube Summarizer application.", "green")
    print_colored("\nActivate the virtual environment:", "blue")
    if platform.system() == "Windows":
        print_colored(".venv\\Scripts\\activate", "blue")
    else:
        print_colored("source .venv/bin/activate", "blue")
    print_colored("\nRun the application:", "blue")
    print_colored("python src/main.py https://www.youtube.com/watch?v=YOUR_VIDEO_ID", "blue")

if __name__ == "__main__":
    main()