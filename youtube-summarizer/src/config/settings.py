"""
Configuration settings for the YouTube Video Summarizer application.
"""

import os
import torch
from dotenv import load_dotenv
import shutil
import platform

# Load environment variables from .env file
load_dotenv()

# Model settings
LOCAL_MODEL_NAME = os.environ.get("LOCAL_MODEL_NAME", "google/flan-t5-large")  # Better with long texts than BART

# Whisper settings
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")  # tiny, base, small, medium, large, large-v2, large-v3

# Device settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
TEMP_DIR = os.environ.get("TEMP_DIR", "temp")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")

# FFmpeg settings
# First try to get from environment variable
FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "")

# If not set in environment, try to find in PATH
if not FFMPEG_PATH or not os.path.exists(FFMPEG_PATH):
    ffmpeg_binary = shutil.which("ffmpeg")
    if ffmpeg_binary:
        FFMPEG_PATH = os.path.dirname(ffmpeg_binary)
    else:
        # Try to find in common locations based on the operating system
        if platform.system() == "Windows":
            # Common locations on Windows
            paths_to_check = [
                # WinGet installation
                os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Microsoft', 'WinGet', 'Packages'),
                # Common program locations
                r"C:\Program Files\ffmpeg\bin",
                r"C:\ffmpeg\bin",
                # Chocolatey
                r"C:\ProgramData\chocolatey\bin"
            ]
            
            # Check WinGet packages directory recursively
            winget_path = paths_to_check[0]
            if os.path.exists(winget_path):
                for root, dirs, files in os.walk(winget_path):
                    if "ffmpeg.exe" in files:
                        FFMPEG_PATH = os.path.dirname(os.path.join(root, "ffmpeg.exe"))
                        break
            
            # Check other common paths
            if not FFMPEG_PATH:
                for path in paths_to_check[1:]:
                    if os.path.exists(os.path.join(path, "ffmpeg.exe")):
                        FFMPEG_PATH = path
                        break
        
        elif platform.system() == "Darwin":  # macOS
            # Homebrew, MacPorts
            paths_to_check = [
                "/usr/local/bin",
                "/opt/local/bin",
                "/opt/homebrew/bin"
            ]
            
            for path in paths_to_check:
                if os.path.exists(os.path.join(path, "ffmpeg")):
                    FFMPEG_PATH = path
                    break
        
        else:  # Linux and others
            # Common Linux paths
            paths_to_check = [
                "/usr/bin",
                "/usr/local/bin",
                "/opt/ffmpeg/bin"
            ]
            
            for path in paths_to_check:
                if os.path.exists(os.path.join(path, "ffmpeg")):
                    FFMPEG_PATH = path
                    break
                    
        # Default to empty string if not found
        if not FFMPEG_PATH:
            FFMPEG_PATH = ""

# Create directories if they don't exist
for directory in [TEMP_DIR, OUTPUT_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)