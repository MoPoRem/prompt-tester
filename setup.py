#!/usr/bin/env python3
"""Setup script for the AI Model Tester"""

import subprocess
import sys
import os

def setup_environment():
    """Set up the virtual environment and install dependencies."""
    print("Setting up AI Model Tester...")
    
    # Activate virtual environment and install requirements
    if os.name == 'nt':  # Windows
        pip_cmd = r"venv\Scripts\pip"
        python_cmd = r"venv\Scripts\python"
    else:  # Unix/Linux/Mac
        pip_cmd = "venv/bin/pip"
        python_cmd = "venv/bin/python"
    
    try:
        # Install requirements
        print("Installing dependencies...")
        subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], check=True)
        
        print("\nSetup complete!")
        print("\nNext steps:")
        print("1. Copy .env.example to .env")
        print("2. Add your API keys to the .env file")
        print("3. Run the tester:")
        if os.name == 'nt':
            print(f"   {python_cmd} model_tester.py")
        else:
            print(f"   {python_cmd} model_tester.py")
            
    except subprocess.CalledProcessError as e:
        print(f"Error during setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup_environment()