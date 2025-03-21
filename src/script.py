"""Imports necessary modules"""

import os
import subprocess
import sys
from pathlib import Path

"""Ensure the script works cross-platform (Windows and Ubuntu)"""


def get_project_root():
    """Get the root directory of the project"""
    return Path(__file__).resolve().parent.parent


def add_src_to_path():
    """Add the src directory to the Python path"""
    src_path = get_project_root() / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def run_python_file(file_name):
    """Run a Python file using subprocess"""
    file_path = get_project_root() / "src" / file_name
    if not file_path.exists():
        print(f"Error: {file_name} not found in src directory!")
        sys.exit(1)

    try:
        """Use the current Python interpreter to run the file"""
        result = subprocess.run([sys.executable, str(file_path)], check=True, text=True, capture_output=True)
        print(f"Output from {file_name}:\n{result.stdout}")
        if result.stderr:
            print(f"Errors from {file_name}:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to run {file_name}: {e}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)


def main():
    """Main function to orchestrate the execution of other scripts"""
    print("Starting the automation script...")

    """Add src to the Python path so we can import other modules"""
    add_src_to_path()

    """List of scripts to run in order"""
    scripts_to_run = ["main.py"]

    for script in scripts_to_run:
        print(f"\nRunning {script}...")
        run_python_file(script)

    print("\nAll scripts executed successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
