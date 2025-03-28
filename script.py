"""Imports necessary modules"""

import importlib
import sys
from pathlib import Path


def run_script():
    """Dynamically imports and runs the main.py script from the src directory."""

    src_path = Path(__file__).resolve().parent / "src"
    sys.path.insert(0, str(src_path))

    try:
        main_module = importlib.import_module("main")
        main_module.main(sys.argv[1:])
    except ImportError:
        print("Error: Could not find main.py in the current directory!")
        sys.exit(1)
    except AttributeError:
        print("Error: main.py must have a main() function!")
        sys.exit(1)


if __name__ == "__main__":
    """Entry point of the script. Checks for command-line arguments and initiates the script execution."""

    if len(sys.argv) < 2:
        print("Usage: python run.py [param1] [param2] ...")
        sys.exit(1)

    run_script()
