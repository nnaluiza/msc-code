"""Imports necessary modules"""

import importlib
import sys
from pathlib import Path


def run_script():
    """Dynamically imports and runs the main.py script from the src directory."""
    script_dir = Path(__file__).resolve().parent
    src_path = script_dir / "src"

    sys.path.insert(0, str(src_path))

    try:
        main_module = importlib.import_module("main")
        main_module.main(sys.argv[1:])
    except ImportError as e:
        print(f"Error: Could not import main.py: {e}")
        sys.exit(1)
    except AttributeError as e:
        print(f"Error: main.py must have a main() function: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error while running main.py: {e}")
        sys.exit(1)


if __name__ == "__main__":
    """Entry point of the script. Checks for command-line arguments and initiates the script execution."""
    if len(sys.argv) < 2:
        print("Usage: python run.py [param1] [param2] ...")
        sys.exit(1)

    run_script()
