"""A simple cross-platform Hello World program."""

import os
import platform


def print_hello():
    """Print a simple Hello World message."""
    print(
        "Hello, World! This is a very long line that might need to be reformatted by black depending on the line length setting."
    )


def main():
    """Main entry point of the program."""
    print_hello()


if __name__ == "__main__":
    main()
