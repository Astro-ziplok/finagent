"""
tests/run_tests.py — Run All Tests
====================================
Discovers and runs all unit tests in the tests/ folder.

Usage:
    python tests/run_tests.py          # Run all tests
    python -m pytest tests/ -v         # Run with pytest (more detailed output)
    python -m pytest tests/ -v -k cleaning   # Run only cleaning tests
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == "__main__":
    # Discover all test files matching test_*.py in this folder
    loader = unittest.TestLoader()
    suite  = loader.discover(start_dir=str(Path(__file__).parent),
                             pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with non-zero code if any test failed (useful for CI)
    sys.exit(0 if result.wasSuccessful() else 1)