"""Compatibility entry point for the corrected benchmark.

Use this file or scripts/run_corrected_benchmark.py. The earlier exploratory
pipeline is intentionally no longer executed because it used invalid splits.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_corrected_benchmark import main


if __name__ == "__main__":
    main()
