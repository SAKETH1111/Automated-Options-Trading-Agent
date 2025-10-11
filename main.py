#!/usr/bin/env python3
"""
Automated Options Trading Agent - Main Entry Point
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.orchestrator import main

if __name__ == "__main__":
    main()





