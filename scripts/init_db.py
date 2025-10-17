#!/usr/bin/env python3
"""Initialize the database"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.session import get_db
from loguru import logger


def main():
    """Initialize database tables"""
    logger.info("Initializing database...")
    
    db = get_db()
    db.create_tables()
    
    logger.info("✅ Database initialized successfully")


if __name__ == "__main__":
    main()












