#!/usr/bin/env python3
"""
Database migration script for Phase 1 analysis tables
Creates tables for technical indicators, market regimes, and pattern detection
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from src.database.session import get_engine
from src.database.models import Base, TechnicalIndicators, MarketRegime, PatternDetection


def main():
    """Create Phase 1 analysis tables"""
    logger.info("Starting Phase 1 database migration...")
    
    try:
        # Get database engine
        engine = get_engine()
        
        # Create all tables
        logger.info("Creating analysis tables...")
        Base.metadata.create_all(engine, tables=[
            TechnicalIndicators.__table__,
            MarketRegime.__table__,
            PatternDetection.__table__
        ])
        
        logger.info("✅ Phase 1 tables created successfully!")
        logger.info("Tables created:")
        logger.info("  - technical_indicators")
        logger.info("  - market_regimes")
        logger.info("  - pattern_detections")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

