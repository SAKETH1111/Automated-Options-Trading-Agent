#!/usr/bin/env python3
"""
Database migration script for Phase 2 options tables
Creates tables for options chains, IV metrics, opportunities, and unusual activity
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from src.database.session import get_engine
from src.database.models import (
    Base, OptionsChain, ImpliedVolatility, 
    OptionsOpportunity, UnusualOptionsActivity
)


def main():
    """Create Phase 2 options tables"""
    logger.info("Starting Phase 2 database migration...")
    
    try:
        # Get database engine
        engine = get_engine()
        
        # Create all tables
        logger.info("Creating options tables...")
        Base.metadata.create_all(engine, tables=[
            OptionsChain.__table__,
            ImpliedVolatility.__table__,
            OptionsOpportunity.__table__,
            UnusualOptionsActivity.__table__
        ])
        
        logger.info("✅ Phase 2 tables created successfully!")
        logger.info("Tables created:")
        logger.info("  - options_chains (25+ fields)")
        logger.info("  - implied_volatility (15+ fields)")
        logger.info("  - options_opportunities (25+ fields)")
        logger.info("  - unusual_options_activity (15+ fields)")
        
        logger.info("\n🎯 You can now:")
        logger.info("  - Collect options chains")
        logger.info("  - Calculate Greeks")
        logger.info("  - Track IV Rank and IV Percentile")
        logger.info("  - Find trading opportunities")
        logger.info("  - Detect unusual activity")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

