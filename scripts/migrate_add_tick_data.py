"""Database migration: Add IndexTickData table for real-time data collection"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from sqlalchemy import inspect

from src.database.models import Base, IndexTickData
from src.database.session import get_db


def check_table_exists(engine, table_name: str) -> bool:
    """Check if table exists in database"""
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()


def migrate():
    """Add IndexTickData table to database"""
    try:
        logger.info("=" * 80)
        logger.info("DATABASE MIGRATION: Adding IndexTickData table")
        logger.info("=" * 80)
        
        db = get_db()
        engine = db.engine
        
        # Check if table already exists
        if check_table_exists(engine, 'index_tick_data'):
            logger.warning("⚠️  Table 'index_tick_data' already exists")
            response = input("Do you want to drop and recreate it? (yes/no): ")
            
            if response.lower() == 'yes':
                logger.info("Dropping existing table...")
                IndexTickData.__table__.drop(engine)
                logger.info("✅ Table dropped")
            else:
                logger.info("Migration cancelled")
                return
        
        # Create table
        logger.info("Creating index_tick_data table...")
        IndexTickData.__table__.create(engine)
        logger.info("✅ Table created successfully")
        
        # Verify table was created
        if check_table_exists(engine, 'index_tick_data'):
            logger.info("✅ Migration completed successfully")
            logger.info("")
            logger.info("Table structure:")
            logger.info("  - tick_id (Primary Key)")
            logger.info("  - symbol (indexed)")
            logger.info("  - timestamp (indexed)")
            logger.info("  - price, bid, ask, spread")
            logger.info("  - volume, vix")
            logger.info("  - Technical indicators (SMA, RSI)")
            logger.info("  - Price changes and momentum")
            logger.info("")
            logger.info("Ready to collect real-time tick data! 📊")
        else:
            logger.error("❌ Migration failed - table not found after creation")
    
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise


if __name__ == "__main__":
    migrate()

