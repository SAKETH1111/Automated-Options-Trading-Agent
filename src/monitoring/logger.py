"""Logging configuration"""

import sys
from pathlib import Path

from loguru import logger

from src.config.settings import get_settings


def setup_logging():
    """Configure logging for the application"""
    settings = get_settings()
    
    # Remove default handler
    logger.remove()
    
    # Console handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.log_level,
        colorize=True,
    )
    
    # File handler
    log_file = Path(settings.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        settings.log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=settings.log_level,
        rotation="100 MB",
        retention="30 days",
        compression="zip",
    )
    
    # Trade journal file (separate, structured)
    logger.add(
        "logs/trade_journal.jsonl",
        format="{message}",
        level="INFO",
        filter=lambda record: "TRADE_JOURNAL" in record["extra"],
        rotation="10 MB",
        retention="1 year",
        serialize=True,
    )
    
    logger.info("Logging configured")













