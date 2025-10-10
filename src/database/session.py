"""Database session management"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config.settings import get_settings
from .models import Base


class DatabaseSession:
    """Database session manager"""
    
    def __init__(self):
        settings = get_settings()
        self.engine = create_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
    
    def create_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)
    
    def drop_tables(self):
        """Drop all database tables"""
        Base.metadata.drop_all(bind=self.engine)
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


# Global database session instance
_db_session: DatabaseSession = None


def get_db() -> DatabaseSession:
    """Get global database session instance"""
    global _db_session
    if _db_session is None:
        _db_session = DatabaseSession()
        _db_session.create_tables()
    return _db_session


def get_session() -> Session:
    """Get a database session (for direct use)"""
    db = get_db()
    return db.SessionLocal()


def get_engine():
    """Get database engine"""
    db = get_db()
    return db.engine


