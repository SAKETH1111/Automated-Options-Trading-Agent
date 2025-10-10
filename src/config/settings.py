"""Configuration management using Pydantic"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class RiskSettings(BaseSettings):
    """Risk management settings"""
    max_daily_loss_pct: float = Field(default=5.0, description="Maximum daily loss percentage")
    max_position_size_pct: float = Field(default=20.0, description="Max position size as % of portfolio")
    max_trades_per_day: int = Field(default=10, description="Maximum trades per day")
    max_positions_per_symbol: int = Field(default=2, description="Max positions per symbol")
    stop_loss_pct: float = Field(default=50, description="Stop loss as % of credit received")
    take_profit_pct: float = Field(default=50, description="Take profit as % of credit received")
    max_portfolio_heat: float = Field(default=30, description="Max % of portfolio at risk")


class Settings(BaseSettings):
    """Main application settings"""
    
    # Alpaca API
    alpaca_api_key: str = Field(default="", env="ALPACA_API_KEY")
    alpaca_secret_key: str = Field(default="", env="ALPACA_SECRET_KEY")
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        env="ALPACA_BASE_URL"
    )
    
    # Database
    database_url: str = Field(
        default="sqlite:///./trading_agent.db",
        env="DATABASE_URL"
    )
    
    # Trading Mode
    trading_mode: str = Field(default="paper", env="TRADING_MODE")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/trading_agent.log", env="LOG_FILE")
    
    # Alerts
    alert_email: Optional[str] = Field(default=None, env="ALERT_EMAIL")
    alert_webhook_url: Optional[str] = Field(default=None, env="ALERT_WEBHOOK_URL")
    
    # Learning
    enable_learning: bool = Field(default=True, env="ENABLE_LEARNING")
    learning_update_frequency: str = Field(default="daily", env="LEARNING_UPDATE_FREQUENCY")
    
    # Risk settings
    risk: RiskSettings = Field(default_factory=RiskSettings)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"  # Allow extra fields from .env
    
    @classmethod
    def load_yaml_config(cls, config_path: str = "config/config.yaml"):
        """Load additional configuration from YAML file"""
        path = Path(config_path)
        if path.exists():
            with open(path, "r") as f:
                return yaml.safe_load(f)
        return {}


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


def get_config() -> dict:
    """Get full configuration including YAML"""
    settings = get_settings()
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    
    if config_path.exists():
        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f)
    else:
        yaml_config = {}
    
    return yaml_config

