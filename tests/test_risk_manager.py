"""Tests for risk manager"""

import pytest
from src.risk.manager import RiskManager
from src.risk.position_sizer import PositionSizer


@pytest.fixture
def risk_config():
    return {
        "trading": {
            "risk": {
                "max_daily_loss_pct": 5.0,
                "max_position_size_pct": 20.0,
                "max_trades_per_day": 10,
                "max_positions_per_symbol": 2,
                "max_portfolio_heat": 30.0,
            }
        }
    }


@pytest.fixture
def sample_signal():
    return {
        "symbol": "SPY",
        "strategy_name": "Bull Put Spread",
        "max_loss": 500.0,
        "max_profit": 100.0,
    }


class TestRiskManager:
    """Test risk management"""
    
    def test_position_size_check(self, risk_config):
        risk_manager = RiskManager(risk_config)
        
        # Position within limits
        assert risk_manager._check_position_size(1000.0, 10000.0) == True
        
        # Position too large
        assert risk_manager._check_position_size(3000.0, 10000.0) == False
    
    def test_portfolio_heat(self, risk_config):
        risk_manager = RiskManager(risk_config)
        
        current_positions = [
            {"max_loss": 500.0},
            {"max_loss": 500.0},
        ]
        
        # Within heat limit
        assert risk_manager._check_portfolio_heat(500.0, 10000.0, current_positions) == True
        
        # Exceeds heat limit
        assert risk_manager._check_portfolio_heat(2000.0, 10000.0, current_positions) == False


class TestPositionSizer:
    """Test position sizing"""
    
    def test_fixed_risk_sizing(self, risk_config):
        sizer = PositionSizer(risk_config)
        
        signal = {
            "max_loss": 500.0,
            "max_profit": 100.0,
        }
        
        # Should risk 1% of 10000 = 100, so 100/500 = 0.2 contracts, floor to 1
        size = sizer.calculate_size(signal, 10000.0)
        assert size >= 1
    
    def test_validate_size(self, risk_config):
        sizer = PositionSizer(risk_config)
        
        # Size within limits
        validated = sizer.validate_size(2, 500.0, 10000.0, 20.0)
        assert validated == 2
        
        # Size too large, should reduce
        validated = sizer.validate_size(10, 500.0, 10000.0, 20.0)
        assert validated < 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])











