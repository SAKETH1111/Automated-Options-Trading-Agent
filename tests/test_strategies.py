"""Tests for trading strategies"""

import pytest
from datetime import datetime, timedelta

from src.strategies.bull_put_spread import BullPutSpreadStrategy
from src.strategies.cash_secured_put import CashSecuredPutStrategy
from src.strategies.iron_condor import IronCondorStrategy


@pytest.fixture
def bull_put_config():
    return {
        "enabled": True,
        "dte_range": [25, 45],
        "short_delta_range": [-0.30, -0.20],
        "width_range": [5, 10],
        "min_credit": 0.30,
        "max_risk_reward": 4.0,
    }


@pytest.fixture
def sample_stock_data():
    return {
        "symbol": "SPY",
        "price": 450.0,
        "bid": 449.95,
        "ask": 450.05,
        "iv_rank": 45,
        "volume": 50000000,
    }


@pytest.fixture
def sample_options_chain():
    expiration = datetime.now() + timedelta(days=35)
    return [
        {
            "option_symbol": "SPY230120P00440000",
            "underlying_symbol": "SPY",
            "underlying_price": 450.0,
            "option_type": "put",
            "strike": 440.0,
            "expiration": expiration,
            "dte": 35,
            "bid": 2.90,
            "ask": 3.10,
            "mid": 3.00,
            "volume": 1000,
            "open_interest": 5000,
            "spread": 0.20,
            "spread_pct": 6.67,
            "iv": 0.20,
            "delta": -0.25,
            "gamma": 0.015,
            "theta": -0.05,
            "vega": 0.25,
            "liquidity_score": 75.0,
        },
        {
            "option_symbol": "SPY230120P00435000",
            "underlying_symbol": "SPY",
            "underlying_price": 450.0,
            "option_type": "put",
            "strike": 435.0,
            "expiration": expiration,
            "dte": 35,
            "bid": 1.90,
            "ask": 2.10,
            "mid": 2.00,
            "volume": 800,
            "open_interest": 4000,
            "spread": 0.20,
            "spread_pct": 10.0,
            "iv": 0.19,
            "delta": -0.18,
            "gamma": 0.012,
            "theta": -0.04,
            "vega": 0.20,
            "liquidity_score": 70.0,
        }
    ]


class TestBullPutSpreadStrategy:
    """Test Bull Put Spread strategy"""
    
    def test_initialization(self, bull_put_config):
        strategy = BullPutSpreadStrategy(bull_put_config)
        assert strategy.name == "Bull Put Spread"
        assert strategy.enabled == True
    
    def test_generate_signals(self, bull_put_config, sample_stock_data, sample_options_chain):
        strategy = BullPutSpreadStrategy(bull_put_config)
        signals = strategy.generate_signals(
            "SPY",
            sample_stock_data,
            sample_options_chain
        )
        
        # Should generate at least one signal with good setup
        assert isinstance(signals, list)
    
    def test_should_exit_take_profit(self, bull_put_config):
        strategy = BullPutSpreadStrategy(bull_put_config)
        
        trade = {
            "params": {"dte": 30},
            "execution": {"fill_credit": 1.0},
            "risk": {"size": 1},
            "days_held": 10,
        }
        
        # Test take profit
        exit_signal = strategy.should_exit(trade, [], 50.0, 50.0)
        assert exit_signal is not None
        assert exit_signal["reason"] == "take_profit"
    
    def test_should_exit_stop_loss(self, bull_put_config):
        strategy = BullPutSpreadStrategy(bull_put_config)
        
        trade = {
            "params": {"dte": 30},
            "execution": {"fill_credit": 1.0},
            "risk": {"size": 1},
            "days_held": 10,
        }
        
        # Test stop loss
        exit_signal = strategy.should_exit(trade, [], -100.0, -100.0)
        assert exit_signal is not None
        assert exit_signal["reason"] == "stop_loss"


class TestCashSecuredPutStrategy:
    """Test Cash Secured Put strategy"""
    
    def test_initialization(self):
        config = {
            "enabled": True,
            "dte_range": [30, 45],
            "delta_range": [-0.30, -0.20],
            "min_premium": 0.50,
        }
        strategy = CashSecuredPutStrategy(config)
        assert strategy.name == "Cash Secured Put"


class TestIronCondorStrategy:
    """Test Iron Condor strategy"""
    
    def test_initialization(self):
        config = {
            "enabled": True,
            "dte_range": [30, 45],
            "short_put_delta_range": [-0.20, -0.15],
            "short_call_delta_range": [0.15, 0.20],
            "width": 5,
        }
        strategy = IronCondorStrategy(config)
        assert strategy.name == "Iron Condor"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])













