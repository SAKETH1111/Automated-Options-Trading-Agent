"""Database models for trade journal and analytics"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON, Column, DateTime, Float, ForeignKey, Integer, String, Text, Boolean
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Trade(Base):
    """Trade journal entry - captures complete trade lifecycle"""
    __tablename__ = "trades"
    
    # Identity
    trade_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Timestamps
    timestamp_enter = Column(DateTime, nullable=False, default=datetime.utcnow)
    timestamp_exit = Column(DateTime, nullable=True)
    
    # Trade Details
    symbol = Column(String(10), nullable=False, index=True)
    strategy = Column(String(50), nullable=False, index=True)
    
    # Strategy Parameters
    params = Column(JSON, nullable=False)  # {dte, delta, width, etc.}
    
    # Market Snapshot at Entry
    market_snapshot = Column(JSON, nullable=False)  # {price, ivr, oi, spread, etc.}
    
    # Execution Details
    execution = Column(JSON, nullable=False)  # {limit_credit, fill_credit, slippage}
    
    # Risk
    risk = Column(JSON, nullable=False)  # {size, risk_pct, max_loss}
    
    # Outcome
    pnl = Column(Float, default=0.0)
    pnl_pct = Column(Float, default=0.0)
    days_held = Column(Integer, default=0)
    exit_reason = Column(String(50))  # stop_loss, take_profit, expiration, manual, roll
    
    # Learning
    reason_tags = Column(JSON, default=list)  # [error_category1, error_category2]
    notes = Column(Text)
    
    # Status
    status = Column(String(20), default="open")  # open, closed, rolled
    
    # Relations
    positions = relationship("Position", back_populates="trade", cascade="all, delete-orphan")
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Position(Base):
    """Individual option positions within a trade"""
    __tablename__ = "positions"
    
    position_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    trade_id = Column(String, ForeignKey("trades.trade_id"), nullable=False)
    
    # Option Details
    symbol = Column(String(10), nullable=False)
    option_symbol = Column(String(50), nullable=False)  # OCC format
    option_type = Column(String(4), nullable=False)  # CALL or PUT
    strike = Column(Float, nullable=False)
    expiration = Column(DateTime, nullable=False)
    
    # Position
    side = Column(String(10), nullable=False)  # long or short
    quantity = Column(Integer, nullable=False)
    
    # Entry
    entry_price = Column(Float, nullable=False)
    entry_time = Column(DateTime, nullable=False)
    entry_iv = Column(Float)
    entry_delta = Column(Float)
    entry_gamma = Column(Float)
    entry_theta = Column(Float)
    entry_vega = Column(Float)
    
    # Exit
    exit_price = Column(Float)
    exit_time = Column(DateTime)
    exit_iv = Column(Float)
    
    # Current (for open positions)
    current_price = Column(Float)
    current_iv = Column(Float)
    current_delta = Column(Float)
    current_gamma = Column(Float)
    current_theta = Column(Float)
    current_vega = Column(Float)
    
    # P&L
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    
    # Status
    status = Column(String(20), default="open")  # open, closed
    
    # Relations
    trade = relationship("Trade", back_populates="positions")
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PerformanceMetric(Base):
    """Daily/weekly/monthly performance metrics"""
    __tablename__ = "performance_metrics"
    
    metric_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Time Period
    period_type = Column(String(20), nullable=False)  # daily, weekly, monthly
    period_start = Column(DateTime, nullable=False, index=True)
    period_end = Column(DateTime, nullable=False)
    
    # Performance
    total_pnl = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float, default=0.0)
    
    # Trade Statistics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    avg_win = Column(Float, default=0.0)
    avg_loss = Column(Float, default=0.0)
    
    # Strategy Breakdown
    strategy_performance = Column(JSON)  # {strategy_name: {pnl, win_rate, etc.}}
    
    # Error Analysis
    error_counts = Column(JSON)  # {error_category: count}
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)


class LearningLog(Base):
    """Learning system updates and parameter adjustments"""
    __tablename__ = "learning_logs"
    
    log_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Timestamp
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Update Type
    update_type = Column(String(50), nullable=False)  # parameter_adjustment, attribution_analysis, etc.
    
    # Strategy
    strategy = Column(String(50), nullable=False, index=True)
    
    # Old vs New Parameters
    old_params = Column(JSON)
    new_params = Column(JSON)
    
    # Reasoning
    reason = Column(Text, nullable=False)
    confidence = Column(Float)  # 0-1 confidence score
    
    # Impact Tracking
    trades_analyzed = Column(Integer)
    expected_improvement = Column(Float)  # Expected improvement in win rate or profit
    
    # Attribution Analysis
    factor_contributions = Column(JSON)  # {factor: contribution_score}
    
    # A/B Test Results (if applicable)
    ab_test_results = Column(JSON)
    
    # Status
    status = Column(String(20), default="active")  # active, rolled_back, superseded
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)


class MarketRegime(Base):
    """Track market regime for context-aware strategy selection"""
    __tablename__ = "market_regimes"
    
    regime_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Timestamp
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Regime Classification
    regime = Column(String(50), nullable=False)  # bull, bear, sideways, high_vol, low_vol
    
    # Market Metrics
    vix = Column(Float)
    vix_rank = Column(Float)  # 0-100
    market_trend = Column(String(20))  # up, down, sideways
    volatility = Column(Float)
    
    # Correlation
    spy_correlation = Column(Float)
    
    # Additional Context
    market_metadata = Column(JSON)  # Additional market data
    
    created_at = Column(DateTime, default=datetime.utcnow)


class IndexTickData(Base):
    """Store second-by-second tick data for indexes (SPY, QQQ, etc.)"""
    __tablename__ = "index_tick_data"
    
    tick_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Symbol and Timestamp
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Price Data
    price = Column(Float, nullable=False)
    bid = Column(Float)
    ask = Column(Float)
    bid_size = Column(Integer)
    ask_size = Column(Integer)
    spread = Column(Float)
    spread_pct = Column(Float)
    
    # Volume
    volume = Column(Integer)  # Cumulative daily volume at this tick
    last_trade_size = Column(Integer)
    
    # Market Metrics
    vix = Column(Float)  # VIX value (for SPY/QQQ context)
    
    # Technical Indicators (can be calculated later)
    sma_5 = Column(Float)  # 5-second moving average
    sma_60 = Column(Float)  # 1-minute moving average
    rsi = Column(Float)  # RSI if calculated
    
    # Momentum
    price_change = Column(Float)  # Change from previous tick
    price_change_pct = Column(Float)
    
    # Metadata
    market_state = Column(String(20))  # open, pre_market, after_hours, closed
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<IndexTickData(symbol={self.symbol}, timestamp={self.timestamp}, price={self.price})>"


class TechnicalIndicators(Base):
    """Store calculated technical indicators for analysis"""
    __tablename__ = "technical_indicators"
    
    indicator_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Symbol and Timestamp
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Moving Averages
    sma_10 = Column(Float)
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    sma_200 = Column(Float)
    ema_12 = Column(Float)
    ema_26 = Column(Float)
    ema_50 = Column(Float)
    
    # Momentum Indicators
    rsi_14 = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_histogram = Column(Float)
    stoch_k = Column(Float)
    stoch_d = Column(Float)
    
    # Volatility Indicators
    bb_upper = Column(Float)
    bb_middle = Column(Float)
    bb_lower = Column(Float)
    bb_width = Column(Float)
    atr_14 = Column(Float)
    
    # Trend Indicators
    adx = Column(Float)
    plus_di = Column(Float)
    minus_di = Column(Float)
    
    # Volume Indicators
    obv = Column(Float)
    vwap = Column(Float)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<TechnicalIndicators(symbol={self.symbol}, timestamp={self.timestamp})>"


class MarketRegime(Base):
    """Store market regime classifications"""
    __tablename__ = "market_regimes"
    
    regime_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Symbol and Timestamp
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Regime Classifications
    volatility_regime = Column(String(50))
    trend_regime = Column(String(50))
    momentum_regime = Column(String(50))
    volume_regime = Column(String(50))
    market_hours_regime = Column(String(50))
    
    # Regime Metrics
    volatility_percentile = Column(Float)
    trend_strength = Column(Float)
    rsi_value = Column(Float)
    volume_ratio = Column(Float)
    
    # Overall Assessment
    overall_regime = Column(String(50))
    recommended_action = Column(String(50))
    recommended_strategy = Column(Text)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<MarketRegime(symbol={self.symbol}, regime={self.overall_regime})>"


class PatternDetection(Base):
    """Store detected chart patterns"""
    __tablename__ = "pattern_detections"
    
    pattern_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Symbol and Timestamp
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Support/Resistance
    support_levels = Column(JSON)  # List of support prices
    resistance_levels = Column(JSON)  # List of resistance prices
    near_support = Column(Boolean)
    near_resistance = Column(Boolean)
    
    # Trend Detection
    trend_direction = Column(String(20))
    trend_strength = Column(String(20))
    trend_angle = Column(Float)
    
    # Pattern Recognition
    higher_highs = Column(Boolean)
    higher_lows = Column(Boolean)
    lower_highs = Column(Boolean)
    lower_lows = Column(Boolean)
    pattern_type = Column(String(50))
    
    # Breakout Detection
    breakout_detected = Column(Boolean)
    breakout_direction = Column(String(10))
    volume_confirmed = Column(Boolean)
    
    # Consolidation
    is_consolidating = Column(Boolean)
    consolidation_range = Column(Float)
    
    # Reversal Patterns
    reversal_detected = Column(Boolean)
    reversal_type = Column(String(50))
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<PatternDetection(symbol={self.symbol}, pattern={self.pattern_type})>"


class OptionsChain(Base):
    """Store real-time options chain data"""
    __tablename__ = "options_chains"
    
    chain_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Underlying
    symbol = Column(String(10), nullable=False, index=True)
    underlying_price = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Option Details
    option_symbol = Column(String(50), nullable=False, index=True)
    option_type = Column(String(4), nullable=False)  # CALL or PUT
    strike = Column(Float, nullable=False, index=True)
    expiration = Column(DateTime, nullable=False, index=True)
    dte = Column(Integer, nullable=False)  # Days to expiration
    
    # Pricing
    bid = Column(Float)
    ask = Column(Float)
    mid_price = Column(Float)
    last_price = Column(Float)
    mark = Column(Float)
    
    # Greeks
    delta = Column(Float)
    gamma = Column(Float)
    theta = Column(Float)
    vega = Column(Float)
    rho = Column(Float)
    
    # Implied Volatility
    implied_volatility = Column(Float)
    
    # Volume & Open Interest
    volume = Column(Integer)
    open_interest = Column(Integer)
    
    # Spread Analysis
    bid_ask_spread = Column(Float)
    bid_ask_spread_pct = Column(Float)
    
    # Moneyness
    intrinsic_value = Column(Float)
    extrinsic_value = Column(Float)
    moneyness = Column(String(10))  # ITM, ATM, OTM
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<OptionsChain(symbol={self.symbol}, strike={self.strike}, type={self.option_type})>"


class ImpliedVolatility(Base):
    """Track implied volatility metrics over time"""
    __tablename__ = "implied_volatility"
    
    iv_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Symbol and Timestamp
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Current IV Metrics
    iv_30 = Column(Float)  # 30-day IV
    iv_60 = Column(Float)  # 60-day IV
    iv_90 = Column(Float)  # 90-day IV
    
    # IV Statistics
    iv_mean = Column(Float)
    iv_std = Column(Float)
    iv_min = Column(Float)
    iv_max = Column(Float)
    
    # IV Rank and Percentile
    iv_rank = Column(Float)  # 0-100, where current IV stands in 52-week range
    iv_percentile = Column(Float)  # 0-100, percentage of days IV was below current
    
    # Historical Volatility
    hv_10 = Column(Float)  # 10-day historical volatility
    hv_20 = Column(Float)  # 20-day historical volatility
    hv_30 = Column(Float)  # 30-day historical volatility
    
    # IV vs HV
    iv_hv_ratio = Column(Float)  # IV / HV ratio
    
    # Term Structure
    iv_skew = Column(Float)  # Put-call IV skew
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ImpliedVolatility(symbol={self.symbol}, iv_rank={self.iv_rank})>"


class OptionsOpportunity(Base):
    """Store identified options trading opportunities"""
    __tablename__ = "options_opportunities"
    
    opportunity_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Symbol and Timestamp
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Opportunity Details
    strategy_type = Column(String(50), nullable=False)  # bull_put_spread, iron_condor, etc.
    opportunity_score = Column(Float)  # 0-100 score
    confidence = Column(Float)  # 0-1 confidence level
    
    # Strategy Parameters
    strikes = Column(JSON)  # List of strike prices involved
    expiration = Column(DateTime)
    dte = Column(Integer)
    
    # Pricing
    entry_credit = Column(Float)  # For credit spreads
    entry_debit = Column(Float)  # For debit spreads
    max_profit = Column(Float)
    max_loss = Column(Float)
    breakeven = Column(Float)
    
    # Greeks
    position_delta = Column(Float)
    position_theta = Column(Float)
    position_vega = Column(Float)
    
    # Probabilities
    pop = Column(Float)  # Probability of profit
    pop_50 = Column(Float)  # Probability of 50% max profit
    
    # Risk Metrics
    risk_reward_ratio = Column(Float)
    required_margin = Column(Float)
    return_on_risk = Column(Float)
    
    # Market Conditions
    iv_rank = Column(Float)
    underlying_price = Column(Float)
    trend = Column(String(20))
    
    # Reasons
    reasons = Column(JSON)  # List of reasons for this opportunity
    
    # Status
    status = Column(String(20), default="identified")  # identified, executed, expired, closed
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<OptionsOpportunity(symbol={self.symbol}, strategy={self.strategy_type}, score={self.opportunity_score})>"


class UnusualOptionsActivity(Base):
    """Track unusual options activity"""
    __tablename__ = "unusual_options_activity"
    
    activity_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Symbol and Timestamp
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Option Details
    option_symbol = Column(String(50), nullable=False)
    option_type = Column(String(4), nullable=False)
    strike = Column(Float, nullable=False)
    expiration = Column(DateTime, nullable=False)
    
    # Activity Metrics
    volume = Column(Integer)
    open_interest = Column(Integer)
    volume_oi_ratio = Column(Float)  # Volume / OI ratio
    avg_volume_20d = Column(Integer)  # 20-day average volume
    volume_ratio = Column(Float)  # Current volume / Average volume
    
    # Unusual Activity Indicators
    is_unusual_volume = Column(Boolean, default=False)
    is_sweep = Column(Boolean, default=False)  # Block trade sweep
    is_block_trade = Column(Boolean, default=False)
    
    # Trade Details
    premium_spent = Column(Float)  # Total premium for the trades
    avg_price = Column(Float)
    
    # Sentiment
    sentiment = Column(String(20))  # bullish, bearish, neutral
    
    # Greeks at Detection
    delta = Column(Float)
    implied_volatility = Column(Float)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<UnusualOptionsActivity(symbol={self.symbol}, strike={self.strike}, volume_ratio={self.volume_ratio})>"

