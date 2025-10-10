"""
Trading Automation Module
Provides automated signal generation, order execution, and trade management
"""

from .signal_generator import AutomatedSignalGenerator
from .order_executor import AutomatedOrderExecutor
from .position_manager import AutomatedPositionManager
from .trade_manager import AutomatedTradeManager
from .performance_tracker import PerformanceTracker
from .auto_trader import AutomatedTrader

__all__ = [
    'AutomatedSignalGenerator',
    'AutomatedOrderExecutor',
    'AutomatedPositionManager',
    'AutomatedTradeManager',
    'PerformanceTracker',
    'AutomatedTrader'
]

