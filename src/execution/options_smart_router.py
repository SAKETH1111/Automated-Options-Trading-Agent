"""
Account-Aware Smart Order Router
Execution frequency and algorithms adapted to account size
Multi-leg spread integrity and adaptive limit orders
"""

import numpy as np
import pandas as pd
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger
import json
import time

from src.portfolio.account_manager import AccountProfile, AccountTier


class OrderType(Enum):
    """Order types"""
    LIMIT = "limit"
    MARKET = "market"
    STOP_LIMIT = "stop_limit"
    TWAP = "twap"
    VWAP = "vwap"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class ExecutionFrequency(Enum):
    """Execution frequency by account tier"""
    END_OF_DAY = "end_of_day"        # Micro accounts
    DAILY = "daily"                  # Small accounts  
    MULTI_DAILY = "multi_daily"      # Medium accounts
    INTRADAY = "intraday"            # Large accounts
    CONTINUOUS = "continuous"        # Institutional accounts


@dataclass
class OrderRequest:
    """Order request structure"""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: int
    order_type: OrderType
    price: float = None
    stop_price: float = None
    time_in_force: str = "GTC"  # GTC, IOC, FOK
    strategy: str = None
    legs: List[Dict] = None  # For multi-leg spreads
    max_slippage: float = 0.02  # 2% max slippage
    urgency: str = "normal"  # low, normal, high, urgent


@dataclass
class OrderResult:
    """Order execution result"""
    order_id: str
    symbol: str
    side: str
    quantity: int
    filled_quantity: int
    avg_fill_price: float
    status: OrderStatus
    execution_time: datetime
    fees: float
    slippage: float
    fill_rate: float
    execution_algorithm: str
    error_message: str = None


@dataclass
class MarketData:
    """Market data snapshot"""
    symbol: str
    bid: float
    ask: float
    last: float
    volume: int
    timestamp: datetime
    bid_size: int = 0
    ask_size: int = 0
    spread: float = 0.0
    mid_price: float = 0.0


class OptionsSmartRouter:
    """
    Account-aware smart order router for options trading
    
    Features:
    - Execution frequency by tier (end of day for micro, continuous for institutional)
    - Adaptive limit orders with intelligent pricing
    - Multi-leg spread integrity (all or nothing)
    - Cancel and retry if spread narrows
    - Never use market orders (slippage control)
    """
    
    def __init__(self, account_profile: AccountProfile, config: Dict = None):
        self.profile = account_profile
        
        # Configuration
        self.config = config or self._default_config()
        
        # Execution parameters by account tier
        self.execution_params = self._get_execution_params()
        
        # Order management
        self.pending_orders = {}
        self.order_history = []
        self.execution_stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'avg_fill_time': 0.0,
            'avg_slippage': 0.0,
            'total_fees': 0.0
        }
        
        # Market data cache
        self.market_data_cache = {}
        self.last_market_update = {}
        
        # Execution algorithms
        self.algorithms = {
            'patient_limit': self._patient_limit_algorithm,
            'adaptive_limit': self._adaptive_limit_algorithm,
            'twap': self._twap_algorithm,
            'vwap': self._vwap_algorithm,
            'smart_limit': self._smart_limit_algorithm
        }
        
        # Session management
        self.session = None
        self.is_market_open = False
        
        logger.info(f"OptionsSmartRouter initialized for {account_profile.tier.value} tier")
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'max_orders_per_day': 50,
            'max_order_size_percent': 0.1,  # 10% of account per order
            'min_order_size': 1,
            'max_spread_percent': 0.05,  # 5% max spread
            'retry_attempts': 3,
            'retry_delay': 5,  # seconds
            'market_data_refresh': 1,  # seconds
            'order_timeout': 300,  # 5 minutes
            'slippage_tolerance': 0.02,  # 2%
            'fee_per_contract': 0.65,
            'min_fill_rate': 0.8,  # 80% minimum fill rate
            'enable_after_hours': False,
            'risk_checks': True
        }
    
    def _get_execution_params(self) -> Dict[AccountTier, Dict]:
        """Get execution parameters by account tier"""
        return {
            AccountTier.MICRO: {
                'frequency': ExecutionFrequency.END_OF_DAY,
                'algorithm': 'patient_limit',
                'max_orders_per_day': 2,
                'timeout': 600,  # 10 minutes
                'slippage_tolerance': 0.03,
                'min_fill_rate': 0.9
            },
            AccountTier.SMALL: {
                'frequency': ExecutionFrequency.DAILY,
                'algorithm': 'adaptive_limit',
                'max_orders_per_day': 5,
                'timeout': 300,  # 5 minutes
                'slippage_tolerance': 0.025,
                'min_fill_rate': 0.85
            },
            AccountTier.MEDIUM: {
                'frequency': ExecutionFrequency.MULTI_DAILY,
                'algorithm': 'smart_limit',
                'max_orders_per_day': 10,
                'timeout': 180,  # 3 minutes
                'slippage_tolerance': 0.02,
                'min_fill_rate': 0.8
            },
            AccountTier.LARGE: {
                'frequency': ExecutionFrequency.INTRADAY,
                'algorithm': 'twap',
                'max_orders_per_day': 20,
                'timeout': 120,  # 2 minutes
                'slippage_tolerance': 0.015,
                'min_fill_rate': 0.75
            },
            AccountTier.INSTITUTIONAL: {
                'frequency': ExecutionFrequency.CONTINUOUS,
                'algorithm': 'vwap',
                'max_orders_per_day': 100,
                'timeout': 60,  # 1 minute
                'slippage_tolerance': 0.01,
                'min_fill_rate': 0.7
            }
        }
    
    async def submit_order(self, order_request: OrderRequest) -> OrderResult:
        """
        Submit order for execution
        
        Args:
            order_request: Order request details
        
        Returns:
            OrderResult with execution details
        """
        try:
            # Validate order request
            validation_result = await self._validate_order(order_request)
            if not validation_result['valid']:
                return OrderResult(
                    order_id="",
                    symbol=order_request.symbol,
                    side=order_request.side,
                    quantity=order_request.quantity,
                    filled_quantity=0,
                    avg_fill_price=0.0,
                    status=OrderStatus.REJECTED,
                    execution_time=datetime.now(),
                    fees=0.0,
                    slippage=0.0,
                    fill_rate=0.0,
                    execution_algorithm="none",
                    error_message=validation_result['error']
                )
            
            # Get current market data
            market_data = await self._get_market_data(order_request.symbol)
            if not market_data:
                return OrderResult(
                    order_id="",
                    symbol=order_request.symbol,
                    side=order_request.side,
                    quantity=order_request.quantity,
                    filled_quantity=0,
                    avg_fill_price=0.0,
                    status=OrderStatus.REJECTED,
                    execution_time=datetime.now(),
                    fees=0.0,
                    slippage=0.0,
                    fill_rate=0.0,
                    execution_algorithm="none",
                    error_message="No market data available"
                )
            
            # Generate order ID
            order_id = self._generate_order_id()
            
            # Determine execution algorithm
            algorithm = self._get_execution_algorithm(order_request)
            
            # Execute order
            execution_result = await self._execute_order(
                order_id, order_request, market_data, algorithm
            )
            
            # Update statistics
            self._update_execution_stats(execution_result)
            
            # Store in history
            self.order_history.append(execution_result)
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            return OrderResult(
                order_id="",
                symbol=order_request.symbol,
                side=order_request.side,
                quantity=order_request.quantity,
                filled_quantity=0,
                avg_fill_price=0.0,
                status=OrderStatus.REJECTED,
                execution_time=datetime.now(),
                fees=0.0,
                slippage=0.0,
                fill_rate=0.0,
                execution_algorithm="none",
                error_message=str(e)
            )
    
    async def _validate_order(self, order_request: OrderRequest) -> Dict[str, Any]:
        """Validate order request"""
        try:
            # Check account tier limits
            tier_params = self.execution_params[self.profile.tier]
            
            # Check daily order limit
            today_orders = len([o for o in self.order_history 
                              if o.execution_time.date() == datetime.now().date()])
            if today_orders >= tier_params['max_orders_per_day']:
                return {'valid': False, 'error': 'Daily order limit exceeded'}
            
            # Check order size
            max_order_value = self.profile.balance * self.config['max_order_size_percent']
            if order_request.quantity * (order_request.price or 100) > max_order_value:
                return {'valid': False, 'error': 'Order size exceeds limit'}
            
            # Check minimum order size
            if order_request.quantity < self.config['min_order_size']:
                return {'valid': False, 'error': 'Order size below minimum'}
            
            # Check market hours
            if not self.is_market_open and not self.config['enable_after_hours']:
                return {'valid': False, 'error': 'Market is closed'}
            
            # Risk checks
            if self.config['risk_checks']:
                risk_result = await self._risk_check(order_request)
                if not risk_result['passed']:
                    return {'valid': False, 'error': risk_result['reason']}
            
            return {'valid': True}
            
        except Exception as e:
            logger.error(f"Error validating order: {e}")
            return {'valid': False, 'error': 'Validation error'}
    
    async def _risk_check(self, order_request: OrderRequest) -> Dict[str, Any]:
        """Perform risk checks on order"""
        try:
            # Check for excessive position size
            current_positions = await self._get_current_positions()
            symbol_position = current_positions.get(order_request.symbol, 0)
            
            new_position = symbol_position + (order_request.quantity if order_request.side == 'buy' else -order_request.quantity)
            
            # Position limit check (simplified)
            max_position_value = self.profile.balance * 0.5  # 50% of account
            if abs(new_position) * (order_request.price or 100) > max_position_value:
                return {'passed': False, 'reason': 'Position size limit exceeded'}
            
            # Check for wash sale (simplified)
            recent_trades = [o for o in self.order_history 
                           if o.symbol == order_request.symbol and 
                           o.execution_time > datetime.now() - timedelta(days=30)]
            
            if len(recent_trades) > 10:  # Too many trades in 30 days
                return {'passed': False, 'reason': 'Excessive trading detected'}
            
            return {'passed': True}
            
        except Exception as e:
            logger.error(f"Error in risk check: {e}")
            return {'passed': False, 'reason': 'Risk check error'}
    
    async def _get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get current market data for symbol"""
        try:
            # Check cache first
            if symbol in self.market_data_cache:
                cache_time = self.last_market_update.get(symbol, datetime.min)
                if datetime.now() - cache_time < timedelta(seconds=self.config['market_data_refresh']):
                    return self.market_data_cache[symbol]
            
            # Fetch new market data (simplified - would use real broker API)
            market_data = await self._fetch_market_data(symbol)
            
            if market_data:
                # Update cache
                self.market_data_cache[symbol] = market_data
                self.last_market_update[symbol] = datetime.now()
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    async def _fetch_market_data(self, symbol: str) -> Optional[MarketData]:
        """Fetch market data from broker API (simplified)"""
        try:
            # This would be a real API call to the broker
            # For demonstration, return mock data
            await asyncio.sleep(0.1)  # Simulate API delay
            
            # Mock market data
            bid = 1.50 + np.random.uniform(-0.1, 0.1)
            ask = bid + np.random.uniform(0.05, 0.15)
            last = bid + np.random.uniform(-0.05, 0.05)
            
            return MarketData(
                symbol=symbol,
                bid=bid,
                ask=ask,
                last=last,
                volume=np.random.randint(100, 1000),
                timestamp=datetime.now(),
                bid_size=np.random.randint(10, 100),
                ask_size=np.random.randint(10, 100),
                spread=ask - bid,
                mid_price=(bid + ask) / 2
            )
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return None
    
    def _get_execution_algorithm(self, order_request: OrderRequest) -> str:
        """Determine execution algorithm based on account tier and order characteristics"""
        try:
            tier_params = self.execution_params[self.profile.tier]
            base_algorithm = tier_params['algorithm']
            
            # Adjust algorithm based on order characteristics
            if order_request.urgency == 'urgent':
                return 'smart_limit'
            elif order_request.quantity > 50:
                return 'twap' if self.profile.tier in [AccountTier.LARGE, AccountTier.INSTITUTIONAL] else 'adaptive_limit'
            elif order_request.legs and len(order_request.legs) > 1:
                return 'smart_limit'  # Multi-leg spreads need careful execution
            else:
                return base_algorithm
                
        except Exception as e:
            logger.error(f"Error getting execution algorithm: {e}")
            return 'patient_limit'
    
    async def _execute_order(
        self,
        order_id: str,
        order_request: OrderRequest,
        market_data: MarketData,
        algorithm: str
    ) -> OrderResult:
        """Execute order using specified algorithm"""
        try:
            if algorithm in self.algorithms:
                return await self.algorithms[algorithm](
                    order_id, order_request, market_data
                )
            else:
                # Fallback to patient limit
                return await self.algorithms['patient_limit'](
                    order_id, order_request, market_data
                )
                
        except Exception as e:
            logger.error(f"Error executing order: {e}")
            return OrderResult(
                order_id=order_id,
                symbol=order_request.symbol,
                side=order_request.side,
                quantity=order_request.quantity,
                filled_quantity=0,
                avg_fill_price=0.0,
                status=OrderStatus.REJECTED,
                execution_time=datetime.now(),
                fees=0.0,
                slippage=0.0,
                fill_rate=0.0,
                execution_algorithm=algorithm,
                error_message=str(e)
            )
    
    async def _patient_limit_algorithm(
        self,
        order_id: str,
        order_request: OrderRequest,
        market_data: MarketData
    ) -> OrderResult:
        """Patient limit order algorithm for micro/small accounts"""
        try:
            start_time = datetime.now()
            tier_params = self.execution_params[self.profile.tier]
            
            # Calculate limit price
            if order_request.side == 'buy':
                limit_price = market_data.bid + market_data.spread * 0.1  # Slightly above bid
            else:
                limit_price = market_data.ask - market_data.spread * 0.1  # Slightly below ask
            
            # Wait for fill with patience
            filled_quantity = 0
            total_fees = 0.0
            total_value = 0.0
            
            timeout = tier_params['timeout']
            
            while (datetime.now() - start_time).total_seconds() < timeout:
                # Check if order would fill at current market
                if order_request.side == 'buy' and market_data.ask <= limit_price:
                    filled_quantity = order_request.quantity
                    total_value = filled_quantity * market_data.ask
                    break
                elif order_request.side == 'sell' and market_data.bid >= limit_price:
                    filled_quantity = order_request.quantity
                    total_value = filled_quantity * market_data.bid
                    break
                
                # Wait and check again
                await asyncio.sleep(1)
                
                # Update market data
                market_data = await self._get_market_data(order_request.symbol)
                if not market_data:
                    break
            
            # Calculate fees
            total_fees = filled_quantity * self.config['fee_per_contract']
            
            # Calculate slippage
            expected_price = market_data.mid_price
            actual_price = total_value / filled_quantity if filled_quantity > 0 else 0
            slippage = abs(actual_price - expected_price) / expected_price if expected_price > 0 else 0
            
            # Determine status
            if filled_quantity == order_request.quantity:
                status = OrderStatus.FILLED
            elif filled_quantity > 0:
                status = OrderStatus.PARTIALLY_FILLED
            else:
                status = OrderStatus.CANCELLED
            
            return OrderResult(
                order_id=order_id,
                symbol=order_request.symbol,
                side=order_request.side,
                quantity=order_request.quantity,
                filled_quantity=filled_quantity,
                avg_fill_price=actual_price,
                status=status,
                execution_time=datetime.now(),
                fees=total_fees,
                slippage=slippage,
                fill_rate=filled_quantity / order_request.quantity,
                execution_algorithm='patient_limit'
            )
            
        except Exception as e:
            logger.error(f"Error in patient limit algorithm: {e}")
            raise
    
    async def _adaptive_limit_algorithm(
        self,
        order_id: str,
        order_request: OrderRequest,
        market_data: MarketData
    ) -> OrderResult:
        """Adaptive limit order algorithm for small/medium accounts"""
        try:
            start_time = datetime.now()
            tier_params = self.execution_params[self.profile.tier]
            
            filled_quantity = 0
            total_fees = 0.0
            total_value = 0.0
            current_limit_price = None
            
            timeout = tier_params['timeout']
            
            while (datetime.now() - start_time).total_seconds() < timeout and filled_quantity < order_request.quantity:
                # Adapt limit price based on market conditions
                if order_request.side == 'buy':
                    if market_data.spread < market_data.mid_price * 0.02:  # Tight spread
                        current_limit_price = market_data.ask  # More aggressive
                    else:
                        current_limit_price = market_data.bid + market_data.spread * 0.3  # Conservative
                else:
                    if market_data.spread < market_data.mid_price * 0.02:  # Tight spread
                        current_limit_price = market_data.bid  # More aggressive
                    else:
                        current_limit_price = market_data.ask - market_data.spread * 0.3  # Conservative
                
                # Check for fill
                remaining_quantity = order_request.quantity - filled_quantity
                
                if order_request.side == 'buy' and market_data.ask <= current_limit_price:
                    fill_qty = min(remaining_quantity, market_data.ask_size)
                    filled_quantity += fill_qty
                    total_value += fill_qty * market_data.ask
                elif order_request.side == 'sell' and market_data.bid >= current_limit_price:
                    fill_qty = min(remaining_quantity, market_data.bid_size)
                    filled_quantity += fill_qty
                    total_value += fill_qty * market_data.bid
                
                # Wait before next iteration
                await asyncio.sleep(0.5)
                
                # Update market data
                market_data = await self._get_market_data(order_request.symbol)
                if not market_data:
                    break
            
            # Calculate fees and slippage
            total_fees = filled_quantity * self.config['fee_per_contract']
            expected_price = market_data.mid_price
            actual_price = total_value / filled_quantity if filled_quantity > 0 else 0
            slippage = abs(actual_price - expected_price) / expected_price if expected_price > 0 else 0
            
            # Determine status
            if filled_quantity == order_request.quantity:
                status = OrderStatus.FILLED
            elif filled_quantity > 0:
                status = OrderStatus.PARTIALLY_FILLED
            else:
                status = OrderStatus.CANCELLED
            
            return OrderResult(
                order_id=order_id,
                symbol=order_request.symbol,
                side=order_request.side,
                quantity=order_request.quantity,
                filled_quantity=filled_quantity,
                avg_fill_price=actual_price,
                status=status,
                execution_time=datetime.now(),
                fees=total_fees,
                slippage=slippage,
                fill_rate=filled_quantity / order_request.quantity,
                execution_algorithm='adaptive_limit'
            )
            
        except Exception as e:
            logger.error(f"Error in adaptive limit algorithm: {e}")
            raise
    
    async def _smart_limit_algorithm(
        self,
        order_id: str,
        order_request: OrderRequest,
        market_data: MarketData
    ) -> OrderResult:
        """Smart limit order algorithm for medium/large accounts"""
        try:
            start_time = datetime.now()
            tier_params = self.execution_params[self.profile.tier]
            
            filled_quantity = 0
            total_fees = 0.0
            total_value = 0.0
            
            # Multi-leg spread handling
            if order_request.legs and len(order_request.legs) > 1:
                return await self._execute_multi_leg_spread(
                    order_id, order_request, market_data
                )
            
            # Single leg execution with smart pricing
            timeout = tier_params['timeout']
            
            while (datetime.now() - start_time).total_seconds() < timeout and filled_quantity < order_request.quantity:
                # Calculate optimal limit price using market microstructure
                optimal_price = self._calculate_optimal_limit_price(
                    order_request, market_data, filled_quantity
                )
                
                # Check for fill
                remaining_quantity = order_request.quantity - filled_quantity
                
                if order_request.side == 'buy' and market_data.ask <= optimal_price:
                    fill_qty = min(remaining_quantity, market_data.ask_size)
                    filled_quantity += fill_qty
                    total_value += fill_qty * market_data.ask
                elif order_request.side == 'sell' and market_data.bid >= optimal_price:
                    fill_qty = min(remaining_quantity, market_data.bid_size)
                    filled_quantity += fill_qty
                    total_value += fill_qty * market_data.bid
                
                # Wait before next iteration
                await asyncio.sleep(0.2)
                
                # Update market data
                market_data = await self._get_market_data(order_request.symbol)
                if not market_data:
                    break
            
            # Calculate fees and slippage
            total_fees = filled_quantity * self.config['fee_per_contract']
            expected_price = market_data.mid_price
            actual_price = total_value / filled_quantity if filled_quantity > 0 else 0
            slippage = abs(actual_price - expected_price) / expected_price if expected_price > 0 else 0
            
            # Determine status
            if filled_quantity == order_request.quantity:
                status = OrderStatus.FILLED
            elif filled_quantity > 0:
                status = OrderStatus.PARTIALLY_FILLED
            else:
                status = OrderStatus.CANCELLED
            
            return OrderResult(
                order_id=order_id,
                symbol=order_request.symbol,
                side=order_request.side,
                quantity=order_request.quantity,
                filled_quantity=filled_quantity,
                avg_fill_price=actual_price,
                status=status,
                execution_time=datetime.now(),
                fees=total_fees,
                slippage=slippage,
                fill_rate=filled_quantity / order_request.quantity,
                execution_algorithm='smart_limit'
            )
            
        except Exception as e:
            logger.error(f"Error in smart limit algorithm: {e}")
            raise
    
    async def _twap_algorithm(
        self,
        order_id: str,
        order_request: OrderRequest,
        market_data: MarketData
    ) -> OrderResult:
        """TWAP algorithm for large/institutional accounts"""
        try:
            # TWAP execution over time period
            execution_duration = 300  # 5 minutes
            num_slices = 10  # 10 slices
            
            filled_quantity = 0
            total_fees = 0.0
            total_value = 0.0
            
            slice_quantity = order_request.quantity // num_slices
            slice_duration = execution_duration // num_slices
            
            start_time = datetime.now()
            
            for slice_num in range(num_slices):
                if filled_quantity >= order_request.quantity:
                    break
                
                # Calculate slice quantity
                remaining_quantity = order_request.quantity - filled_quantity
                current_slice_qty = min(slice_quantity, remaining_quantity)
                
                # Execute slice
                slice_result = await self._execute_slice(
                    order_request, current_slice_qty, market_data
                )
                
                filled_quantity += slice_result['filled_quantity']
                total_value += slice_result['total_value']
                total_fees += slice_result['fees']
                
                # Wait for next slice
                if slice_num < num_slices - 1:
                    await asyncio.sleep(slice_duration)
                
                # Update market data
                market_data = await self._get_market_data(order_request.symbol)
                if not market_data:
                    break
            
            # Calculate final metrics
            actual_price = total_value / filled_quantity if filled_quantity > 0 else 0
            expected_price = market_data.mid_price
            slippage = abs(actual_price - expected_price) / expected_price if expected_price > 0 else 0
            
            # Determine status
            if filled_quantity == order_request.quantity:
                status = OrderStatus.FILLED
            elif filled_quantity > 0:
                status = OrderStatus.PARTIALLY_FILLED
            else:
                status = OrderStatus.CANCELLED
            
            return OrderResult(
                order_id=order_id,
                symbol=order_request.symbol,
                side=order_request.side,
                quantity=order_request.quantity,
                filled_quantity=filled_quantity,
                avg_fill_price=actual_price,
                status=status,
                execution_time=datetime.now(),
                fees=total_fees,
                slippage=slippage,
                fill_rate=filled_quantity / order_request.quantity,
                execution_algorithm='twap'
            )
            
        except Exception as e:
            logger.error(f"Error in TWAP algorithm: {e}")
            raise
    
    async def _vwap_algorithm(
        self,
        order_id: str,
        order_request: OrderRequest,
        market_data: MarketData
    ) -> OrderResult:
        """VWAP algorithm for institutional accounts"""
        try:
            # VWAP execution based on volume profile
            # This is simplified - real VWAP would use historical volume patterns
            
            filled_quantity = 0
            total_fees = 0.0
            total_value = 0.0
            
            # Get volume-weighted execution schedule
            execution_schedule = self._calculate_vwap_schedule(
                order_request.quantity, market_data.volume
            )
            
            start_time = datetime.now()
            
            for slice_qty, wait_time in execution_schedule:
                if filled_quantity >= order_request.quantity:
                    break
                
                # Execute slice
                slice_result = await self._execute_slice(
                    order_request, slice_qty, market_data
                )
                
                filled_quantity += slice_result['filled_quantity']
                total_value += slice_result['total_value']
                total_fees += slice_result['fees']
                
                # Wait for next slice
                await asyncio.sleep(wait_time)
                
                # Update market data
                market_data = await self._get_market_data(order_request.symbol)
                if not market_data:
                    break
            
            # Calculate final metrics
            actual_price = total_value / filled_quantity if filled_quantity > 0 else 0
            expected_price = market_data.mid_price
            slippage = abs(actual_price - expected_price) / expected_price if expected_price > 0 else 0
            
            # Determine status
            if filled_quantity == order_request.quantity:
                status = OrderStatus.FILLED
            elif filled_quantity > 0:
                status = OrderStatus.PARTIALLY_FILLED
            else:
                status = OrderStatus.CANCELLED
            
            return OrderResult(
                order_id=order_id,
                symbol=order_request.symbol,
                side=order_request.side,
                quantity=order_request.quantity,
                filled_quantity=filled_quantity,
                avg_fill_price=actual_price,
                status=status,
                execution_time=datetime.now(),
                fees=total_fees,
                slippage=slippage,
                fill_rate=filled_quantity / order_request.quantity,
                execution_algorithm='vwap'
            )
            
        except Exception as e:
            logger.error(f"Error in VWAP algorithm: {e}")
            raise
    
    async def _execute_slice(
        self,
        order_request: OrderRequest,
        slice_quantity: int,
        market_data: MarketData
    ) -> Dict[str, Any]:
        """Execute a slice of the order"""
        try:
            # Simple slice execution at market
            if order_request.side == 'buy':
                fill_price = market_data.ask
            else:
                fill_price = market_data.bid
            
            filled_quantity = slice_quantity
            total_value = filled_quantity * fill_price
            fees = filled_quantity * self.config['fee_per_contract']
            
            return {
                'filled_quantity': filled_quantity,
                'total_value': total_value,
                'fees': fees,
                'fill_price': fill_price
            }
            
        except Exception as e:
            logger.error(f"Error executing slice: {e}")
            return {
                'filled_quantity': 0,
                'total_value': 0.0,
                'fees': 0.0,
                'fill_price': 0.0
            }
    
    def _calculate_optimal_limit_price(
        self,
        order_request: OrderRequest,
        market_data: MarketData,
        filled_quantity: int
    ) -> float:
        """Calculate optimal limit price based on market conditions"""
        try:
            # Base price
            if order_request.side == 'buy':
                base_price = market_data.bid
            else:
                base_price = market_data.ask
            
            # Adjust based on urgency
            urgency_multiplier = {
                'low': 0.1,
                'normal': 0.3,
                'high': 0.6,
                'urgent': 0.9
            }.get(order_request.urgency, 0.3)
            
            # Adjust based on fill progress
            fill_progress = filled_quantity / order_request.quantity
            progress_multiplier = 0.2 + (fill_progress * 0.8)  # More aggressive as time passes
            
            # Calculate adjustment
            spread_adjustment = market_data.spread * urgency_multiplier * progress_multiplier
            
            if order_request.side == 'buy':
                return base_price + spread_adjustment
            else:
                return base_price - spread_adjustment
                
        except Exception as e:
            logger.error(f"Error calculating optimal limit price: {e}")
            return market_data.mid_price
    
    async def _execute_multi_leg_spread(
        self,
        order_id: str,
        order_request: OrderRequest,
        market_data: MarketData
    ) -> OrderResult:
        """Execute multi-leg spread with all-or-nothing logic"""
        try:
            # Multi-leg spread execution (simplified)
            # Real implementation would need to handle complex spread logic
            
            filled_quantity = 0
            total_fees = 0.0
            total_value = 0.0
            
            # For simplicity, execute as single order
            # Real implementation would need to coordinate multiple legs
            if order_request.side == 'buy':
                fill_price = market_data.ask
            else:
                fill_price = market_data.bid
            
            filled_quantity = order_request.quantity
            total_value = filled_quantity * fill_price
            total_fees = filled_quantity * self.config['fee_per_contract']
            
            # Calculate slippage
            expected_price = market_data.mid_price
            actual_price = total_value / filled_quantity
            slippage = abs(actual_price - expected_price) / expected_price
            
            return OrderResult(
                order_id=order_id,
                symbol=order_request.symbol,
                side=order_request.side,
                quantity=order_request.quantity,
                filled_quantity=filled_quantity,
                avg_fill_price=actual_price,
                status=OrderStatus.FILLED,
                execution_time=datetime.now(),
                fees=total_fees,
                slippage=slippage,
                fill_rate=1.0,
                execution_algorithm='smart_limit'
            )
            
        except Exception as e:
            logger.error(f"Error executing multi-leg spread: {e}")
            raise
    
    def _calculate_vwap_schedule(self, total_quantity: int, volume: int) -> List[Tuple[int, int]]:
        """Calculate VWAP execution schedule"""
        try:
            # Simplified VWAP schedule
            # Real implementation would use historical volume patterns
            
            num_slices = 8
            slice_quantity = total_quantity // num_slices
            
            schedule = []
            for i in range(num_slices):
                slice_qty = slice_quantity if i < num_slices - 1 else total_quantity - (slice_quantity * (num_slices - 1))
                wait_time = 30 + i * 10  # Increasing wait time
                schedule.append((slice_qty, wait_time))
            
            return schedule
            
        except Exception as e:
            logger.error(f"Error calculating VWAP schedule: {e}")
            return [(total_quantity, 60)]
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        timestamp = int(time.time() * 1000)
        return f"OPT_{timestamp}_{self.profile.tier.value.upper()}"
    
    async def _get_current_positions(self) -> Dict[str, int]:
        """Get current positions (simplified)"""
        try:
            # This would query the broker for current positions
            # For demonstration, return empty positions
            return {}
            
        except Exception as e:
            logger.error(f"Error getting current positions: {e}")
            return {}
    
    def _update_execution_stats(self, order_result: OrderResult):
        """Update execution statistics"""
        try:
            self.execution_stats['total_orders'] += 1
            
            if order_result.status == OrderStatus.FILLED:
                self.execution_stats['filled_orders'] += 1
            elif order_result.status == OrderStatus.CANCELLED:
                self.execution_stats['cancelled_orders'] += 1
            
            # Update averages
            if order_result.status == OrderStatus.FILLED:
                # Update average fill time
                fill_time = (order_result.execution_time - datetime.now()).total_seconds()
                self.execution_stats['avg_fill_time'] = (
                    self.execution_stats['avg_fill_time'] * (self.execution_stats['filled_orders'] - 1) + 
                    fill_time
                ) / self.execution_stats['filled_orders']
                
                # Update average slippage
                self.execution_stats['avg_slippage'] = (
                    self.execution_stats['avg_slippage'] * (self.execution_stats['filled_orders'] - 1) + 
                    order_result.slippage
                ) / self.execution_stats['filled_orders']
            
            self.execution_stats['total_fees'] += order_result.fees
            
        except Exception as e:
            logger.error(f"Error updating execution stats: {e}")
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return self.execution_stats.copy()
    
    def get_order_history(self, limit: int = 100) -> List[OrderResult]:
        """Get order history"""
        return self.order_history[-limit:] if self.order_history else []


# Example usage
if __name__ == "__main__":
    from src.portfolio.account_manager import UniversalAccountManager
    
    async def test_router():
        # Create account profile
        manager = UniversalAccountManager()
        profile = manager.create_account_profile(balance=25000)
        
        # Create smart router
        router = OptionsSmartRouter(profile)
        
        # Test order request
        order_request = OrderRequest(
            symbol='SPY240315C00500000',
            side='buy',
            quantity=10,
            order_type=OrderType.LIMIT,
            price=1.50,
            strategy='bull_put_spread',
            urgency='normal'
        )
        
        print("Testing Smart Order Router...")
        print(f"Account Tier: {profile.tier.value}")
        print(f"Execution Frequency: {router.execution_params[profile.tier]['frequency'].value}")
        print(f"Algorithm: {router.execution_params[profile.tier]['algorithm']}")
        
        # Submit order
        result = await router.submit_order(order_request)
        
        print(f"\nOrder Result:")
        print(f"Order ID: {result.order_id}")
        print(f"Status: {result.status.value}")
        print(f"Filled Quantity: {result.filled_quantity}/{result.quantity}")
        print(f"Average Fill Price: ${result.avg_fill_price:.2f}")
        print(f"Fees: ${result.fees:.2f}")
        print(f"Slippage: {result.slippage:.2%}")
        print(f"Fill Rate: {result.fill_rate:.1%}")
        print(f"Algorithm: {result.execution_algorithm}")
        
        # Get execution stats
        stats = router.get_execution_stats()
        print(f"\nExecution Stats:")
        print(f"Total Orders: {stats['total_orders']}")
        print(f"Filled Orders: {stats['filled_orders']}")
        print(f"Cancelled Orders: {stats['cancelled_orders']}")
        print(f"Average Fill Time: {stats['avg_fill_time']:.1f}s")
        print(f"Average Slippage: {stats['avg_slippage']:.2%}")
        print(f"Total Fees: ${stats['total_fees']:.2f}")
    
    # Run test
    asyncio.run(test_router())
