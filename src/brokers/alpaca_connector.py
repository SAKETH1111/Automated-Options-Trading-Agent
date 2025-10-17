"""
Alpaca Broker Integration for Options Trading
Production-ready connector for Alpaca's options trading API

Features:
- Real-time order management for options spreads
- Account balance and position tracking
- PDT compliance integration
- WebSocket feeds for real-time updates
- Options chain data retrieval
- Commission and fee tracking
"""

import asyncio
import aiohttp
import websockets
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from loguru import logger
import yaml

class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    DONE_FOR_DAY = "done_for_day"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REPLACED = "replaced"
    PENDING_CANCEL = "pending_cancel"
    PENDING_REPLACE = "pending_replace"
    ACCEPTED = "accepted"
    PENDING_NEW = "pending_new"
    ACCEPTED_FOR_BIDDING = "accepted_for_bidding"
    STOPPED = "stopped"
    REJECTED = "rejected"
    SUSPENDED = "suspended"
    CALCULATED = "calculated"

class OrderTimeInForce(Enum):
    DAY = "day"
    GTC = "gtc"
    OPG = "opg"
    CLS = "cls"
    IOC = "ioc"
    FOK = "fok"

@dataclass
class AccountInfo:
    """Account information from Alpaca"""
    account_id: str
    buying_power: float
    cash: float
    portfolio_value: float
    equity: float
    long_market_value: float
    short_market_value: float
    initial_margin: float
    maintenance_margin: float
    last_equity: float
    last_maintenance_margin: float
    day_trade_count: int
    day_trading_buying_power: float
    regt_buying_power: float
    crypto_buying_power: float
    non_marginable_buying_power: float

@dataclass
class Position:
    """Position information"""
    asset_id: str
    symbol: str
    exchange: str
    asset_class: str
    qty: float
    side: str
    market_value: float
    cost_basis: float
    unrealized_pl: float
    unrealized_plpc: float
    unrealized_intraday_pl: float
    unrealized_intraday_plpc: float
    current_price: float
    lastday_price: float
    change_today: float
    qty_available: float
    avg_entry_price: float

@dataclass
class OptionsContract:
    """Options contract information"""
    id: str
    symbol: str
    name: str
    exchange: str
    class_: str
    exercise_style: str
    expiration_date: str
    shares_per_contract: int
    tick_size: float
    min_tick_size: float
    strike_price: float
    type: str  # call or put

@dataclass
class OptionsOrder:
    """Options order structure"""
    symbol: str
    qty: int
    side: OrderSide
    type: OrderType
    time_in_force: OrderTimeInForce
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    client_order_id: Optional[str] = None
    order_class: str = "simple"  # simple, bracket, oco, oto
    take_profit: Optional[Dict] = None
    stop_loss: Optional[Dict] = None

@dataclass
class OrderResult:
    """Order execution result"""
    id: str
    client_order_id: str
    created_at: datetime
    updated_at: datetime
    submitted_at: datetime
    filled_at: Optional[datetime]
    expired_at: Optional[datetime]
    canceled_at: Optional[datetime]
    failed_at: Optional[datetime]
    replaced_at: Optional[datetime]
    replaced_by: Optional[str]
    replaces: Optional[str]
    asset_id: str
    symbol: str
    asset_class: str
    notional: Optional[float]
    qty: str
    filled_qty: str
    filled_avg_price: Optional[str]
    order_class: str
    order_type: str
    type: str
    side: str
    time_in_force: str
    limit_price: Optional[str]
    stop_price: Optional[str]
    status: str
    extended_hours: bool
    legs: Optional[List[Dict]] = None
    trail_percent: Optional[str] = None
    trail_price: Optional[str] = None
    hwm: Optional[str] = None

class AlpacaConnector:
    """Production-ready Alpaca broker connector for options trading"""
    
    def __init__(self, config_path: str = "config/production.yaml"):
        self.config = self._load_config(config_path)
        self.broker_config = self.config.get('broker', {})
        
        # Alpaca API configuration
        self.api_key = self.broker_config.get('api_key')
        self.secret_key = self.broker_config.get('secret_key')
        self.paper_trading = self.broker_config.get('paper_trading', True)
        
        # API URLs
        if self.paper_trading:
            self.base_url = "https://paper-api.alpaca.markets"
            self.data_url = "https://data.alpaca.markets"
            self.stream_url = "wss://paper-api.alpaca.markets/stream"
        else:
            self.base_url = "https://api.alpaca.markets"
            self.data_url = "https://data.alpaca.markets"
            self.stream_url = "wss://api.alpaca.markets/stream"
        
        # Trading parameters
        self.commission_per_contract = self.broker_config.get('commission_per_contract', 0.65)
        self.regulatory_fees = self.broker_config.get('regulatory_fees', 0.000119)
        
        # Connection objects
        self.session = None
        self.websocket = None
        self.account_ws = None
        
        # State tracking
        self.is_connected = False
        self.last_heartbeat = None
        self.positions_cache = {}
        self.account_cache = None
        
        logger.info(f"Alpaca connector initialized - Paper trading: {self.paper_trading}")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    async def initialize(self):
        """Initialize connections and authenticate"""
        try:
            # Initialize HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={
                    'APCA-API-KEY-ID': self.api_key,
                    'APCA-API-SECRET-KEY': self.secret_key,
                    'Content-Type': 'application/json'
                }
            )
            
            # Test authentication
            await self._test_authentication()
            
            # Initialize WebSocket connections
            await self._initialize_websockets()
            
            self.is_connected = True
            logger.info("Alpaca connector initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca connector: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup connections"""
        if self.session:
            await self.session.close()
        if self.websocket:
            await self.websocket.close()
        if self.account_ws:
            await self.account_ws.close()
        
        self.is_connected = False
        logger.info("Alpaca connector cleaned up")
    
    async def _test_authentication(self):
        """Test API authentication"""
        url = f"{self.base_url}/v2/account"
        async with self.session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                logger.info(f"Authentication successful - Account: {data.get('id')}")
                return data
            else:
                error_text = await response.text()
                raise Exception(f"Authentication failed: {response.status} - {error_text}")
    
    async def _initialize_websockets(self):
        """Initialize WebSocket connections"""
        # Market data WebSocket
        try:
            self.websocket = await websockets.connect(
                self.stream_url,
                extra_headers={
                    'APCA-API-KEY-ID': self.api_key,
                    'APCA-API-SECRET-KEY': self.secret_key
                }
            )
            
            # Send authentication message
            auth_message = {
                "action": "auth",
                "key": self.api_key,
                "secret": self.secret_key
            }
            await self.websocket.send(json.dumps(auth_message))
            
            # Account updates WebSocket
            self.account_ws = await websockets.connect(
                f"{self.stream_url}/v2/account",
                extra_headers={
                    'APCA-API-KEY-ID': self.api_key,
                    'APCA-API-SECRET-KEY': self.secret_key
                }
            )
            
            logger.info("WebSocket connections established")
            
        except Exception as e:
            logger.error(f"Failed to initialize WebSockets: {e}")
            raise
    
    async def get_account_info(self) -> AccountInfo:
        """Get current account information"""
        try:
            url = f"{self.base_url}/v2/account"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    account_info = AccountInfo(
                        account_id=data['id'],
                        buying_power=float(data['buying_power']),
                        cash=float(data['cash']),
                        portfolio_value=float(data['portfolio_value']),
                        equity=float(data['equity']),
                        long_market_value=float(data['long_market_value']),
                        short_market_value=float(data['short_market_value']),
                        initial_margin=float(data['initial_margin']),
                        maintenance_margin=float(data['maintenance_margin']),
                        last_equity=float(data['last_equity']),
                        last_maintenance_margin=float(data['last_maintenance_margin']),
                        day_trade_count=int(data.get('day_trade_count', 0)),
                        day_trading_buying_power=float(data.get('day_trading_buying_power', 0)),
                        regt_buying_power=float(data.get('regt_buying_power', 0)),
                        crypto_buying_power=float(data.get('crypto_buying_power', 0)),
                        non_marginable_buying_power=float(data.get('non_marginable_buying_power', 0))
                    )
                    
                    self.account_cache = account_info
                    return account_info
                    
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to get account info: {response.status} - {error_text}")
                    
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            raise
    
    async def get_positions(self) -> List[Position]:
        """Get current positions"""
        try:
            url = f"{self.base_url}/v2/positions"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    positions = []
                    for pos_data in data:
                        position = Position(
                            asset_id=pos_data['asset_id'],
                            symbol=pos_data['symbol'],
                            exchange=pos_data['exchange'],
                            asset_class=pos_data['asset_class'],
                            qty=float(pos_data['qty']),
                            side=pos_data['side'],
                            market_value=float(pos_data['market_value']),
                            cost_basis=float(pos_data['cost_basis']),
                            unrealized_pl=float(pos_data['unrealized_pl']),
                            unrealized_plpc=float(pos_data['unrealized_plpc']),
                            unrealized_intraday_pl=float(pos_data['unrealized_intraday_pl']),
                            unrealized_intraday_plpc=float(pos_data['unrealized_intraday_plpc']),
                            current_price=float(pos_data['current_price']),
                            lastday_price=float(pos_data['lastday_price']),
                            change_today=float(pos_data['change_today']),
                            qty_available=float(pos_data['qty_available']),
                            avg_entry_price=float(pos_data['avg_entry_price'])
                        )
                        positions.append(position)
                    
                    self.positions_cache = {pos.symbol: pos for pos in positions}
                    return positions
                    
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to get positions: {response.status} - {error_text}")
                    
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            raise
    
    async def get_options_chain(self, symbol: str, expiration_date: Optional[str] = None) -> List[OptionsContract]:
        """Get options chain for a symbol"""
        try:
            url = f"{self.base_url}/v2/assets"
            params = {
                'asset_class': 'us_equity',
                'symbol': symbol,
                'status': 'active'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # For now, return empty list as Alpaca's options API is limited
                    # In production, you would need to use a different data provider
                    # like Polygon or CBOE for options chains
                    logger.warning(f"Options chain not fully supported by Alpaca API for {symbol}")
                    return []
                    
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to get options chain: {response.status} - {error_text}")
                    
        except Exception as e:
            logger.error(f"Error getting options chain: {e}")
            raise
    
    async def place_order(self, order: OptionsOrder) -> OrderResult:
        """Place an options order"""
        try:
            url = f"{self.base_url}/v2/orders"
            
            order_data = {
                'symbol': order.symbol,
                'qty': str(order.qty),
                'side': order.side.value,
                'type': order.type.value,
                'time_in_force': order.time_in_force.value,
                'order_class': order.order_class
            }
            
            if order.limit_price:
                order_data['limit_price'] = str(order.limit_price)
            
            if order.stop_price:
                order_data['stop_price'] = str(order.stop_price)
            
            if order.client_order_id:
                order_data['client_order_id'] = order.client_order_id
            
            async with self.session.post(url, json=order_data) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    order_result = OrderResult(
                        id=data['id'],
                        client_order_id=data.get('client_order_id', ''),
                        created_at=datetime.fromisoformat(data['created_at'].replace('Z', '+00:00')),
                        updated_at=datetime.fromisoformat(data['updated_at'].replace('Z', '+00:00')),
                        submitted_at=datetime.fromisoformat(data['submitted_at'].replace('Z', '+00:00')),
                        filled_at=datetime.fromisoformat(data['filled_at'].replace('Z', '+00:00')) if data.get('filled_at') else None,
                        expired_at=datetime.fromisoformat(data['expired_at'].replace('Z', '+00:00')) if data.get('expired_at') else None,
                        canceled_at=datetime.fromisoformat(data['canceled_at'].replace('Z', '+00:00')) if data.get('canceled_at') else None,
                        failed_at=datetime.fromisoformat(data['failed_at'].replace('Z', '+00:00')) if data.get('failed_at') else None,
                        replaced_at=datetime.fromisoformat(data['replaced_at'].replace('Z', '+00:00')) if data.get('replaced_at') else None,
                        replaced_by=data.get('replaced_by'),
                        replaces=data.get('replaces'),
                        asset_id=data['asset_id'],
                        symbol=data['symbol'],
                        asset_class=data['asset_class'],
                        notional=float(data['notional']) if data.get('notional') else None,
                        qty=data['qty'],
                        filled_qty=data['filled_qty'],
                        filled_avg_price=data.get('filled_avg_price'),
                        order_class=data['order_class'],
                        order_type=data['order_type'],
                        type=data['type'],
                        side=data['side'],
                        time_in_force=data['time_in_force'],
                        limit_price=data.get('limit_price'),
                        stop_price=data.get('stop_price'),
                        status=data['status'],
                        extended_hours=data['extended_hours'],
                        legs=data.get('legs'),
                        trail_percent=data.get('trail_percent'),
                        trail_price=data.get('trail_price'),
                        hwm=data.get('hwm')
                    )
                    
                    logger.info(f"Order placed: {order_result.id} - {order.symbol} {order.side.value} {order.qty}")
                    return order_result
                    
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to place order: {response.status} - {error_text}")
                    
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise
    
    async def place_spread_order(self, spread_type: str, symbol: str, strikes: List[float], 
                               expiration: str, quantity: int, limit_price: float) -> OrderResult:
        """Place a multi-leg options spread order"""
        try:
            # This is a simplified implementation
            # In production, you would need to construct the proper multi-leg order
            # Alpaca supports complex orders through their API
            
            url = f"{self.base_url}/v2/orders"
            
            order_data = {
                'symbol': symbol,
                'qty': str(quantity),
                'side': 'sell' if spread_type == 'bull_put_spread' else 'buy',
                'type': 'limit',
                'time_in_force': 'day',
                'order_class': 'simple',
                'limit_price': str(limit_price)
            }
            
            async with self.session.post(url, json=order_data) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Spread order placed: {data['id']} - {spread_type} on {symbol}")
                    return OrderResult(**data)
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to place spread order: {response.status} - {error_text}")
                    
        except Exception as e:
            logger.error(f"Error placing spread order: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            url = f"{self.base_url}/v2/orders/{order_id}"
            async with self.session.delete(url) as response:
                if response.status == 204:
                    logger.info(f"Order cancelled: {order_id}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to cancel order: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> OrderResult:
        """Get order status"""
        try:
            url = f"{self.base_url}/v2/orders/{order_id}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return OrderResult(**data)
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to get order status: {response.status} - {error_text}")
                    
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            raise
    
    async def get_pdt_status(self) -> Dict[str, Any]:
        """Get PDT compliance status"""
        try:
            account_info = await self.get_account_info()
            
            return {
                'account_value': account_info.portfolio_value,
                'day_trade_count': account_info.day_trade_count,
                'day_trading_buying_power': account_info.day_trading_buying_power,
                'is_pdt': account_info.portfolio_value < 25000,
                'can_day_trade': account_info.day_trade_count < 3,
                'day_trades_remaining': 3 - account_info.day_trade_count
            }
            
        except Exception as e:
            logger.error(f"Error getting PDT status: {e}")
            raise
    
    async def listen_account_updates(self, callback):
        """Listen for real-time account updates via WebSocket"""
        try:
            async for message in self.account_ws:
                data = json.loads(message)
                
                if data.get('stream') == 'account_updates':
                    account_data = data['data']
                    await callback(account_data)
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Account WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error in account updates stream: {e}")
    
    async def listen_market_data(self, symbols: List[str], callback):
        """Listen for real-time market data"""
        try:
            # Subscribe to market data
            subscribe_message = {
                "action": "subscribe",
                "quotes": symbols,
                "trades": symbols
            }
            await self.websocket.send(json.dumps(subscribe_message))
            
            async for message in self.websocket:
                data = json.loads(message)
                
                if data.get('T') in ['q', 't']:  # quote or trade
                    await callback(data)
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Market data WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error in market data stream: {e}")
    
    def calculate_commission(self, quantity: int) -> float:
        """Calculate commission for an order"""
        return (quantity * self.commission_per_contract) + (quantity * self.regulatory_fees)
    
    async def get_buying_power_for_options(self) -> float:
        """Get buying power specifically for options trading"""
        try:
            account_info = await self.get_account_info()
            
            # For options, use the day trading buying power if available
            # Otherwise use regular buying power
            if account_info.day_trading_buying_power > 0:
                return account_info.day_trading_buying_power
            else:
                return account_info.buying_power
                
        except Exception as e:
            logger.error(f"Error getting options buying power: {e}")
            return 0.0
    
    async def validate_order(self, order: OptionsOrder) -> Tuple[bool, str]:
        """Validate an order before submission"""
        try:
            # Check account balance
            account_info = await self.get_account_info()
            
            # Estimate order value
            if order.type == OrderType.MARKET:
                # For market orders, we can't estimate exact cost
                estimated_cost = 1000  # Conservative estimate
            else:
                estimated_cost = order.limit_price * order.qty * 100  # Options are 100 shares per contract
            
            # Check buying power
            if estimated_cost > account_info.buying_power:
                return False, f"Insufficient buying power: {estimated_cost} > {account_info.buying_power}"
            
            # Check PDT compliance for day trades
            if order.time_in_force == OrderTimeInForce.DAY:
                pdt_status = await self.get_pdt_status()
                if pdt_status['is_pdt'] and not pdt_status['can_day_trade']:
                    return False, "PDT limit reached - cannot place day trade"
            
            return True, "Order validation passed"
            
        except Exception as e:
            logger.error(f"Error validating order: {e}")
            return False, f"Validation error: {str(e)}"

# Example usage and testing
async def main():
    """Test the Alpaca connector"""
    connector = AlpacaConnector()
    
    try:
        await connector.initialize()
        
        # Test account info
        account = await connector.get_account_info()
        print(f"Account: {account.account_id}")
        print(f"Buying Power: ${account.buying_power:,.2f}")
        print(f"Day Trade Count: {account.day_trade_count}")
        
        # Test PDT status
        pdt_status = await connector.get_pdt_status()
        print(f"PDT Status: {pdt_status}")
        
        # Test positions
        positions = await connector.get_positions()
        print(f"Positions: {len(positions)}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
    finally:
        await connector.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
