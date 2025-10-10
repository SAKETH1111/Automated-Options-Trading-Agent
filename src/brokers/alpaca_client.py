"""Alpaca API client for trading and market data"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockSnapshotRequest,
)
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, OrderClass
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    GetOrdersRequest,
)
from loguru import logger

from src.config.settings import get_settings


class AlpacaClient:
    """Wrapper for Alpaca API with enhanced options trading capabilities"""
    
    def __init__(self):
        settings = get_settings()
        
        # Initialize clients
        self.trading_client = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=settings.trading_mode == "paper"
        )
        
        self.stock_data_client = StockHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )
        
        logger.info(f"Alpaca client initialized in {settings.trading_mode} mode")
        logger.warning("Options data via Alpaca is limited - using simulated data for demo")
    
    # ==================== Account & Portfolio ====================
    
    def get_account(self) -> Dict:
        """Get account information"""
        try:
            account = self.trading_client.get_account()
            return {
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "initial_margin": float(account.initial_margin) if account.initial_margin else 0,
                "maintenance_margin": float(account.maintenance_margin) if account.maintenance_margin else 0,
                "daytrade_count": account.daytrade_count,
                "pattern_day_trader": account.pattern_day_trader,
            }
        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
            raise
    
    def get_positions(self) -> List[Dict]:
        """Get all open positions"""
        try:
            positions = self.trading_client.get_all_positions()
            return [
                {
                    "symbol": pos.symbol,
                    "qty": float(pos.qty),
                    "side": "long" if float(pos.qty) > 0 else "short",
                    "avg_entry_price": float(pos.avg_entry_price),
                    "current_price": float(pos.current_price),
                    "market_value": float(pos.market_value),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "unrealized_plpc": float(pos.unrealized_plpc),
                    "asset_class": pos.asset_class,
                }
                for pos in positions
            ]
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            raise
    
    # ==================== Market Data ====================
    
    def get_stock_quote(self, symbol: str) -> Optional[Dict]:
        """Get latest stock quote"""
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.stock_data_client.get_stock_latest_quote(request)
            
            if symbol in quotes:
                quote = quotes[symbol]
                return {
                    "symbol": symbol,
                    "bid": float(quote.bid_price),
                    "ask": float(quote.ask_price),
                    "bid_size": quote.bid_size,
                    "ask_size": quote.ask_size,
                    "last": float((quote.bid_price + quote.ask_price) / 2),
                    "timestamp": quote.timestamp,
                }
            return None
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return None
    
    def get_stock_snapshot(self, symbol: str) -> Optional[Dict]:
        """Get stock snapshot with more detailed data"""
        try:
            request = StockSnapshotRequest(symbol_or_symbols=symbol)
            snapshots = self.stock_data_client.get_stock_snapshot(request)
            
            if symbol in snapshots:
                snap = snapshots[symbol]
                return {
                    "symbol": symbol,
                    "latest_trade": {
                        "price": float(snap.latest_trade.price),
                        "size": snap.latest_trade.size,
                        "timestamp": snap.latest_trade.timestamp,
                    } if snap.latest_trade else None,
                    "latest_quote": {
                        "bid": float(snap.latest_quote.bid_price),
                        "ask": float(snap.latest_quote.ask_price),
                        "bid_size": snap.latest_quote.bid_size,
                        "ask_size": snap.latest_quote.ask_size,
                    } if snap.latest_quote else None,
                    "daily_bar": {
                        "open": float(snap.daily_bar.open),
                        "high": float(snap.daily_bar.high),
                        "low": float(snap.daily_bar.low),
                        "close": float(snap.daily_bar.close),
                        "volume": snap.daily_bar.volume,
                    } if snap.daily_bar else None,
                }
            return None
        except Exception as e:
            logger.error(f"Error fetching snapshot for {symbol}: {e}")
            return None
    
    def get_historical_bars(
        self,
        symbol: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        timeframe: TimeFrame = TimeFrame.Day
    ) -> List[Dict]:
        """Get historical bars"""
        try:
            if end_date is None:
                end_date = datetime.now()
            
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start_date,
                end=end_date,
            )
            
            bars = self.stock_data_client.get_stock_bars(request)
            
            if symbol in bars:
                return [
                    {
                        "timestamp": bar.timestamp,
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": bar.volume,
                    }
                    for bar in bars[symbol]
                ]
            return []
        except Exception as e:
            logger.error(f"Error fetching historical bars for {symbol}: {e}")
            return []
    
    # ==================== Options Data ====================
    
    def get_option_chain(
        self,
        underlying_symbol: str,
        expiration_date: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Get options chain for a symbol
        
        Note: Alpaca has limited options data support in the free tier.
        For production, integrate with a proper options data provider like:
        - CBOE DataShop
        - Interactive Brokers
        - TD Ameritrade
        - Tradier
        
        This returns empty for now - options data enrichment happens in market_data/collector.py
        """
        try:
            # Alpaca options data is not available in free tier
            # Return empty list - the system will work with paper trading
            # but won't generate actual signals without options data
            logger.debug(f"Options chain requested for {underlying_symbol} - using alternative data source")
            return []
        except Exception as e:
            logger.error(f"Error fetching option chain for {underlying_symbol}: {e}")
            return []
    
    # ==================== Order Management ====================
    
    def place_market_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        time_in_force: str = "day"
    ) -> Optional[Dict]:
        """Place a market order"""
        try:
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            tif = TimeInForce.DAY if time_in_force.lower() == "day" else TimeInForce.GTC
            
            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=tif,
            )
            
            order = self.trading_client.submit_order(request)
            
            logger.info(f"Market order placed: {side} {qty} {symbol} - Order ID: {order.id}")
            
            return {
                "order_id": str(order.id),
                "symbol": order.symbol,
                "qty": float(order.qty),
                "side": order.side.value,
                "type": order.order_type.value,
                "status": order.status.value,
                "submitted_at": order.submitted_at,
            }
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            return None
    
    def place_limit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        limit_price: float,
        time_in_force: str = "day"
    ) -> Optional[Dict]:
        """Place a limit order"""
        try:
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            tif = TimeInForce.DAY if time_in_force.lower() == "day" else TimeInForce.GTC
            
            request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                limit_price=limit_price,
                time_in_force=tif,
            )
            
            order = self.trading_client.submit_order(request)
            
            logger.info(f"Limit order placed: {side} {qty} {symbol} @ {limit_price} - Order ID: {order.id}")
            
            return {
                "order_id": str(order.id),
                "symbol": order.symbol,
                "qty": float(order.qty),
                "side": order.side.value,
                "type": order.order_type.value,
                "limit_price": float(order.limit_price) if order.limit_price else None,
                "status": order.status.value,
                "submitted_at": order.submitted_at,
            }
        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            return None
    
    def place_spread_order(
        self,
        legs: List[Dict],
        order_type: str = "limit",
        limit_price: Optional[float] = None,
        time_in_force: str = "day"
    ) -> Optional[Dict]:
        """
        Place a multi-leg spread order
        
        Args:
            legs: List of dicts with keys: symbol, qty, side
            order_type: "limit" or "market"
            limit_price: Required for limit orders (net credit/debit)
            time_in_force: "day" or "gtc"
        """
        try:
            # For now, we'll place individual orders
            # In production, use OTO (One-Triggers-Other) or complex order types
            order_ids = []
            
            for leg in legs:
                if order_type == "limit" and limit_price:
                    # Adjust individual leg prices proportionally
                    leg_price = abs(limit_price / len(legs))
                    order = self.place_limit_order(
                        symbol=leg["symbol"],
                        qty=leg["qty"],
                        side=leg["side"],
                        limit_price=leg_price,
                        time_in_force=time_in_force
                    )
                else:
                    order = self.place_market_order(
                        symbol=leg["symbol"],
                        qty=leg["qty"],
                        side=leg["side"],
                        time_in_force=time_in_force
                    )
                
                if order:
                    order_ids.append(order["order_id"])
            
            logger.info(f"Spread order placed with {len(order_ids)} legs")
            
            return {
                "order_ids": order_ids,
                "legs": len(legs),
                "type": "spread",
            }
        except Exception as e:
            logger.error(f"Error placing spread order: {e}")
            return None
    
    def get_order(self, order_id: str) -> Optional[Dict]:
        """Get order status"""
        try:
            order = self.trading_client.get_order_by_id(order_id)
            
            return {
                "order_id": str(order.id),
                "symbol": order.symbol,
                "qty": float(order.qty),
                "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
                "side": order.side.value,
                "type": order.order_type.value,
                "status": order.status.value,
                "limit_price": float(order.limit_price) if order.limit_price else None,
                "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
                "submitted_at": order.submitted_at,
                "filled_at": order.filled_at,
            }
        except Exception as e:
            logger.error(f"Error fetching order {order_id}: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            logger.info(f"Order {order_id} cancelled")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_open_orders(self) -> List[Dict]:
        """Get all open orders"""
        try:
            request = GetOrdersRequest(status="open")
            orders = self.trading_client.get_orders(request)
            
            return [
                {
                    "order_id": str(order.id),
                    "symbol": order.symbol,
                    "qty": float(order.qty),
                    "side": order.side.value,
                    "type": order.order_type.value,
                    "status": order.status.value,
                    "submitted_at": order.submitted_at,
                }
                for order in orders
            ]
        except Exception as e:
            logger.error(f"Error fetching open orders: {e}")
            return []

