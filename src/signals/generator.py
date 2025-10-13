"""Signal generation engine that coordinates strategies"""

from typing import Dict, List, Optional

from loguru import logger

from src.config.settings import get_config
from src.market_data.collector import MarketDataCollector
from src.strategies import BullPutSpreadStrategy, CashSecuredPutStrategy, IronCondorStrategy


class SignalGenerator:
    """Generate trading signals from multiple strategies"""
    
    def __init__(
        self,
        market_data: Optional[MarketDataCollector] = None,
        config: Optional[Dict] = None
    ):
        self.market_data = market_data or MarketDataCollector()
        self.config = config or get_config()
        
        # Initialize strategies
        self.strategies = self._initialize_strategies()
        
        logger.info(f"Signal Generator initialized with {len(self.strategies)} strategies")
    
    def _initialize_strategies(self) -> Dict:
        """Initialize all enabled strategies"""
        strategies = {}
        
        strategy_configs = self.config.get("strategies", {})
        
        # Bull Put Spread
        if strategy_configs.get("bull_put_spread", {}).get("enabled", True):
            strategies["bull_put_spread"] = BullPutSpreadStrategy(
                strategy_configs["bull_put_spread"]
            )
        
        # Cash Secured Put
        if strategy_configs.get("cash_secured_put", {}).get("enabled", True):
            strategies["cash_secured_put"] = CashSecuredPutStrategy(
                strategy_configs["cash_secured_put"]
            )
        
        # Iron Condor
        if strategy_configs.get("iron_condor", {}).get("enabled", True):
            strategies["iron_condor"] = IronCondorStrategy(
                strategy_configs["iron_condor"]
            )
        
        return strategies
    
    def scan_for_signals(self, symbols: Optional[List[str]] = None) -> List[Dict]:
        """
        Scan watchlist for trading signals
        
        Args:
            symbols: List of symbols to scan (uses config watchlist if None)
        
        Returns:
            List of trading signals
        """
        try:
            # Get watchlist from config if not provided
            if symbols is None:
                symbols = self.config.get("scanning", {}).get("watchlist", [])
            
            logger.info(f"Scanning {len(symbols)} symbols for signals")
            
            all_signals = []
            
            for symbol in symbols:
                try:
                    logger.debug(f"Scanning {symbol}...")
                    
                    # Get market data
                    stock_data = self.market_data.get_stock_data(symbol)
                    
                    if not stock_data:
                        logger.warning(f"No stock data for {symbol}")
                        continue
                    
                    # Apply basic filters
                    if not self._passes_basic_filters(symbol, stock_data):
                        logger.debug(f"{symbol} filtered out by basic criteria")
                        continue
                    
                    # Get options chain
                    options_chain = self.market_data.get_options_chain_enriched(
                        symbol, target_dte=35
                    )
                    
                    if not options_chain:
                        logger.warning(f"No options data for {symbol}")
                        continue
                    
                    # Generate signals from each strategy
                    for strategy_name, strategy in self.strategies.items():
                        try:
                            signals = strategy.generate_signals(
                                symbol, stock_data, options_chain
                            )
                            
                            if signals:
                                logger.info(f"Generated {len(signals)} {strategy_name} signals for {symbol}")
                                all_signals.extend([s.to_dict() for s in signals])
                        
                        except Exception as e:
                            logger.error(f"Error generating signals from {strategy_name} for {symbol}: {e}")
                
                except Exception as e:
                    logger.error(f"Error scanning {symbol}: {e}")
                    continue
            
            # Sort by signal quality
            all_signals.sort(key=lambda x: x["signal_quality"], reverse=True)
            
            logger.info(f"Generated {len(all_signals)} total signals")
            
            return all_signals
        
        except Exception as e:
            logger.error(f"Error in signal generation: {e}")
            return []
    
    def _passes_basic_filters(self, symbol: str, stock_data: Dict) -> bool:
        """Apply basic filtering criteria"""
        try:
            scanning_config = self.config.get("scanning", {})
            
            price = stock_data.get("price", 0)
            volume = stock_data.get("volume", 0)
            
            # Price range
            min_price = scanning_config.get("min_stock_price", 20)
            max_price = scanning_config.get("max_stock_price", 500)
            
            if not (min_price <= price <= max_price):
                return False
            
            # Volume
            min_volume = scanning_config.get("min_avg_volume", 1000000)
            if volume < min_volume:
                return False
            
            # Spread check
            spread_pct = stock_data.get("spread_pct", 0)
            if spread_pct > 1.0:  # More than 1% spread
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error in basic filtering: {e}")
            return False
    
    def get_best_signals(self, max_signals: int = 5) -> List[Dict]:
        """Get top N signals across all symbols"""
        signals = self.scan_for_signals()
        return signals[:max_signals]








