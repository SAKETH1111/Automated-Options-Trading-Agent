"""
Smart Symbol Selector
Automatically chooses appropriate symbols based on account size
"""

from typing import List
from loguru import logger

from config.adaptive_account_config import get_adaptive_config


def get_symbols_for_account(account_balance: float) -> List[str]:
    """
    Get appropriate trading symbols based on account size
    
    Args:
        account_balance: Current account equity
        
    Returns:
        List of symbol strings appropriate for the account size
    """
    config = get_adaptive_config(account_balance)
    account_tier = config['account_tier']
    
    # Get symbols for this tier
    adaptive_symbols = config['scanning']['adaptive_symbols']
    tier_key = f'{account_tier}_tier'
    
    symbols = adaptive_symbols.get(tier_key, [])
    
    logger.info(f"Account: ${account_balance:,.2f} | Tier: {account_tier} | Symbols: {symbols}")
    
    return symbols


def get_symbol_info(account_balance: float) -> dict:
    """
    Get detailed symbol selection info
    
    Returns:
        Dictionary with tier info, symbols, and constraints
    """
    config = get_adaptive_config(account_balance)
    account_tier = config['account_tier']
    
    adaptive_symbols = config['scanning']['adaptive_symbols']
    tier_key = f'{account_tier}_tier'
    
    # Get tier-specific info
    tier_config = {}
    for key, value in adaptive_symbols.items():
        if key == tier_key:
            if isinstance(value, list):
                tier_config['symbols'] = value
            else:
                tier_config = value
                break
    
    return {
        'account_balance': account_balance,
        'tier': account_tier,
        'symbols': tier_config.get('symbols', adaptive_symbols.get(tier_key, [])),
        'max_stock_price': tier_config.get('max_stock_price', 'N/A'),
        'preferred_spread_width': tier_config.get('preferred_spread_width', 'N/A'),
        'note': tier_config.get('note', '')
    }


def should_trade_symbol(symbol: str, account_balance: float) -> bool:
    """
    Check if a symbol is appropriate for the account size
    
    Args:
        symbol: Stock symbol to check
        account_balance: Current account equity
        
    Returns:
        True if symbol is appropriate, False otherwise
    """
    allowed_symbols = get_symbols_for_account(account_balance)
    return symbol in allowed_symbols

