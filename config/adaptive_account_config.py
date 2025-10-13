"""
Adaptive Account Configuration Loader
Loads and provides account-specific configuration based on account size
"""

import os
import yaml
from typing import Dict


def get_adaptive_config(account_balance: float) -> Dict:
    """
    Get adaptive configuration based on account balance
    
    Args:
        account_balance: Current account equity
        
    Returns:
        Dictionary with account-specific configuration
    """
    # Load the YAML config
    config_path = os.path.join(os.path.dirname(__file__), 'adaptive_account_config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Determine account tier
    if account_balance < 2500:
        tier = 'micro'
    elif account_balance < 5000:
        tier = 'small'
    elif account_balance < 10000:
        tier = 'medium'
    elif account_balance < 25000:
        tier = 'standard'
    else:
        tier = 'large'
    
    # Add tier to config
    config['account_tier'] = tier
    config['account_balance'] = account_balance
    
    return config


def get_account_tier(account_balance: float) -> str:
    """
    Get the account tier name based on balance
    
    Args:
        account_balance: Current account equity
        
    Returns:
        Tier name (micro, small, medium, standard, large)
    """
    if account_balance < 2500:
        return 'micro'
    elif account_balance < 5000:
        return 'small'
    elif account_balance < 10000:
        return 'medium'
    elif account_balance < 25000:
        return 'standard'
    else:
        return 'large'

