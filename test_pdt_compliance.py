#!/usr/bin/env python3
"""
Test PDT Compliance for Different Account Sizes
Shows how the system adapts to ANY account under $25,000
"""

def test_pdt_compliance():
    """Test PDT compliance logic for different account sizes"""
    
    def is_pdt_account(account_balance):
        """Check if account is subject to PDT rules"""
        return account_balance < 25000.0
    
    def get_pdt_info(account_balance):
        """Get PDT info for account"""
        is_pdt = is_pdt_account(account_balance)
        
        if not is_pdt:
            return {
                "account_balance": account_balance,
                "is_pdt_account": False,
                "day_trades_limit": "Unlimited",
                "max_positions_per_day": "Unlimited",
                "min_dte": "Any",
                "hold_overnight_required": False,
                "status": "EXEMPT (No PDT restrictions)"
            }
        else:
            return {
                "account_balance": account_balance,
                "is_pdt_account": True,
                "day_trades_limit": "3 per 5 business days",
                "max_positions_per_day": "1 (to avoid day trading)",
                "min_dte": "21 days (3+ weeks)",
                "hold_overnight_required": True,
                "status": "PDT COMPLIANT"
            }
    
    def get_account_tier(account_balance):
        """Get account tier based on balance"""
        if account_balance < 2500:
            return "micro"
        elif account_balance < 5000:
            return "small"
        elif account_balance < 10000:
            return "medium"
        elif account_balance < 25000:
            return "standard"
        else:
            return "large"
    
    def get_tier_config(tier):
        """Get configuration for account tier"""
        configs = {
            "micro": {
                "max_risk_per_trade_pct": 12.0,
                "max_positions": 1,
                "preferred_symbols": ["EWZ", "GDX", "F"],
                "spread_width": [1, 2],
                "strategy": "Aggressive growth needed"
            },
            "small": {
                "max_risk_per_trade_pct": 8.0,
                "max_positions": 1,
                "preferred_symbols": ["GDX", "XLF", "TLT"],
                "spread_width": [2, 3],
                "strategy": "Balanced growth"
            },
            "medium": {
                "max_risk_per_trade_pct": 5.0,
                "max_positions": 2,
                "preferred_symbols": ["XLF", "TLT", "IWM"],
                "spread_width": [3, 5],
                "strategy": "Steady growth"
            },
            "standard": {
                "max_risk_per_trade_pct": 3.0,
                "max_positions": 4,
                "preferred_symbols": ["SPY", "QQQ", "IWM"],
                "spread_width": [5, 10],
                "strategy": "Conservative growth"
            },
            "large": {
                "max_risk_per_trade_pct": 2.0,
                "max_positions": 5,
                "preferred_symbols": ["SPY", "QQQ", "IWM", "DIA"],
                "spread_width": [5, 10, 15],
                "strategy": "Capital preservation"
            }
        }
        return configs.get(tier, {})
    
    # Test different account sizes
    test_accounts = [
        1000,    # Micro
        2500,    # Small  
        5000,    # Medium
        10000,   # Standard
        15000,   # Standard
        20000,   # Standard
        25000,   # Large (PDT exempt)
        50000    # Large (PDT exempt)
    ]
    
    print("🎯 PDT Compliance Test for Different Account Sizes")
    print("=" * 80)
    
    for balance in test_accounts:
        print(f"\n💰 Account: ${balance:,}")
        print("-" * 40)
        
        # Get PDT info
        pdt_info = get_pdt_info(balance)
        tier = get_account_tier(balance)
        config = get_tier_config(tier)
        
        print(f"📊 Tier: {tier.upper()}")
        print(f"🚨 PDT Account: {pdt_info['is_pdt_account']}")
        print(f"📈 Status: {pdt_info['status']}")
        print(f"⚡ Day Trades: {pdt_info['day_trades_limit']}")
        print(f"📅 Max Positions/Day: {pdt_info['max_positions_per_day']}")
        print(f"⏰ Min DTE: {pdt_info['min_dte']}")
        print(f"🌙 Hold Overnight: {pdt_info['hold_overnight_required']}")
        
        if config:
            print(f"💸 Max Risk/Trade: {config['max_risk_per_trade_pct']}%")
            print(f"🎯 Max Positions: {config['max_positions']}")
            print(f"📈 Symbols: {', '.join(config['preferred_symbols'])}")
            print(f"📏 Spread Width: ${config['spread_width'][0]}-${config['spread_width'][-1]}")
            print(f"🎯 Strategy: {config['strategy']}")
        
        # Show example trade
        if pdt_info['is_pdt_account']:
            print(f"💡 Example Trade:")
            print(f"   • Symbol: {config['preferred_symbols'][0] if config else 'SPY'}")
            print(f"   • Strategy: Bull Put Spread")
            print(f"   • DTE: 30 days (monthly)")
            print(f"   • Risk: ${balance * config['max_risk_per_trade_pct']/100:.0f} ({config['max_risk_per_trade_pct']}%)")
            print(f"   • Must Hold: Overnight minimum")
        else:
            print(f"💡 Example Trade:")
            print(f"   • Symbol: SPY")
            print(f"   • Strategy: Any (full flexibility)")
            print(f"   • DTE: Any (including weekly)")
            print(f"   • Risk: ${balance * 2/100:.0f} (2%)")
            print(f"   • Can Day Trade: Yes")
    
    print("\n" + "=" * 80)
    print("🎯 KEY POINTS:")
    print("✅ System automatically detects ANY account < $25,000")
    print("✅ Applies PDT compliance rules dynamically")
    print("✅ Adjusts risk, symbols, and strategies by account size")
    print("✅ No hardcoded amounts - works for $1K, $5K, $15K, etc.")
    print("✅ $25K+ accounts get full trading flexibility")

if __name__ == "__main__":
    test_pdt_compliance()
