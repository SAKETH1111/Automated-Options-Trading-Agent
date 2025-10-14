#!/usr/bin/env python3
"""
Test Signal Generation
Verify the signal generator can create signals with Greeks and IV
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from src.brokers.alpaca_client import AlpacaClient
from src.market_data.collector import MarketDataCollector
from src.signals.generator import SignalGenerator
from src.config.settings import get_config
from src.utils.symbol_selector import get_symbols_for_account


def test_signal_generation():
    """Test signal generation end-to-end"""
    print("=" * 80)
    print("🧪 Testing Signal Generation")
    print("=" * 80)
    print()
    
    try:
        # Initialize components
        print("1. Initializing components...")
        alpaca = AlpacaClient()
        market_data = MarketDataCollector(alpaca)
        config = get_config()
        
        print("✅ Components initialized")
        print()
        
        # Get account and symbols
        print("2. Getting account info...")
        try:
            account = alpaca.get_account()
            equity = float(account.get('equity', 3000))
            print(f"✅ Account equity: ${equity:,.2f}")
        except Exception as e:
            print(f"⚠️  Could not get account (using default $3,000): {e}")
            equity = 3000
        print()
        
        # Get smart symbols
        print("3. Selecting symbols...")
        symbols = get_symbols_for_account(equity)
        print(f"✅ Symbols for ${equity:,.2f} account: {', '.join(symbols)}")
        print()
        
        # Initialize signal generator
        print("4. Initializing signal generator...")
        signal_gen = SignalGenerator(market_data, config)
        print(f"✅ Signal generator ready with {len(signal_gen.strategies)} strategies")
        print()
        
        # Test stock data fetching
        print("5. Testing stock data fetching...")
        test_symbol = symbols[0]
        stock_data = market_data.get_stock_data(test_symbol)
        
        if stock_data:
            print(f"✅ Got stock data for {test_symbol}:")
            print(f"   Price: ${stock_data['price']:.2f}")
            print(f"   Spread: {stock_data['spread_pct']:.2f}%")
            print(f"   Volume: {stock_data['volume']:,}")
            print(f"   IV Rank: {stock_data['iv_rank']:.0f}")
        else:
            print(f"⚠️  No stock data for {test_symbol}")
        print()
        
        # Test options chain fetching
        print("6. Testing options chain fetching...")
        options_chain = market_data.get_options_chain_enriched(test_symbol, target_dte=35)
        
        if options_chain:
            print(f"✅ Got {len(options_chain)} options for {test_symbol}")
            # Show first option
            opt = options_chain[0]
            print(f"   Example: Strike ${opt['strike']:.0f}, Delta: {opt['delta']:.2f}, "
                  f"IV: {opt['implied_volatility']:.2%}, Mid: ${opt['mid']:.2f}")
        else:
            print(f"⚠️  No options chain for {test_symbol}")
        print()
        
        # Test signal generation
        print("7. Testing signal generation...")
        signals = signal_gen.scan_for_signals(symbols)
        
        if signals:
            print(f"✅ Generated {len(signals)} signals!")
            print()
            
            # Show top signal
            for i, signal in enumerate(signals[:3], 1):
                print(f"Signal {i}: {signal['symbol']} - {signal['strategy_name']}")
                print(f"   Quality: {signal['signal_quality']:.1f}/100")
                print(f"   Max Profit: ${signal['max_profit']:.2f}")
                print(f"   Max Loss: ${signal['max_loss']:.2f}")
                print(f"   P(Profit): {signal['probability_of_profit']*100:.1f}%")
                print(f"   Risk:Reward: {signal['risk_reward_ratio']:.2f}:1")
                print(f"   Reason: {signal['reason']}")
                print()
        else:
            print("ℹ️  No signals generated (this is OK - may not meet criteria)")
            print("   Reasons could be:")
            print("   • IV Rank too low")
            print("   • No options in target DTE range")
            print("   • Liquidity too low")
            print("   • Risk:reward not favorable")
        print()
        
        print("=" * 80)
        print("✅ SIGNAL GENERATION TEST COMPLETE")
        print("=" * 80)
        print()
        print("Next steps:")
        print("  • Review signal quality scores")
        print("  • Verify Greeks calculations")
        print("  • Test with multiple symbols")
        print("  • Enable paper trade execution")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    test_signal_generation()

