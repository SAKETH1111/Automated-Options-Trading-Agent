#!/usr/bin/env python3
"""
Test Polygon Options Data Integration
Verify we can fetch real Greeks, IV, and Open Interest
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from src.market_data.polygon_options import PolygonOptionsClient
from src.database.session import get_db
from src.ml.options_feature_engineer import OptionsFeatureEngineer

print("=" * 80)
print("🧪 TESTING POLYGON OPTIONS DATA")
print("=" * 80)
print()

# Test 1: Initialize client
print("Test 1: Initializing Polygon Options Client...")
try:
    client = PolygonOptionsClient()
    print("✅ Client initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize: {e}")
    sys.exit(1)

print()

# Test 2: Get options chain
print("Test 2: Fetching options chain for SPY...")
try:
    chain = client.get_options_chain(underlying='SPY')
    print(f"✅ Found {len(chain)} option contracts")
    
    if chain:
        print(f"\nSample contract:")
        print(f"  Ticker: {chain[0]['ticker']}")
        print(f"  Strike: ${chain[0]['strike']}")
        print(f"  Type: {chain[0]['contract_type']}")
        print(f"  Expiration: {chain[0]['expiration_date']}")
except Exception as e:
    print(f"⚠️  Options chain fetch failed: {e}")
    print("   (This might be OK if you need options-specific plan)")

print()

# Test 3: Get option snapshot with Greeks
print("Test 3: Fetching option snapshot with Greeks...")
try:
    if chain:
        sample_ticker = chain[0]['ticker']
        snapshot = client.get_option_snapshot(sample_ticker)
        
        if snapshot:
            print(f"✅ Got snapshot for {sample_ticker}")
            print(f"\nOption Data:")
            print(f"  Last Price: ${snapshot.get('last_price', 'N/A')}")
            print(f"  Bid/Ask: ${snapshot.get('bid', 'N/A')}/${snapshot.get('ask', 'N/A')}")
            print(f"  Volume: {snapshot.get('volume', 0):,}")
            print(f"  Open Interest: {snapshot.get('open_interest', 0):,}")
            print(f"  IV: {snapshot.get('implied_volatility', 'N/A')}")
            
            if snapshot.get('greeks'):
                print(f"\nGreeks:")
                print(f"  Delta: {snapshot['greeks'].get('delta', 'N/A')}")
                print(f"  Gamma: {snapshot['greeks'].get('gamma', 'N/A')}")
                print(f"  Theta: {snapshot['greeks'].get('theta', 'N/A')}")
                print(f"  Vega: {snapshot['greeks'].get('vega', 'N/A')}")
        else:
            print("⚠️  No snapshot data available")
    else:
        print("⚠️  Skipping (no chain data)")
except Exception as e:
    print(f"⚠️  Snapshot fetch failed: {e}")

print()

# Test 4: Options Feature Engineer
print("Test 4: Testing Options Feature Engineer...")
try:
    db = get_db()
    engineer = OptionsFeatureEngineer(db)
    print("✅ Options Feature Engineer initialized")
    
    # Create dummy DataFrame
    import pandas as pd
    df = pd.DataFrame({
        'timestamp': [pd.Timestamp.now()],
        'close': [450.0]
    })
    
    print("\nAdding options features to DataFrame...")
    df_with_options = engineer.add_options_features(df, 'SPY')
    
    new_cols = [col for col in df_with_options.columns if col not in df.columns]
    print(f"✅ Added {len(new_cols)} options features:")
    
    for col in new_cols[:10]:  # Show first 10
        val = df_with_options[col].iloc[0]
        print(f"  • {col}: {val}")
    
    if len(new_cols) > 10:
        print(f"  ... and {len(new_cols) - 10} more")
    
except Exception as e:
    print(f"⚠️  Feature engineering failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 80)
print("🎯 TEST SUMMARY")
print("=" * 80)
print()
print("If all tests passed:")
print("  ✅ Polygon options integration is working!")
print("  ✅ Ready to retrain ML models with options features")
print()
print("If some tests failed:")
print("  ⚠️  You might need options-specific Polygon plan")
print("  ⚠️  Or there may be API limitations")
print("  ℹ️  Models will use default features as fallback")
print()
print("Next step:")
print("  python3 scripts/train_ml_models.py")
print()

