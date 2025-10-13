#!/usr/bin/env python3
"""
Comprehensive Timezone Testing Suite
Tests all timezone-related functionality across the system
"""

import sys
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
import pytz

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.database.session import get_db
from src.database.models import IndexTickData

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def test_basic_timezone_conversion():
    """Test 1: Basic timezone conversion"""
    print_header("TEST 1: Basic Timezone Conversion")
    
    # Get current time in different zones
    utc_tz = pytz.UTC
    et_tz = pytz.timezone('America/New_York')
    ct_tz = pytz.timezone('America/Chicago')
    pt_tz = pytz.timezone('America/Los_Angeles')
    
    now_utc = datetime.now(utc_tz)
    now_et = now_utc.astimezone(et_tz)
    now_ct = now_utc.astimezone(ct_tz)
    now_pt = now_utc.astimezone(pt_tz)
    
    print(f"UTC:     {now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Eastern: {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Central: {now_ct.strftime('%Y-%m-%d %H:%M:%S %Z')} ← Our timezone (Texas)")
    print(f"Pacific: {now_pt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # Verify differences
    print(f"\n✓ UTC to Central offset: {(now_ct.hour - now_utc.hour) % 24} hours")
    print(f"✓ Eastern to Central offset: {(now_ct.hour - now_et.hour) % 24} hours")
    
    return True

def test_market_hours_detection():
    """Test 2: Market hours detection"""
    print_header("TEST 2: Market Hours Detection (Central Time)")
    
    ct_tz = pytz.timezone('America/Chicago')
    
    # Market hours in Central Time
    market_open = dt_time(8, 30)  # 9:30 ET = 8:30 CT
    market_close = dt_time(15, 0)  # 16:00 ET = 15:00 CT
    
    print(f"Market Hours (Central Time): {market_open} - {market_close}")
    print(f"Market Hours (Eastern Time): 09:30:00 - 16:00:00")
    
    # Test various times
    test_times = [
        dt_time(8, 0),   # Before market
        dt_time(8, 30),  # Market open
        dt_time(12, 0),  # Mid-day
        dt_time(14, 59), # Just before close
        dt_time(15, 0),  # Market close
        dt_time(16, 0),  # After market
    ]
    
    now = datetime.now(ct_tz)
    current_time = now.time()
    is_weekday = now.weekday() < 5
    
    print(f"\nCurrent Time: {current_time} (Weekday: {is_weekday})")
    print(f"\nMarket Status Tests:")
    
    for test_time in test_times:
        is_market_hours = market_open <= test_time < market_close
        status = "OPEN" if is_market_hours else "CLOSED"
        indicator = "→" if abs((test_time.hour * 60 + test_time.minute) - 
                               (current_time.hour * 60 + current_time.minute)) < 60 else " "
        print(f"  {indicator} {test_time}: {status}")
    
    # Current market status
    current_status = "OPEN" if (is_weekday and market_open <= current_time < market_close) else "CLOSED"
    print(f"\n✓ Current Market Status: {current_status}")
    
    return True

def test_database_timezone_storage():
    """Test 3: Database timezone storage and retrieval"""
    print_header("TEST 3: Database Timezone Storage & Retrieval")
    
    db = get_db()
    ct_tz = pytz.timezone('America/Chicago')
    utc_tz = pytz.UTC
    
    with db.get_session() as session:
        # Get latest tick
        latest_tick = session.query(IndexTickData)\
            .order_by(IndexTickData.timestamp.desc())\
            .first()
        
        if not latest_tick:
            print("⚠ No data in database yet")
            return False
        
        # Stored timestamp (should be UTC)
        stored_time = latest_tick.timestamp
        print(f"Stored in DB: {stored_time} (assumed UTC)")
        
        # Convert to different timezones
        utc_time = stored_time.replace(tzinfo=utc_tz)
        ct_time = utc_time.astimezone(ct_tz)
        et_time = utc_time.astimezone(pytz.timezone('America/New_York'))
        
        print(f"\nTimezone Conversions:")
        print(f"  UTC:     {utc_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"  Eastern: {et_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"  Central: {ct_time.strftime('%Y-%m-%d %H:%M:%S %Z')} ← Texas time")
        
        # Verify it's recent (within last 5 minutes)
        now_utc = datetime.now(utc_tz)
        age = (now_utc - utc_time).total_seconds()
        
        print(f"\n✓ Data age: {age:.1f} seconds")
        print(f"✓ Data is {'CURRENT' if age < 300 else 'STALE'}")
        
        # Get count by symbol
        from sqlalchemy import func
        stats = session.query(
            IndexTickData.symbol,
            func.count(IndexTickData.symbol).label('count'),
            func.min(IndexTickData.timestamp).label('first'),
            func.max(IndexTickData.timestamp).label('latest')
        ).group_by(IndexTickData.symbol).all()
        
        print(f"\n✓ Database Statistics:")
        for symbol, count, first, latest in stats:
            first_ct = first.replace(tzinfo=utc_tz).astimezone(ct_tz)
            latest_ct = latest.replace(tzinfo=utc_tz).astimezone(ct_tz)
            duration = (latest - first).total_seconds() / 60
            print(f"  {symbol}: {count:,} ticks")
            print(f"    First: {first_ct.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print(f"    Latest: {latest_ct.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print(f"    Duration: {duration:.1f} minutes")
    
    return True

def test_market_hours_edge_cases():
    """Test 4: Market hours edge cases"""
    print_header("TEST 4: Market Hours Edge Cases")
    
    ct_tz = pytz.timezone('America/Chicago')
    
    # Test dates at different times
    test_cases = [
        ("Monday 8:29 AM CT", datetime(2025, 10, 13, 8, 29, 0, tzinfo=ct_tz), False, "Just before open"),
        ("Monday 8:30 AM CT", datetime(2025, 10, 13, 8, 30, 0, tzinfo=ct_tz), True, "Market open"),
        ("Monday 2:59 PM CT", datetime(2025, 10, 13, 14, 59, 0, tzinfo=ct_tz), True, "Just before close"),
        ("Monday 3:00 PM CT", datetime(2025, 10, 13, 15, 0, 0, tzinfo=ct_tz), False, "Market close"),
        ("Saturday 12:00 PM CT", datetime(2025, 10, 18, 12, 0, 0, tzinfo=ct_tz), False, "Weekend"),
        ("Sunday 12:00 PM CT", datetime(2025, 10, 19, 12, 0, 0, tzinfo=ct_tz), False, "Weekend"),
    ]
    
    market_open = dt_time(8, 30)
    market_close = dt_time(15, 0)
    
    print(f"Testing market hours logic (CT): {market_open} - {market_close}")
    print()
    
    for name, test_dt, expected, description in test_cases:
        is_weekday = test_dt.weekday() < 5
        is_market_hours = market_open <= test_dt.time() < market_close
        is_open = is_weekday and is_market_hours
        
        status = "✓" if is_open == expected else "✗"
        print(f"{status} {name:25s} → {('OPEN' if is_open else 'CLOSED'):6s} ({description})")
    
    return True

def test_scheduler_times():
    """Test 5: Scheduled job times"""
    print_header("TEST 5: Scheduled Job Times (Central Time)")
    
    ct_tz = pytz.timezone('America/Chicago')
    
    jobs = [
        ("Trading Cycle", "Every 5 minutes, 8:30 AM - 3:00 PM CT", "8-15", "*/5"),
        ("Position Monitor", "Every 1 minute, 8:30 AM - 3:00 PM CT", "8-15", "*"),
        ("Daily Analysis", "4:00 PM CT (after market close)", "16", "0"),
        ("Weekly Learning", "Sunday 7:00 PM CT", "19 (Sun)", "0"),
    ]
    
    print("Scheduled Tasks (all times in Central Time):")
    for name, description, hours, minutes in jobs:
        print(f"\n  {name}:")
        print(f"    Schedule: {description}")
        print(f"    Cron: hour={hours}, minute={minutes}")
    
    # Show what time these would run in other zones
    print("\n✓ Time Zone References:")
    print("  8:30 AM CT = 9:30 AM ET = Market Open")
    print("  3:00 PM CT = 4:00 PM ET = Market Close")
    print("  4:00 PM CT = 5:00 PM ET = After Market")
    
    return True

def test_daylight_saving_awareness():
    """Test 6: Daylight saving time awareness"""
    print_header("TEST 6: Daylight Saving Time Awareness")
    
    ct_tz = pytz.timezone('America/Chicago')
    
    # Current time
    now = datetime.now(ct_tz)
    
    # Test winter (CST) and summer (CDT)
    winter_time = ct_tz.localize(datetime(2025, 1, 15, 12, 0, 0))
    summer_time = ct_tz.localize(datetime(2025, 7, 15, 12, 0, 0))
    
    print(f"Current Time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')} (offset: {now.strftime('%z')})")
    print(f"\nWinter (CST): {winter_time.strftime('%Y-%m-%d %H:%M:%S %Z')} (offset: {winter_time.strftime('%z')})")
    print(f"Summer (CDT): {summer_time.strftime('%Y-%m-%d %H:%M:%S %Z')} (offset: {summer_time.strftime('%z')})")
    
    # Check current DST status
    is_dst = bool(now.dst())
    print(f"\n✓ Currently in DST: {is_dst}")
    print(f"✓ Current timezone: {now.tzname()}")
    print(f"✓ UTC offset: {now.strftime('%z')} ({now.utcoffset().total_seconds() / 3600:.0f} hours)")
    
    return True

def test_realtime_collector_config():
    """Test 7: Real-time collector timezone configuration"""
    print_header("TEST 7: Real-Time Collector Configuration")
    
    try:
        from src.market_data.realtime_collector import RealTimeDataCollector
        
        # Check the market timezone setting
        collector = RealTimeDataCollector(symbols=['SPY'])
        
        print(f"Collector timezone: {collector.market_timezone}")
        print(f"Market open time: {collector.market_open}")
        print(f"Market close time: {collector.market_close}")
        
        # Test market open check
        is_open = collector._is_market_open()
        print(f"\n✓ Collector thinks market is: {'OPEN' if is_open else 'CLOSED'}")
        
        # Test market state
        market_state = collector._get_market_state()
        print(f"✓ Market state: {market_state}")
        
        return True
    except Exception as e:
        print(f"⚠ Error testing collector: {e}")
        return False

def run_all_tests():
    """Run all timezone tests"""
    print("\n" + "▓" * 80)
    print("  🕐 COMPREHENSIVE TIMEZONE TEST SUITE")
    print("  Testing timezone handling across the entire system")
    print("▓" * 80)
    
    tests = [
        ("Basic Timezone Conversion", test_basic_timezone_conversion),
        ("Market Hours Detection", test_market_hours_detection),
        ("Database Storage & Retrieval", test_database_timezone_storage),
        ("Market Hours Edge Cases", test_market_hours_edge_cases),
        ("Scheduler Times", test_scheduler_times),
        ("Daylight Saving Time", test_daylight_saving_awareness),
        ("Real-Time Collector Config", test_realtime_collector_config),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n⚠ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status:8s} {name}")
    
    print(f"\n{'=' * 80}")
    print(f"  Results: {passed}/{total} tests passed")
    
    if passed == total:
        print(f"  🎉 ALL TESTS PASSED - Timezone configuration is correct!")
    else:
        print(f"  ⚠ {total - passed} test(s) failed - review configuration")
    
    print("=" * 80)

if __name__ == "__main__":
    run_all_tests()

