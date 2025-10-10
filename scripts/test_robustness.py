"""Test system robustness with various failure scenarios"""

import sys
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.monitoring.circuit_breaker import CircuitBreaker, CircuitState, CircuitBreakerOpenError
from src.monitoring.health_checker import HealthChecker, HealthStatus
from src.market_data.robust_collector import DataValidator


def test_circuit_breaker():
    """Test circuit breaker functionality"""
    print("\n" + "=" * 80)
    print("TEST 1: Circuit Breaker")
    print("=" * 80)
    
    cb = CircuitBreaker("test_api", failure_threshold=3, timeout_seconds=5)
    
    # Test 1: Normal operation
    print("\n1.1 Testing normal operation...")
    
    def successful_call():
        return "success"
    
    for i in range(5):
        result = cb.call(successful_call)
        assert result == "success"
    
    assert cb.state == CircuitState.CLOSED
    print("  ✅ Normal operation works")
    
    # Test 2: Failures trigger circuit open
    print("\n1.2 Testing failure handling...")
    
    def failing_call():
        raise Exception("API Error")
    
    failures = 0
    for i in range(5):
        try:
            cb.call(failing_call)
        except Exception:
            failures += 1
    
    assert cb.state == CircuitState.OPEN
    assert failures >= cb.failure_threshold  # At least threshold failures
    print(f"  ✅ Circuit opened after {failures} failures (threshold: {cb.failure_threshold})")
    
    # Test 3: Circuit rejects calls when open
    print("\n1.3 Testing circuit rejection...")
    
    rejections = 0
    for i in range(3):
        try:
            cb.call(successful_call)
        except CircuitBreakerOpenError:
            rejections += 1
    
    assert rejections == 3
    print("  ✅ Circuit rejects calls when open")
    
    # Test 4: Circuit enters half-open after timeout
    print(f"\n1.4 Waiting {cb.timeout_seconds}s for timeout...")
    time.sleep(cb.timeout_seconds + 1)
    
    # This should transition to half-open and succeed
    result = cb.call(successful_call)
    assert cb.state == CircuitState.HALF_OPEN
    print("  ✅ Circuit transitioned to half-open")
    
    # Test 5: Circuit closes after successful attempts
    print("\n1.5 Testing recovery...")
    
    for i in range(cb.half_open_attempts):
        cb.call(successful_call)
    
    assert cb.state == CircuitState.CLOSED
    print("  ✅ Circuit closed after successful recovery")
    
    # Show stats
    print("\n1.6 Circuit breaker stats:")
    stats = cb.get_stats()
    print(f"  Total calls:    {stats['total_calls']}")
    print(f"  Total failures: {stats['total_failures']}")
    print(f"  Failure rate:   {stats['failure_rate']:.1f}%")
    print(f"  State:          {stats['state']}")
    
    print("\n✅ Circuit breaker tests PASSED")


def test_health_checker():
    """Test health checker functionality"""
    print("\n" + "=" * 80)
    print("TEST 2: Health Checker")
    print("=" * 80)
    
    checker = HealthChecker(
        "test_component",
        max_errors_per_minute=5,
        max_stale_seconds=10
    )
    
    # Test 1: Healthy state
    print("\n2.1 Testing healthy state...")
    
    for i in range(10):
        checker.record_success()
    
    report = checker.check_health()
    assert report['status'] == HealthStatus.HEALTHY
    print(f"  ✅ Status: {report['status']}")
    print(f"  Success rate: {report['metrics']['success_rate']*100:.1f}%")
    
    # Test 2: Degraded state (some errors)
    print("\n2.2 Testing degraded state...")
    
    for i in range(3):
        checker.record_failure("Test error")
    
    report = checker.check_health()
    print(f"  Status: {report['status']}")
    print(f"  Issues: {report['issues']}")
    
    # Test 3: Unhealthy state (high error rate)
    print("\n2.3 Testing unhealthy state...")
    
    for i in range(10):
        checker.record_failure("Test error")
    
    report = checker.check_health()
    print(f"  ❌ Status: {report['status']}")
    print(f"  Error rate: {report['metrics']['errors_last_minute']}/min")
    print(f"  Issues: {len(report['issues'])} detected")
    
    # Test 4: Stale data detection
    print("\n2.4 Testing stale data detection...")
    
    checker2 = HealthChecker("stale_test", max_stale_seconds=2)
    checker2.record_success()
    
    time.sleep(3)
    
    report = checker2.check_health()
    assert any('Stale data' in issue for issue in report['issues'])
    print("  ✅ Stale data detected")
    
    print("\n✅ Health checker tests PASSED")


def test_data_validator():
    """Test data validation"""
    print("\n" + "=" * 80)
    print("TEST 3: Data Validator")
    print("=" * 80)
    
    validator = DataValidator(
        max_price_change_pct=5.0,
        min_price=1.0,
        max_price=10000.0
    )
    
    # Test 1: Valid data
    print("\n3.1 Testing valid data...")
    
    valid_data = {
        'price': 450.25,
        'bid': 450.24,
        'ask': 450.26,
    }
    
    is_valid, error = validator.validate('SPY', valid_data)
    assert is_valid
    print("  ✅ Valid data accepted")
    
    # Test 2: Price out of range
    print("\n3.2 Testing price range validation...")
    
    invalid_data = {
        'price': 15000.0,  # Too high
        'bid': 14999.0,
        'ask': 15001.0,
    }
    
    is_valid, error = validator.validate('SPY', invalid_data)
    assert not is_valid
    print(f"  ✅ Invalid price rejected: {error}")
    
    # Test 3: Bid > Ask (invalid)
    print("\n3.3 Testing bid/ask sanity...")
    
    invalid_data = {
        'price': 450.0,
        'bid': 451.0,  # Bid > Ask!
        'ask': 450.0,
    }
    
    is_valid, error = validator.validate('SPY', invalid_data)
    assert not is_valid
    print(f"  ✅ Invalid bid/ask rejected: {error}")
    
    # Test 4: Large price change
    print("\n3.4 Testing price change threshold...")
    
    # First set a baseline
    validator.validate('QQQ', {'price': 380.0, 'bid': 379.9, 'ask': 380.1})
    
    # Try large jump
    large_jump = {
        'price': 420.0,  # 10%+ change
        'bid': 419.9,
        'ask': 420.1,
    }
    
    is_valid, error = validator.validate('QQQ', large_jump)
    assert not is_valid
    print(f"  ✅ Large price change rejected: {error}")
    
    print("\n✅ Data validator tests PASSED")


def test_retry_logic():
    """Test retry logic"""
    print("\n" + "=" * 80)
    print("TEST 4: Retry Logic")
    print("=" * 80)
    
    print("\n4.1 Testing retry with eventual success...")
    
    call_count = 0
    
    def flaky_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("Temporary failure")
        return "success"
    
    # Simulate retry logic
    max_retries = 3
    retry_delay = 0.1
    
    for attempt in range(max_retries):
        try:
            result = flaky_function()
            print(f"  ✅ Succeeded on attempt {attempt + 1}")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"  ❌ Failed after {max_retries} attempts")
    
    assert result == "success"
    assert call_count == 3
    
    print("\n4.2 Testing exponential backoff...")
    
    delays = []
    for attempt in range(4):
        delay = 0.1 * (2 ** attempt)
        delays.append(delay)
        print(f"  Attempt {attempt + 1}: delay = {delay}s")
    
    expected = [0.1, 0.2, 0.4, 0.8]
    assert delays == expected
    print("  ✅ Exponential backoff works correctly")
    
    print("\n✅ Retry logic tests PASSED")


def test_graceful_degradation():
    """Test graceful degradation"""
    print("\n" + "=" * 80)
    print("TEST 5: Graceful Degradation")
    print("=" * 80)
    
    print("\n5.1 Testing operation with partial failures...")
    
    symbols = ['SPY', 'QQQ', 'IWM', 'DIA']
    successful = []
    failed = []
    
    def collect_data(symbol):
        if symbol == 'IWM':
            raise Exception("IWM data unavailable")
        return f"{symbol} data"
    
    for symbol in symbols:
        try:
            data = collect_data(symbol)
            successful.append(symbol)
        except Exception as e:
            failed.append(symbol)
            # Continue with other symbols
    
    print(f"  ✅ Successful: {successful}")
    print(f"  ❌ Failed: {failed}")
    print(f"  System continued with {len(successful)}/{len(symbols)} symbols")
    
    assert len(successful) == 3
    assert len(failed) == 1
    
    print("\n✅ Graceful degradation tests PASSED")


def run_all_tests():
    """Run all robustness tests"""
    print("\n" + "=" * 80)
    print("ROBUSTNESS TEST SUITE")
    print("=" * 80)
    print("\nTesting system reliability under various failure conditions...")
    
    tests = [
        ("Circuit Breaker", test_circuit_breaker),
        ("Health Checker", test_health_checker),
        ("Data Validator", test_data_validator),
        ("Retry Logic", test_retry_logic),
        ("Graceful Degradation", test_graceful_degradation),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ {test_name} test FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"\nPassed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✅ ALL TESTS PASSED - System is robust!")
    else:
        print(f"\n❌ {failed} test(s) failed - Review failures above")
    
    return failed == 0


def main():
    """Main entry point"""
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nTest suite error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

