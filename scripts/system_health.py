"""Monitor system health in real-time"""

import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.monitoring.health_checker import get_health_monitor, HealthStatus


def print_health_report(report: dict):
    """Print formatted health report"""
    print("\n" + "=" * 80)
    print(f"SYSTEM HEALTH CHECK - {report['timestamp']}")
    print("=" * 80)
    
    # Overall status
    status = report['overall_status']
    status_emoji = {
        HealthStatus.HEALTHY: "✅",
        HealthStatus.DEGRADED: "⚠️",
        HealthStatus.UNHEALTHY: "❌",
        HealthStatus.CRITICAL: "🔴"
    }
    
    emoji = status_emoji.get(status, "❓")
    print(f"\n{emoji} Overall Status: {status.upper()}")
    
    # Component details
    print("\n" + "-" * 80)
    print("COMPONENT HEALTH")
    print("-" * 80)
    
    for component_name, component_report in report['components'].items():
        status = component_report['status']
        emoji = status_emoji.get(status, "❓")
        
        print(f"\n{emoji} {component_name}: {status}")
        
        # Metrics
        metrics = component_report.get('metrics', {})
        if metrics:
            print(f"  Operations:    {metrics.get('total_operations', 0):,}")
            print(f"  Successes:     {metrics.get('total_successes', 0):,}")
            print(f"  Failures:      {metrics.get('total_failures', 0):,}")
            print(f"  Success Rate:  {metrics.get('success_rate', 0)*100:.1f}%")
            print(f"  Errors/minute: {metrics.get('errors_last_minute', 0)}")
            
            staleness = metrics.get('staleness_seconds')
            if staleness is not None:
                print(f"  Data Age:      {staleness:.0f}s")
        
        # Issues
        issues = component_report.get('issues', [])
        if issues:
            print(f"  Issues:")
            for issue in issues:
                print(f"    • {issue}")
    
    print("\n" + "=" * 80)


def monitor_continuously(interval: int = 30):
    """Monitor health continuously"""
    print("=" * 80)
    print("CONTINUOUS HEALTH MONITORING")
    print("=" * 80)
    print(f"Checking every {interval} seconds...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            health_monitor = get_health_monitor()
            report = health_monitor.check_all()
            print_health_report(report)
            
            # Wait for next check
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")


def check_once():
    """Run single health check"""
    health_monitor = get_health_monitor()
    
    # Check if any components registered
    if not health_monitor.components:
        print("\n⚠️  No components registered for monitoring")
        print("\nTo monitor health:")
        print("  1. Start the trading agent: python main.py")
        print("  2. Wait a few seconds for initialization")
        print("  3. Run this script again")
        return
    
    report = health_monitor.check_all()
    print_health_report(report)
    
    # Show unhealthy components
    unhealthy = health_monitor.get_unhealthy_components()
    if unhealthy:
        print("\n⚠️  UNHEALTHY COMPONENTS:")
        for component in unhealthy:
            print(f"  • {component}")
        print("\nCheck logs for details: tail -f logs/trading_agent.log")
    else:
        print("\n✅ All systems healthy!")


def show_component_history(component_name: str, limit: int = 10):
    """Show health history for a component"""
    health_monitor = get_health_monitor()
    checker = health_monitor.get_component(component_name)
    
    if not checker:
        print(f"\n❌ Component '{component_name}' not found")
        print(f"\nAvailable components: {list(health_monitor.components.keys())}")
        return
    
    print("\n" + "=" * 80)
    print(f"HEALTH HISTORY: {component_name}")
    print("=" * 80)
    
    history = checker.get_health_history(limit)
    
    if not history:
        print("\nNo health history available")
        return
    
    for report in history:
        timestamp = report['timestamp']
        status = report['status']
        
        status_emoji = {
            HealthStatus.HEALTHY: "✅",
            HealthStatus.DEGRADED: "⚠️",
            HealthStatus.UNHEALTHY: "❌",
            HealthStatus.CRITICAL: "🔴"
        }
        emoji = status_emoji.get(status, "❓")
        
        print(f"\n{timestamp} - {emoji} {status}")
        
        issues = report.get('issues', [])
        if issues:
            for issue in issues:
                print(f"  • {issue}")


def show_recent_errors(component_name: str, limit: int = 10):
    """Show recent errors for a component"""
    health_monitor = get_health_monitor()
    checker = health_monitor.get_component(component_name)
    
    if not checker:
        print(f"\n❌ Component '{component_name}' not found")
        return
    
    print("\n" + "=" * 80)
    print(f"RECENT ERRORS: {component_name}")
    print("=" * 80)
    
    errors = checker.get_recent_errors(limit)
    
    if not errors:
        print("\n✅ No recent errors!")
        return
    
    for error in errors:
        timestamp = error['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        error_msg = error['error']
        print(f"\n{timestamp}")
        print(f"  {error_msg}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor system health")
    parser.add_argument(
        '--mode',
        choices=['once', 'continuous', 'history', 'errors'],
        default='once',
        help='Monitoring mode'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=30,
        help='Check interval for continuous mode (seconds)'
    )
    parser.add_argument(
        '--component',
        type=str,
        help='Component name for history/errors mode'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=10,
        help='Number of records to show'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'once':
        check_once()
    
    elif args.mode == 'continuous':
        monitor_continuously(args.interval)
    
    elif args.mode == 'history':
        if not args.component:
            print("❌ --component required for history mode")
            return
        show_component_history(args.component, args.limit)
    
    elif args.mode == 'errors':
        if not args.component:
            print("❌ --component required for errors mode")
            return
        show_recent_errors(args.component, args.limit)


if __name__ == "__main__":
    main()

