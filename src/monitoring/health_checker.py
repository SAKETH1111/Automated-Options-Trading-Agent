"""Health monitoring for real-time data collection system"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import deque

from loguru import logger


class HealthStatus:
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class HealthChecker:
    """Monitor system health and detect issues"""
    
    def __init__(
        self,
        name: str = "DataCollection",
        max_errors_per_minute: int = 10,
        max_stale_seconds: int = 300,
        min_success_rate: float = 0.90
    ):
        """
        Initialize health checker
        
        Args:
            name: Component name
            max_errors_per_minute: Error threshold
            max_stale_seconds: Maximum data staleness
            min_success_rate: Minimum acceptable success rate
        """
        self.name = name
        self.max_errors_per_minute = max_errors_per_minute
        self.max_stale_seconds = max_stale_seconds
        self.min_success_rate = min_success_rate
        
        # Track recent errors (last 60 seconds)
        self.recent_errors: deque = deque(maxlen=100)
        
        # Track operations
        self.total_operations = 0
        self.total_successes = 0
        self.total_failures = 0
        
        # Track data freshness
        self.last_successful_collection: Optional[datetime] = None
        
        # Health history
        self.health_checks: List[Dict] = []
        
        # Issues detected
        self.current_issues: List[str] = []
        
        logger.info(f"HealthChecker initialized for '{name}'")
    
    def record_success(self):
        """Record a successful operation"""
        self.total_operations += 1
        self.total_successes += 1
        self.last_successful_collection = datetime.now()
    
    def record_failure(self, error: str):
        """Record a failed operation"""
        self.total_operations += 1
        self.total_failures += 1
        self.recent_errors.append({
            'timestamp': datetime.now(),
            'error': str(error)
        })
        
        logger.warning(f"Health check recorded failure: {error}")
    
    def check_health(self) -> Dict:
        """
        Perform comprehensive health check
        
        Returns:
            Health status dict
        """
        now = datetime.now()
        issues = []
        
        # Check 1: Error rate
        errors_last_minute = self._count_recent_errors(60)
        if errors_last_minute > self.max_errors_per_minute:
            issues.append(f"High error rate: {errors_last_minute} errors/minute")
        
        # Check 2: Success rate
        success_rate = self._calculate_success_rate()
        if success_rate < self.min_success_rate:
            issues.append(f"Low success rate: {success_rate*100:.1f}%")
        
        # Check 3: Data staleness
        if self.last_successful_collection:
            staleness = (now - self.last_successful_collection).total_seconds()
            if staleness > self.max_stale_seconds:
                issues.append(f"Stale data: {staleness:.0f}s since last collection")
        else:
            issues.append("No data collected yet")
        
        # Determine overall status
        status = self._determine_status(issues)
        
        health_report = {
            'component': self.name,
            'status': status,
            'timestamp': now.isoformat(),
            'metrics': {
                'total_operations': self.total_operations,
                'total_successes': self.total_successes,
                'total_failures': self.total_failures,
                'success_rate': success_rate,
                'errors_last_minute': errors_last_minute,
                'staleness_seconds': (now - self.last_successful_collection).total_seconds() if self.last_successful_collection else None,
            },
            'issues': issues,
        }
        
        # Store in history
        self.health_checks.append(health_report)
        if len(self.health_checks) > 1000:
            self.health_checks = self.health_checks[-1000:]
        
        self.current_issues = issues
        
        # Log if unhealthy
        if status != HealthStatus.HEALTHY:
            logger.warning(f"Health check for '{self.name}': {status} - Issues: {issues}")
        
        return health_report
    
    def _count_recent_errors(self, seconds: int) -> int:
        """Count errors in recent time window"""
        cutoff = datetime.now() - timedelta(seconds=seconds)
        return sum(1 for err in self.recent_errors if err['timestamp'] > cutoff)
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate"""
        if self.total_operations == 0:
            return 1.0
        
        return self.total_successes / self.total_operations
    
    def _determine_status(self, issues: List[str]) -> str:
        """Determine overall health status"""
        if not issues:
            return HealthStatus.HEALTHY
        
        # Critical if no data collected
        if any('No data collected' in issue for issue in issues):
            return HealthStatus.CRITICAL
        
        # Critical if very stale data
        if any('Stale data' in issue and 'since last collection' in issue for issue in issues):
            staleness_str = [i for i in issues if 'Stale data' in i][0]
            staleness = float(staleness_str.split(':')[1].split('s')[0])
            if staleness > self.max_stale_seconds * 2:
                return HealthStatus.CRITICAL
        
        # Unhealthy if multiple issues
        if len(issues) >= 2:
            return HealthStatus.UNHEALTHY
        
        # Degraded if single issue
        return HealthStatus.DEGRADED
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict]:
        """Get recent errors"""
        errors = list(self.recent_errors)
        return errors[-limit:]
    
    def get_health_history(self, limit: int = 10) -> List[Dict]:
        """Get recent health checks"""
        return self.health_checks[-limit:]
    
    def is_healthy(self) -> bool:
        """Check if system is healthy"""
        report = self.check_health()
        return report['status'] == HealthStatus.HEALTHY
    
    def reset(self):
        """Reset health checker"""
        self.total_operations = 0
        self.total_successes = 0
        self.total_failures = 0
        self.recent_errors.clear()
        self.current_issues.clear()
        
        logger.info(f"HealthChecker '{self.name}' reset")


class SystemHealthMonitor:
    """Monitor health of multiple components"""
    
    def __init__(self):
        self.components: Dict[str, HealthChecker] = {}
        
    def register_component(
        self,
        name: str,
        max_errors_per_minute: int = 10,
        max_stale_seconds: int = 300,
        min_success_rate: float = 0.90
    ) -> HealthChecker:
        """Register a component for health monitoring"""
        checker = HealthChecker(
            name, max_errors_per_minute, max_stale_seconds, min_success_rate
        )
        self.components[name] = checker
        return checker
    
    def get_component(self, name: str) -> Optional[HealthChecker]:
        """Get health checker for a component"""
        return self.components.get(name)
    
    def check_all(self) -> Dict:
        """Check health of all components"""
        reports = {}
        
        for name, checker in self.components.items():
            reports[name] = checker.check_health()
        
        # Determine overall system status
        statuses = [r['status'] for r in reports.values()]
        
        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall_status = HealthStatus.HEALTHY
        elif any(s == HealthStatus.CRITICAL for s in statuses):
            overall_status = HealthStatus.CRITICAL
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall_status = HealthStatus.UNHEALTHY
        else:
            overall_status = HealthStatus.DEGRADED
        
        return {
            'overall_status': overall_status,
            'timestamp': datetime.now().isoformat(),
            'components': reports,
        }
    
    def get_unhealthy_components(self) -> List[str]:
        """Get list of unhealthy components"""
        unhealthy = []
        
        for name, checker in self.components.items():
            report = checker.check_health()
            if report['status'] != HealthStatus.HEALTHY:
                unhealthy.append(name)
        
        return unhealthy
    
    def is_system_healthy(self) -> bool:
        """Check if entire system is healthy"""
        report = self.check_all()
        return report['overall_status'] == HealthStatus.HEALTHY


# Global health monitor
_health_monitor = SystemHealthMonitor()


def get_health_monitor() -> SystemHealthMonitor:
    """Get global health monitor instance"""
    return _health_monitor

