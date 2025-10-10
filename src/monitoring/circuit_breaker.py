"""Circuit breaker pattern for resilient API calls"""

import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Any, Optional

from loguru import logger


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures
    
    States:
    - CLOSED: Normal operation, all calls go through
    - OPEN: Too many failures, reject all calls
    - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        half_open_attempts: int = 3
    ):
        """
        Initialize circuit breaker
        
        Args:
            name: Name for logging
            failure_threshold: Failures before opening circuit
            timeout_seconds: Time to wait before trying again
            half_open_attempts: Successful calls needed to close circuit
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_attempts = half_open_attempts
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_state_change: datetime = datetime.now()
        
        # Statistics
        self.total_calls = 0
        self.total_failures = 0
        self.total_rejections = 0
        
        logger.info(f"CircuitBreaker '{name}' initialized: threshold={failure_threshold}, timeout={timeout_seconds}s")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker
        
        Args:
            func: Function to call
            *args, **kwargs: Function arguments
        
        Returns:
            Function result
        
        Raises:
            Exception if circuit is open or function fails
        """
        self.total_calls += 1
        
        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                self.total_rejections += 1
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Will retry at {self.last_failure_time + timedelta(seconds=self.timeout_seconds)}"
                )
        
        # Try to execute the function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        
        except Exception as e:
            self._on_failure(e)
            raise
    
    def _on_success(self):
        """Handle successful call"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            
            if self.success_count >= self.half_open_attempts:
                self._transition_to_closed()
        
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def _on_failure(self, error: Exception):
        """Handle failed call"""
        self.failure_count += 1
        self.total_failures += 1
        self.last_failure_time = datetime.now()
        
        logger.warning(
            f"CircuitBreaker '{self.name}' failure #{self.failure_count}: {error}"
        )
        
        if self.state == CircuitState.HALF_OPEN:
            # Failed during recovery attempt
            self._transition_to_open()
        
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self._transition_to_open()
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if not self.last_failure_time:
            return True
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.timeout_seconds
    
    def _transition_to_closed(self):
        """Transition to CLOSED state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_state_change = datetime.now()
        
        logger.info(f"✅ CircuitBreaker '{self.name}' → CLOSED (recovered)")
    
    def _transition_to_open(self):
        """Transition to OPEN state"""
        self.state = CircuitState.OPEN
        self.success_count = 0
        self.last_state_change = datetime.now()
        
        retry_time = self.last_failure_time + timedelta(seconds=self.timeout_seconds)
        logger.error(
            f"🔴 CircuitBreaker '{self.name}' → OPEN "
            f"(failed {self.failure_count} times, will retry at {retry_time.strftime('%H:%M:%S')})"
        )
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state"""
        self.state = CircuitState.HALF_OPEN
        self.failure_count = 0
        self.success_count = 0
        self.last_state_change = datetime.now()
        
        logger.info(f"🟡 CircuitBreaker '{self.name}' → HALF_OPEN (testing recovery)")
    
    def reset(self):
        """Manually reset circuit breaker"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        
        logger.info(f"CircuitBreaker '{self.name}' manually reset")
    
    def get_stats(self) -> dict:
        """Get circuit breaker statistics"""
        uptime = (datetime.now() - self.last_state_change).total_seconds()
        
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "total_rejections": self.total_rejections,
            "failure_rate": (self.total_failures / self.total_calls * 100) if self.total_calls > 0 else 0,
            "time_in_current_state_seconds": uptime,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None,
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


# Global circuit breakers for common operations
_circuit_breakers = {}


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    timeout_seconds: int = 60,
    half_open_attempts: int = 3
) -> CircuitBreaker:
    """
    Get or create a circuit breaker
    
    Args:
        name: Circuit breaker name
        failure_threshold: Failures before opening
        timeout_seconds: Wait time before retry
        half_open_attempts: Successes needed to close
    
    Returns:
        CircuitBreaker instance
    """
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(
            name, failure_threshold, timeout_seconds, half_open_attempts
        )
    
    return _circuit_breakers[name]


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    timeout_seconds: int = 60,
    half_open_attempts: int = 3
):
    """
    Decorator to wrap function with circuit breaker
    
    Usage:
        @circuit_breaker("alpaca_api", failure_threshold=3)
        def get_stock_data(symbol):
            return api.get_snapshot(symbol)
    """
    def decorator(func: Callable):
        cb = get_circuit_breaker(name, failure_threshold, timeout_seconds, half_open_attempts)
        
        def wrapper(*args, **kwargs):
            return cb.call(func, *args, **kwargs)
        
        wrapper.circuit_breaker = cb
        return wrapper
    
    return decorator

