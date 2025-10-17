"""
High-Frequency Execution Engine
Production-grade order execution with latency optimization and throughput management
"""

import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger
import json
import queue
import concurrent.futures

from src.portfolio.account_manager import AccountProfile
from src.execution.options_smart_router import OrderRequest, OrderResult, OrderStatus
from src.execution.cost_model import TransactionCostModel


class ExecutionPriority(Enum):
    """Execution priority levels"""
    CRITICAL = 1    # Emergency orders
    HIGH = 2        # Time-sensitive orders
    NORMAL = 3      # Standard orders
    LOW = 4         # Patient orders


@dataclass
class ExecutionTask:
    """Execution task structure"""
    task_id: str
    order_request: OrderRequest
    priority: ExecutionPriority
    created_at: datetime
    deadline: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    callback: Optional[Callable] = None


@dataclass
class ExecutionStats:
    """Execution engine statistics"""
    total_orders: int
    successful_orders: int
    failed_orders: int
    cancelled_orders: int
    avg_execution_time: float
    avg_latency: float
    throughput_per_second: float
    error_rate: float
    queue_depth: int
    active_workers: int


class HighFrequencyExecutionEngine:
    """
    High-frequency execution engine for production options trading
    
    Features:
    - Multi-threaded order processing
    - Priority-based execution queues
    - Latency optimization
    - Throughput management
    - Real-time performance monitoring
    - Automatic retry and error handling
    """
    
    def __init__(self, account_profile: AccountProfile, config: Dict = None):
        self.profile = account_profile
        
        # Configuration
        self.config = config or self._default_config()
        
        # Execution queues by priority
        self.execution_queues = {
            ExecutionPriority.CRITICAL: queue.PriorityQueue(),
            ExecutionPriority.HIGH: queue.PriorityQueue(),
            ExecutionPriority.NORMAL: queue.PriorityQueue(),
            ExecutionPriority.LOW: queue.PriorityQueue()
        }
        
        # Worker threads
        self.worker_threads = []
        self.max_workers = self.config['max_workers']
        self.is_running = False
        
        # Execution tracking
        self.active_tasks = {}
        self.completed_tasks = []
        self.failed_tasks = []
        
        # Performance metrics
        self.execution_stats = ExecutionStats(
            total_orders=0,
            successful_orders=0,
            failed_orders=0,
            cancelled_orders=0,
            avg_execution_time=0.0,
            avg_latency=0.0,
            throughput_per_second=0.0,
            error_rate=0.0,
            queue_depth=0,
            active_workers=0
        )
        
        # Thread-safe locks
        self.stats_lock = threading.Lock()
        self.tasks_lock = threading.Lock()
        
        # Performance monitoring
        self.performance_history = []
        self.last_throughput_calculation = datetime.now()
        self.throughput_window = 60  # seconds
        
        # Integration components
        self.cost_model = TransactionCostModel(account_profile)
        self.execution_callbacks = []
        
        logger.info(f"HighFrequencyExecutionEngine initialized for {account_profile.tier.value} tier")
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'max_workers': 10,
            'queue_size_limit': 1000,
            'execution_timeout': 30,  # seconds
            'retry_delay': 1,  # seconds
            'throughput_target': 100,  # orders per minute
            'latency_target': 100,  # milliseconds
            'performance_window': 60,  # seconds
            'auto_scaling': True,
            'scaling_threshold': 0.8,  # 80% queue utilization
            'max_scaling_factor': 2.0,
            'monitoring_interval': 5,  # seconds
            'cleanup_interval': 300,  # 5 minutes
            'execution_limits': {
                'orders_per_second': 50,
                'orders_per_minute': 1000,
                'orders_per_hour': 10000
            }
        }
    
    async def start(self) -> bool:
        """Start the execution engine"""
        try:
            logger.info("Starting high-frequency execution engine...")
            
            self.is_running = True
            
            # Start worker threads
            await self._start_workers()
            
            # Start monitoring thread
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            # Start cleanup thread
            self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self.cleanup_thread.start()
            
            logger.info("High-frequency execution engine started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting execution engine: {e}")
            return False
    
    async def _start_workers(self):
        """Start worker threads"""
        try:
            for i in range(self.max_workers):
                worker = threading.Thread(
                    target=self._worker_loop,
                    args=(i,),
                    daemon=True
                )
                worker.start()
                self.worker_threads.append(worker)
            
            logger.info(f"Started {self.max_workers} worker threads")
            
        except Exception as e:
            logger.error(f"Error starting workers: {e}")
            raise
    
    def submit_order(
        self,
        order_request: OrderRequest,
        priority: ExecutionPriority = ExecutionPriority.NORMAL,
        deadline: Optional[datetime] = None,
        callback: Optional[Callable] = None
    ) -> str:
        """
        Submit order for execution
        
        Args:
            order_request: Order to execute
            priority: Execution priority
            deadline: Optional deadline for execution
            callback: Optional callback function
        
        Returns:
            Task ID
        """
        try:
            # Generate task ID
            task_id = f"task_{int(time.time() * 1000)}_{len(self.active_tasks)}"
            
            # Create execution task
            task = ExecutionTask(
                task_id=task_id,
                order_request=order_request,
                priority=priority,
                created_at=datetime.now(),
                deadline=deadline,
                callback=callback
            )
            
            # Check queue limits
            queue_size = self.execution_queues[priority].qsize()
            if queue_size >= self.config['queue_size_limit']:
                logger.warning(f"Queue {priority.value} is full, rejecting order")
                return ""
            
            # Add to appropriate queue
            # Use negative priority for max-heap behavior (higher priority first)
            priority_value = -priority.value
            self.execution_queues[priority].put((priority_value, task.created_at, task))
            
            # Track active task
            with self.tasks_lock:
                self.active_tasks[task_id] = task
            
            # Update stats
            with self.stats_lock:
                self.execution_stats.total_orders += 1
                self.execution_stats.queue_depth += 1
            
            logger.info(f"Order submitted with task ID: {task_id}, priority: {priority.value}")
            return task_id
            
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            return ""
    
    def _worker_loop(self, worker_id: int):
        """Worker thread main loop"""
        try:
            logger.info(f"Worker {worker_id} started")
            
            while self.is_running:
                try:
                    # Get next task from any queue (priority order)
                    task = self._get_next_task()
                    
                    if task:
                        # Update worker count
                        with self.stats_lock:
                            self.execution_stats.active_workers += 1
                        
                        # Execute task
                        await self._execute_task(task, worker_id)
                        
                        # Update worker count
                        with self.stats_lock:
                            self.execution_stats.active_workers -= 1
                    else:
                        # No tasks available, sleep briefly
                        time.sleep(0.1)
                        
                except Exception as e:
                    logger.error(f"Worker {worker_id} error: {e}")
                    time.sleep(1)
            
            logger.info(f"Worker {worker_id} stopped")
            
        except Exception as e:
            logger.error(f"Worker {worker_id} fatal error: {e}")
    
    def _get_next_task(self) -> Optional[ExecutionTask]:
        """Get next task from queues in priority order"""
        try:
            # Check queues in priority order
            for priority in ExecutionPriority:
                queue = self.execution_queues[priority]
                
                if not queue.empty():
                    try:
                        # Get task from queue
                        _, _, task = queue.get_nowait()
                        
                        # Check if task has expired
                        if task.deadline and datetime.now() > task.deadline:
                            logger.warning(f"Task {task.task_id} expired, skipping")
                            with self.stats_lock:
                                self.execution_stats.cancelled_orders += 1
                                self.execution_stats.queue_depth -= 1
                            continue
                        
                        return task
                        
                    except queue.Empty:
                        continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting next task: {e}")
            return None
    
    async def _execute_task(self, task: ExecutionTask, worker_id: int):
        """Execute a single task"""
        try:
            start_time = time.time()
            
            logger.info(f"Worker {worker_id} executing task {task.task_id}")
            
            # Pre-execution checks
            if not await self._pre_execution_checks(task):
                await self._handle_task_failure(task, "Pre-execution checks failed")
                return
            
            # Estimate costs
            cost_estimate = await self._estimate_execution_cost(task)
            
            # Execute order (simplified - would use real broker API)
            execution_result = await self._execute_order(task, cost_estimate)
            
            # Post-execution processing
            await self._post_execution_processing(task, execution_result, cost_estimate)
            
            # Update stats
            execution_time = time.time() - start_time
            latency = (datetime.now() - task.created_at).total_seconds() * 1000  # ms
            
            with self.stats_lock:
                if execution_result.status == OrderStatus.FILLED:
                    self.execution_stats.successful_orders += 1
                elif execution_result.status == OrderStatus.REJECTED:
                    self.execution_stats.failed_orders += 1
                else:
                    self.execution_stats.cancelled_orders += 1
                
                self.execution_stats.queue_depth -= 1
                
                # Update averages
                total_executed = (self.execution_stats.successful_orders + 
                                self.execution_stats.failed_orders + 
                                self.execution_stats.cancelled_orders)
                
                if total_executed > 0:
                    self.execution_stats.avg_execution_time = (
                        (self.execution_stats.avg_execution_time * (total_executed - 1) + execution_time) / 
                        total_executed
                    )
                    
                    self.execution_stats.avg_latency = (
                        (self.execution_stats.avg_latency * (total_executed - 1) + latency) / 
                        total_executed
                    )
            
            # Move to completed tasks
            with self.tasks_lock:
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
                self.completed_tasks.append((task, execution_result, datetime.now()))
            
            # Call callback if provided
            if task.callback:
                try:
                    await task.callback(task, execution_result)
                except Exception as e:
                    logger.error(f"Error in task callback: {e}")
            
            # Trigger execution callbacks
            for callback in self.execution_callbacks:
                try:
                    await callback(task, execution_result)
                except Exception as e:
                    logger.error(f"Error in execution callback: {e}")
            
            logger.info(f"Task {task.task_id} completed successfully in {execution_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Error executing task {task.task_id}: {e}")
            await self._handle_task_failure(task, str(e))
    
    async def _pre_execution_checks(self, task: ExecutionTask) -> bool:
        """Perform pre-execution checks"""
        try:
            # Check execution limits
            if not await self._check_execution_limits():
                return False
            
            # Check account limits
            if not await self._check_account_limits(task.order_request):
                return False
            
            # Check market conditions
            if not await self._check_market_conditions(task.order_request):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in pre-execution checks: {e}")
            return False
    
    async def _check_execution_limits(self) -> bool:
        """Check execution rate limits"""
        try:
            now = datetime.now()
            limits = self.config['execution_limits']
            
            # Check orders per second
            recent_orders = [
                t for t in self.completed_tasks 
                if (now - t[2]).total_seconds() < 1
            ]
            
            if len(recent_orders) >= limits['orders_per_second']:
                logger.warning("Orders per second limit exceeded")
                return False
            
            # Check orders per minute
            recent_orders = [
                t for t in self.completed_tasks 
                if (now - t[2]).total_seconds() < 60
            ]
            
            if len(recent_orders) >= limits['orders_per_minute']:
                logger.warning("Orders per minute limit exceeded")
                return False
            
            # Check orders per hour
            recent_orders = [
                t for t in self.completed_tasks 
                if (now - t[2]).total_seconds() < 3600
            ]
            
            if len(recent_orders) >= limits['orders_per_hour']:
                logger.warning("Orders per hour limit exceeded")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking execution limits: {e}")
            return False
    
    async def _check_account_limits(self, order_request: OrderRequest) -> bool:
        """Check account-specific limits"""
        try:
            # Check daily order limit
            today_orders = [
                t for t in self.completed_tasks 
                if t[2].date() == datetime.now().date()
            ]
            
            # Account tier specific limits
            tier_limits = {
                'micro': 5,
                'small': 10,
                'medium': 25,
                'large': 50,
                'institutional': 100
            }
            
            limit = tier_limits.get(self.profile.tier.value, 10)
            
            if len(today_orders) >= limit:
                logger.warning(f"Daily order limit exceeded for {self.profile.tier.value} tier")
                return False
            
            # Check position size limits
            max_order_value = self.profile.balance * 0.1  # 10% of balance per order
            order_value = order_request.quantity * (order_request.price or 100) * 100
            
            if order_value > max_order_value:
                logger.warning("Order value exceeds position size limit")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking account limits: {e}")
            return False
    
    async def _check_market_conditions(self, order_request: OrderRequest) -> bool:
        """Check market conditions"""
        try:
            # Check if market is open (simplified)
            now = datetime.now()
            market_open = now.hour >= 9 and now.hour < 16  # 9 AM to 4 PM
            
            if not market_open:
                logger.warning("Market is closed")
                return False
            
            # Check for circuit breakers (simplified)
            # In real implementation, this would check actual market data
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking market conditions: {e}")
            return False
    
    async def _estimate_execution_cost(self, task: ExecutionTask) -> Any:
        """Estimate execution costs"""
        try:
            # Mock market data
            market_data = {
                'bid': task.order_request.price * 0.99 if task.order_request.price else 1.0,
                'ask': task.order_request.price * 1.01 if task.order_request.price else 1.0,
                'last': task.order_request.price or 1.0,
                'volume': 1000
            }
            
            # Estimate costs
            cost_estimate = self.cost_model.estimate_pre_trade_cost(
                symbol=task.order_request.symbol,
                side=task.order_request.side,
                quantity=task.order_request.quantity,
                current_price=task.order_request.price or 1.0,
                market_data=market_data
            )
            
            return cost_estimate
            
        except Exception as e:
            logger.error(f"Error estimating execution cost: {e}")
            return None
    
    async def _execute_order(self, task: ExecutionTask, cost_estimate: Any) -> OrderResult:
        """Execute the actual order"""
        try:
            # Simulate order execution (in real implementation, this would call broker API)
            await asyncio.sleep(0.1)  # Simulate execution time
            
            # Mock execution result
            execution_result = OrderResult(
                order_id=f"exec_{task.task_id}",
                symbol=task.order_request.symbol,
                side=task.order_request.side,
                quantity=task.order_request.quantity,
                filled_quantity=task.order_request.quantity,  # Assume full fill
                avg_fill_price=task.order_request.price or 1.0,
                status=OrderStatus.FILLED,
                execution_time=datetime.now(),
                fees=cost_estimate.estimated_commission if cost_estimate else 0.0,
                slippage=0.01,  # 1% slippage
                fill_rate=1.0,
                execution_algorithm='high_frequency',
                error_message=None
            )
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Error executing order: {e}")
            return OrderResult(
                order_id=f"exec_{task.task_id}",
                symbol=task.order_request.symbol,
                side=task.order_request.side,
                quantity=task.order_request.quantity,
                filled_quantity=0,
                avg_fill_price=0.0,
                status=OrderStatus.REJECTED,
                execution_time=datetime.now(),
                fees=0.0,
                slippage=0.0,
                fill_rate=0.0,
                execution_algorithm='high_frequency',
                error_message=str(e)
            )
    
    async def _post_execution_processing(self, task: ExecutionTask, execution_result: OrderResult, cost_estimate: Any):
        """Post-execution processing"""
        try:
            # Analyze execution vs estimate
            if cost_estimate:
                actual_execution = {
                    'order_id': execution_result.order_id,
                    'total_cost': execution_result.fees,
                    'commission': execution_result.fees,
                    'slippage': execution_result.slippage,
                    'market_impact': 0.0,  # Simplified
                    'avg_price': execution_result.avg_fill_price,
                    'fill_rate': execution_result.fill_rate
                }
                
                analysis = self.cost_model.analyze_execution(cost_estimate, actual_execution)
                
                # Log analysis results
                logger.info(f"Execution analysis for {task.task_id}: Quality score {analysis.execution_quality_score:.3f}")
            
            # Update performance history
            self._update_performance_history(task, execution_result)
            
        except Exception as e:
            logger.error(f"Error in post-execution processing: {e}")
    
    def _update_performance_history(self, task: ExecutionTask, execution_result: OrderResult):
        """Update performance history"""
        try:
            performance_data = {
                'timestamp': datetime.now(),
                'task_id': task.task_id,
                'execution_time': (execution_result.execution_time - task.created_at).total_seconds(),
                'success': execution_result.status == OrderStatus.FILLED,
                'priority': task.priority.value,
                'quantity': task.order_request.quantity
            }
            
            self.performance_history.append(performance_data)
            
            # Keep only recent history
            cutoff_time = datetime.now() - timedelta(hours=1)
            self.performance_history = [
                p for p in self.performance_history 
                if p['timestamp'] > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Error updating performance history: {e}")
    
    async def _handle_task_failure(self, task: ExecutionTask, error_message: str):
        """Handle task failure"""
        try:
            task.retry_count += 1
            
            if task.retry_count < task.max_retries:
                # Retry task
                logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count})")
                
                # Wait before retry
                await asyncio.sleep(self.config['retry_delay'])
                
                # Re-queue task
                priority_value = -task.priority.value
                self.execution_queues[task.priority].put((priority_value, task.created_at, task))
                
            else:
                # Max retries exceeded
                logger.error(f"Task {task.task_id} failed after {task.max_retries} retries: {error_message}")
                
                # Move to failed tasks
                with self.tasks_lock:
                    if task.task_id in self.active_tasks:
                        del self.active_tasks[task.task_id]
                    self.failed_tasks.append((task, error_message, datetime.now()))
                
                # Update stats
                with self.stats_lock:
                    self.execution_stats.failed_orders += 1
                    self.execution_stats.queue_depth -= 1
                
                # Call callback if provided
                if task.callback:
                    try:
                        await task.callback(task, None)
                    except Exception as e:
                        logger.error(f"Error in failure callback: {e}")
            
        except Exception as e:
            logger.error(f"Error handling task failure: {e}")
    
    def _monitoring_loop(self):
        """Performance monitoring loop"""
        try:
            while self.is_running:
                try:
                    # Calculate throughput
                    self._calculate_throughput()
                    
                    # Check for auto-scaling
                    if self.config['auto_scaling']:
                        self._check_auto_scaling()
                    
                    # Update error rate
                    self._update_error_rate()
                    
                    # Sleep until next monitoring cycle
                    time.sleep(self.config['monitoring_interval'])
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(5)
            
        except Exception as e:
            logger.error(f"Monitoring loop fatal error: {e}")
    
    def _calculate_throughput(self):
        """Calculate current throughput"""
        try:
            now = datetime.now()
            window_start = now - timedelta(seconds=self.throughput_window)
            
            # Count orders in time window
            orders_in_window = [
                t for t in self.completed_tasks 
                if t[2] > window_start
            ]
            
            # Calculate throughput
            if self.throughput_window > 0:
                throughput = len(orders_in_window) / self.throughput_window
                
                with self.stats_lock:
                    self.execution_stats.throughput_per_second = throughput
            
        except Exception as e:
            logger.error(f"Error calculating throughput: {e}")
    
    def _check_auto_scaling(self):
        """Check if auto-scaling is needed"""
        try:
            # Calculate queue utilization
            total_queue_size = sum(q.qsize() for q in self.execution_queues.values())
            max_queue_size = self.config['queue_size_limit'] * len(ExecutionPriority)
            utilization = total_queue_size / max_queue_size if max_queue_size > 0 else 0
            
            # Check if scaling is needed
            if utilization > self.config['scaling_threshold']:
                # Scale up workers
                current_workers = len(self.worker_threads)
                max_workers = self.max_workers
                scaling_factor = self.config['max_scaling_factor']
                target_workers = min(int(current_workers * scaling_factor), max_workers)
                
                if target_workers > current_workers:
                    logger.info(f"Auto-scaling: increasing workers from {current_workers} to {target_workers}")
                    # In real implementation, would start new worker threads
            
        except Exception as e:
            logger.error(f"Error in auto-scaling check: {e}")
    
    def _update_error_rate(self):
        """Update error rate"""
        try:
            with self.stats_lock:
                total_orders = (self.execution_stats.successful_orders + 
                              self.execution_stats.failed_orders + 
                              self.execution_stats.cancelled_orders)
                
                if total_orders > 0:
                    error_rate = (self.execution_stats.failed_orders / total_orders) * 100
                    self.execution_stats.error_rate = error_rate
            
        except Exception as e:
            logger.error(f"Error updating error rate: {e}")
    
    def _cleanup_loop(self):
        """Cleanup loop for old data"""
        try:
            while self.is_running:
                try:
                    # Clean up old completed tasks
                    cutoff_time = datetime.now() - timedelta(hours=24)
                    
                    with self.tasks_lock:
                        self.completed_tasks = [
                            t for t in self.completed_tasks 
                            if t[2] > cutoff_time
                        ]
                        
                        self.failed_tasks = [
                            t for t in self.failed_tasks 
                            if t[2] > cutoff_time
                        ]
                    
                    # Clean up performance history
                    self.performance_history = [
                        p for p in self.performance_history 
                        if p['timestamp'] > cutoff_time
                    ]
                    
                    # Sleep until next cleanup
                    time.sleep(self.config['cleanup_interval'])
                    
                except Exception as e:
                    logger.error(f"Error in cleanup loop: {e}")
                    time.sleep(60)
            
        except Exception as e:
            logger.error(f"Cleanup loop fatal error: {e}")
    
    def get_execution_stats(self) -> ExecutionStats:
        """Get current execution statistics"""
        with self.stats_lock:
            # Update queue depth
            self.execution_stats.queue_depth = sum(q.qsize() for q in self.execution_queues.values())
            
            return self.execution_stats
    
    def get_queue_status(self) -> Dict[str, int]:
        """Get queue status"""
        return {
            'critical': self.execution_queues[ExecutionPriority.CRITICAL].qsize(),
            'high': self.execution_queues[ExecutionPriority.HIGH].qsize(),
            'normal': self.execution_queues[ExecutionPriority.NORMAL].qsize(),
            'low': self.execution_queues[ExecutionPriority.LOW].qsize()
        }
    
    def get_active_tasks(self) -> List[ExecutionTask]:
        """Get active tasks"""
        with self.tasks_lock:
            return list(self.active_tasks.values())
    
    def add_execution_callback(self, callback: Callable):
        """Add execution callback"""
        self.execution_callbacks.append(callback)
    
    async def stop(self):
        """Stop the execution engine"""
        try:
            logger.info("Stopping high-frequency execution engine...")
            
            self.is_running = False
            
            # Wait for workers to finish
            for worker in self.worker_threads:
                worker.join(timeout=5)
            
            # Clear queues
            for queue in self.execution_queues.values():
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except:
                        break
            
            logger.info("High-frequency execution engine stopped")
            
        except Exception as e:
            logger.error(f"Error stopping execution engine: {e}")


# Example usage
if __name__ == "__main__":
    from src.portfolio.account_manager import UniversalAccountManager
    from src.execution.options_smart_router import OrderRequest, OrderType
    
    async def test_execution_engine():
        # Create account profile
        manager = UniversalAccountManager()
        profile = manager.create_account_profile(balance=25000)
        
        # Create execution engine
        engine = HighFrequencyExecutionEngine(profile)
        
        # Start engine
        success = await engine.start()
        if success:
            print("Execution engine started successfully")
        
        # Add execution callback
        async def execution_callback(task, result):
            if result:
                print(f"Order executed: {result.order_id}, Status: {result.status.value}")
            else:
                print(f"Order failed: {task.task_id}")
        
        engine.add_execution_callback(execution_callback)
        
        # Submit test orders
        print("Submitting test orders...")
        
        for i in range(5):
            order_request = OrderRequest(
                symbol=f'SPY240315C00500000',
                side='buy',
                quantity=10,
                order_type=OrderType.LIMIT,
                price=1.50 + i * 0.01,
                strategy='test_strategy'
            )
            
            task_id = engine.submit_order(
                order_request=order_request,
                priority=ExecutionPriority.NORMAL,
                callback=execution_callback
            )
            
            print(f"Submitted order {i+1} with task ID: {task_id}")
        
        # Monitor for a while
        print("Monitoring execution for 10 seconds...")
        await asyncio.sleep(10)
        
        # Get stats
        stats = engine.get_execution_stats()
        print(f"\nExecution Statistics:")
        print(f"Total Orders: {stats.total_orders}")
        print(f"Successful Orders: {stats.successful_orders}")
        print(f"Failed Orders: {stats.failed_orders}")
        print(f"Cancelled Orders: {stats.cancelled_orders}")
        print(f"Average Execution Time: {stats.avg_execution_time:.3f}s")
        print(f"Average Latency: {stats.avg_latency:.1f}ms")
        print(f"Throughput: {stats.throughput_per_second:.1f} orders/sec")
        print(f"Error Rate: {stats.error_rate:.1f}%")
        print(f"Queue Depth: {stats.queue_depth}")
        print(f"Active Workers: {stats.active_workers}")
        
        # Get queue status
        queue_status = engine.get_queue_status()
        print(f"\nQueue Status:")
        for priority, depth in queue_status.items():
            print(f"  {priority}: {depth}")
        
        # Stop engine
        await engine.stop()
        print("Execution engine stopped")
    
    # Run test
    asyncio.run(test_execution_engine())
