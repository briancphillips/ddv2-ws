"""
Memory tracking utilities for DynaDetect v2.
"""

import logging
import time
import psutil
from typing import Dict, Any, Optional, Tuple

class MemoryTracker:
    def __init__(self):
        """Initialize MemoryTracker."""
        self.logger = logging.getLogger(__name__)
        self.process = psutil.Process()
        
    def measure_memory_usage(self) -> float:
        """
        Measure current memory usage in MB.
        
        Returns:
            Current memory usage in MB
        """
        return self.process.memory_info().rss / 1024 / 1024
        
    def track_operation(self, operation_name: str) -> Tuple[float, float, float]:
        """
        Context manager to track memory usage and time for an operation.
        
        Args:
            operation_name: Name of the operation being tracked
            
        Returns:
            Tuple of (elapsed_time, memory_used, peak_memory)
        """
        start_time = time.time()
        start_memory = self.measure_memory_usage()
        peak_memory = start_memory
        
        def update_peak_memory():
            nonlocal peak_memory
            current = self.measure_memory_usage()
            peak_memory = max(peak_memory, current)
            return current
            
        return time.time() - start_time, update_peak_memory() - start_memory, peak_memory
        
    def get_system_metrics(self) -> Dict[str, float]:
        """
        Get overall system metrics.
        
        Returns:
            Dictionary containing system metrics
        """
        vm = psutil.virtual_memory()
        return {
            'total_memory': vm.total / 1024 / 1024,  # MB
            'available_memory': vm.available / 1024 / 1024,  # MB
            'memory_percent': vm.percent,
            'cpu_percent': psutil.cpu_percent(interval=1)
        } 