"""Performance monitoring system for DynaDetect v2."""

import time
from typing import Dict, List, Optional, Any
import numpy as np
from dataclasses import dataclass, field
import logging
import torch
import psutil
from contextlib import contextmanager
from collections import defaultdict

@dataclass
class MetricPoint:
    """Single metric measurement point."""
    value: float
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)

class PerformanceMonitor:
    """Non-invasive performance monitoring system."""
    
    def __init__(self, enabled: bool = True):
        """Initialize the monitor.
        
        Args:
            enabled: Whether monitoring is enabled
        """
        self.enabled = enabled
        self.metrics: Dict[str, List[MetricPoint]] = defaultdict(list)
        self.current_context: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        
    def set_context(self, **kwargs):
        """Set context for subsequent measurements."""
        if not self.enabled:
            return
        self.current_context.update(kwargs)
        
    def clear_context(self):
        """Clear the current context."""
        if not self.enabled:
            return
        self.current_context = {}
        
    def record_metric(self, name: str, value: float):
        """Record a metric value with current timestamp and context."""
        if not self.enabled:
            return
        self.metrics[name].append(
            MetricPoint(
                value=value,
                timestamp=time.time(),
                context=self.current_context.copy()
            )
        )
        
    @contextmanager
    def measure_time(self, name: str):
        """Context manager to measure execution time of a block."""
        if not self.enabled:
            yield
            return
            
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_metric(f"{name}_time", duration)
            
    def record_memory(self, name: str):
        """Record current memory usage."""
        if not self.enabled:
            return
            
        # CPU Memory
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / 1024 / 1024  # MB
        self.record_metric(f"{name}_cpu_memory", cpu_memory)
        
        # GPU Memory if available
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            self.record_metric(f"{name}_gpu_memory", gpu_memory)
            
    def get_metric_summary(self, name: str) -> Dict[str, float]:
        """Get summary statistics for a metric."""
        if not self.enabled or name not in self.metrics:
            return {}
            
        values = [point.value for point in self.metrics[name]]
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'count': len(values)
        }
        
    def get_metrics_by_context(self, name: str, context_key: str) -> Dict[Any, Dict[str, float]]:
        """Get metric summaries grouped by a context value."""
        if not self.enabled or name not in self.metrics:
            return {}
            
        grouped_metrics = defaultdict(list)
        for point in self.metrics[name]:
            if context_key in point.context:
                grouped_metrics[point.context[context_key]].append(point.value)
                
        return {
            key: {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }
            for key, values in grouped_metrics.items()
        }
        
    def clear_metrics(self):
        """Clear all recorded metrics."""
        if not self.enabled:
            return
        self.metrics.clear()
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        if not self.enabled:
            return {}
            
        report = {
            'metrics': {},
            'context_analysis': {},
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total / (1024 * 1024 * 1024),  # GB
                'gpu_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }
        
        # Add metric summaries
        for metric_name in self.metrics:
            report['metrics'][metric_name] = self.get_metric_summary(metric_name)
            
            # Add context analysis for relevant metrics
            if self.metrics[metric_name]:
                contexts = self.metrics[metric_name][0].context.keys()
                for context_key in contexts:
                    analysis = self.get_metrics_by_context(metric_name, context_key)
                    if analysis:
                        report['context_analysis'][f"{metric_name}_by_{context_key}"] = analysis
                        
        return report

# Global monitor instance
monitor = PerformanceMonitor() 