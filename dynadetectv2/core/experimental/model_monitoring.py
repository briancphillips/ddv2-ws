"""Model monitoring wrapper for DynaDetect v2."""

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from .monitoring import monitor
import numpy as np
import logging
from functools import wraps

class MonitoredModelMixin:
    """Mixin to add monitoring capabilities to models."""
    
    def __init__(self, *args, **kwargs):
        """Initialize monitoring context."""
        super().__init__(*args, **kwargs)
        self._monitoring_enabled = True
        self._model_name = self.__class__.__name__
        monitor.set_context(model_type=self._model_name)
        
    def _monitor_method(self, method_name):
        """Decorator to monitor method execution."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self._monitoring_enabled:
                    return func(*args, **kwargs)
                
                with monitor.measure_time(f"{self._model_name}_{method_name}"):
                    monitor.record_memory(f"{self._model_name}_{method_name}_start")
                    result = func(*args, **kwargs)
                    monitor.record_memory(f"{self._model_name}_{method_name}_end")
                    
                    # Record accuracy if available
                    if method_name == "predict" and isinstance(result, (torch.Tensor, np.ndarray)):
                        if len(args) > 1 and isinstance(args[1], (torch.Tensor, np.ndarray)):  # args[1] would be y_true
                            accuracy = (result == args[1]).mean()
                            monitor.record_metric(f"{self._model_name}_accuracy", float(accuracy))
                    
                    return result
            return wrapper
        return decorator

def create_monitored_model(model_class):
    """Create a monitored version of a model class."""
    class MonitoredModel(MonitoredModelMixin, model_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
        @property
        def monitoring_enabled(self):
            return self._monitoring_enabled
            
        @monitoring_enabled.setter
        def monitoring_enabled(self, value: bool):
            self._monitoring_enabled = value
            
        def fit(self, *args, **kwargs):
            return self._monitor_method("fit")(super().fit)(*args, **kwargs)
            
        def predict(self, *args, **kwargs):
            return self._monitor_method("predict")(super().predict)(*args, **kwargs)
            
        def predict_proba(self, *args, **kwargs):
            if hasattr(super(), 'predict_proba'):
                return self._monitor_method("predict_proba")(super().predict_proba)(*args, **kwargs)
            raise NotImplementedError
            
        def forward(self, *args, **kwargs):
            if isinstance(self, nn.Module):
                return self._monitor_method("forward")(super().forward)(*args, **kwargs)
            raise NotImplementedError
            
        def get_performance_metrics(self) -> Dict[str, Any]:
            """Get model-specific performance metrics."""
            if not self._monitoring_enabled:
                return {}
                
            metrics = {}
            for name in monitor.metrics:
                if name.startswith(self._model_name):
                    metrics[name] = monitor.get_metric_summary(name)
            return metrics
            
        def clear_metrics(self):
            """Clear model-specific metrics."""
            if not self._monitoring_enabled:
                return
                
            for name in list(monitor.metrics.keys()):
                if name.startswith(self._model_name):
                    del monitor.metrics[name]
    
    return MonitoredModel

# Example usage:
# MonitoredDDKNN = create_monitored_model(DDKNN)
# model = MonitoredDDKNN(n_neighbors=5) 