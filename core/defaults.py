"""
Default Parameter Values

This module defines sensible default values for new parameters added to the
memory management system, ensuring backward compatibility.

Validates: Requirement 8.6
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryManagementDefaults:
    """
    Default values for memory management parameters.

    These defaults are designed to work well for most use cases while
    maintaining backward compatibility with existing code.

    Validates: Requirement 8.6
    """

    # Model Management Defaults
    model_cache_size: int = 2
    """Maximum number of models to keep in LRU cache (default: 2)"""

    enable_lazy_loading: bool = True
    """Enable lazy loading of models (default: True)"""

    enable_lru_cache: bool = True
    """Enable LRU cache for models (default: True)"""

    # Image Processing Defaults
    mmap_threshold_pixels: int = 2000 * 2000
    """Threshold for using memory-mapped file loading (default: 4M pixels)"""

    tile_size: tuple = (512, 512)
    """Default tile size for large image processing (default: 512x512)"""

    tile_overlap: int = 64
    """Default overlap between tiles in pixels (default: 64)"""

    enable_inplace_operations: bool = True
    """Enable in-place image transformations (default: True)"""

    # Session Management Defaults
    session_timeout_minutes: int = 30
    """Session timeout in minutes (default: 30)"""

    max_session_memory_gb: float = 10.0
    """Maximum total session memory in GB (default: 10.0)"""

    enable_session_compression: bool = False
    """Enable image compression in session storage (default: False)"""

    session_cleanup_interval_minutes: int = 5
    """Interval for background session cleanup in minutes (default: 5)"""

    # Memory Monitoring Defaults
    memory_warning_threshold: float = 90.0
    """Memory usage threshold for warnings in percent (default: 90%)"""

    monitoring_interval_sec: float = 1.0
    """Memory monitoring sampling interval in seconds (default: 1.0)"""

    enable_memory_profiling: bool = False
    """Enable detailed memory profiling (default: False, for performance)"""

    enable_background_monitoring: bool = False
    """Enable background memory monitoring thread (default: False)"""

    # Garbage Collection Defaults
    gc_aggressiveness: str = "moderate"
    """GC aggressiveness level: 'low', 'moderate', 'high' (default: 'moderate')"""

    enable_auto_gc: bool = True
    """Enable automatic garbage collection after operations (default: True)"""

    gc_threshold_gen0: int = 700
    """GC threshold for generation 0 (default: 700)"""

    gc_threshold_gen1: int = 10
    """GC threshold for generation 1 (default: 10)"""

    gc_threshold_gen2: int = 10
    """GC threshold for generation 2 (default: 10)"""

    # TensorFlow Session Defaults
    enable_session_cleanup: bool = True
    """Enable automatic TensorFlow session cleanup (default: True)"""

    enable_gpu_memory_release: bool = True
    """Enable GPU memory release after inference (default: True)"""

    # Error Handling Defaults
    emergency_cleanup_threshold_mb: float = 500.0
    """Available memory threshold for emergency cleanup in MB (default: 500)"""

    enable_proactive_gc: bool = True
    """Enable proactive GC when memory is high (default: True)"""

    proactive_gc_threshold: float = 90.0
    """Memory threshold for proactive GC in percent (default: 90%)"""

    enable_request_rejection: bool = True
    """Enable request rejection when memory is low (default: True)"""

    # Performance Defaults
    max_peak_memory_gb: float = 1.5
    """Target maximum peak memory per operation in GB (default: 1.5)"""

    batch_size: int = 32
    """Default batch size for inference (default: 32)"""

    num_workers: int = 1
    """Number of worker threads for parallel processing (default: 1)"""

    # Logging Defaults
    enable_structured_logging: bool = True
    """Enable structured logging for memory events (default: True)"""

    log_level: str = "INFO"
    """Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR' (default: 'INFO')"""

    enable_metrics_endpoint: bool = False
    """Enable REST API metrics endpoint (default: False)"""

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert defaults to dictionary.

        Returns:
            Dictionary of all default values
        """
        return {
            "model_cache_size": self.model_cache_size,
            "enable_lazy_loading": self.enable_lazy_loading,
            "enable_lru_cache": self.enable_lru_cache,
            "mmap_threshold_pixels": self.mmap_threshold_pixels,
            "tile_size": self.tile_size,
            "tile_overlap": self.tile_overlap,
            "enable_inplace_operations": self.enable_inplace_operations,
            "session_timeout_minutes": self.session_timeout_minutes,
            "max_session_memory_gb": self.max_session_memory_gb,
            "enable_session_compression": self.enable_session_compression,
            "session_cleanup_interval_minutes": self.session_cleanup_interval_minutes,
            "memory_warning_threshold": self.memory_warning_threshold,
            "monitoring_interval_sec": self.monitoring_interval_sec,
            "enable_memory_profiling": self.enable_memory_profiling,
            "enable_background_monitoring": self.enable_background_monitoring,
            "gc_aggressiveness": self.gc_aggressiveness,
            "enable_auto_gc": self.enable_auto_gc,
            "gc_threshold_gen0": self.gc_threshold_gen0,
            "gc_threshold_gen1": self.gc_threshold_gen1,
            "gc_threshold_gen2": self.gc_threshold_gen2,
            "enable_session_cleanup": self.enable_session_cleanup,
            "enable_gpu_memory_release": self.enable_gpu_memory_release,
            "emergency_cleanup_threshold_mb": self.emergency_cleanup_threshold_mb,
            "enable_proactive_gc": self.enable_proactive_gc,
            "proactive_gc_threshold": self.proactive_gc_threshold,
            "enable_request_rejection": self.enable_request_rejection,
            "max_peak_memory_gb": self.max_peak_memory_gb,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "enable_structured_logging": self.enable_structured_logging,
            "log_level": self.log_level,
            "enable_metrics_endpoint": self.enable_metrics_endpoint,
        }

    @classmethod
    def get_default(cls, param_name: str) -> Any:
        """
        Get default value for a specific parameter.

        Args:
            param_name: Name of the parameter

        Returns:
            Default value for the parameter

        Raises:
            KeyError: If parameter name is not recognized

        Example:
            >>> default_cache_size = MemoryManagementDefaults.get_default('model_cache_size')
            >>> print(default_cache_size)  # 2
        """
        defaults = cls()
        if not hasattr(defaults, param_name):
            raise KeyError(f"Unknown parameter: {param_name}")
        return getattr(defaults, param_name)


# Singleton instance for easy access
DEFAULT_VALUES = MemoryManagementDefaults()


def get_default_value(param_name: str, fallback: Any = None) -> Any:
    """
    Get default value for a parameter with optional fallback.

    Args:
        param_name: Name of the parameter
        fallback: Value to return if parameter not found (default: None)

    Returns:
        Default value or fallback

    Example:
        >>> cache_size = get_default_value('model_cache_size', fallback=1)
        >>> print(cache_size)  # 2
    """
    try:
        return MemoryManagementDefaults.get_default(param_name)
    except KeyError:
        return fallback


def merge_with_defaults(user_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge user configuration with default values.

    This function takes a partial configuration dictionary and fills in
    missing values with sensible defaults.

    Args:
        user_config: User-provided configuration (partial or complete)

    Returns:
        Complete configuration with defaults filled in

    Example:
        >>> user_config = {'model_cache_size': 3}
        >>> full_config = merge_with_defaults(user_config)
        >>> print(full_config['session_timeout_minutes'])  # 30 (default)
        >>> print(full_config['model_cache_size'])  # 3 (user value)

    Validates: Requirement 8.6
    """
    # Start with all defaults
    merged = DEFAULT_VALUES.to_dict()

    # Override with user values
    merged.update(user_config)

    return merged


def validate_parameter_value(param_name: str, value: Any) -> bool:
    """
    Validate that a parameter value is acceptable.

    Args:
        param_name: Name of the parameter
        value: Value to validate

    Returns:
        True if value is valid, False otherwise

    Example:
        >>> is_valid = validate_parameter_value('model_cache_size', 2)
        >>> print(is_valid)  # True
        >>> is_valid = validate_parameter_value('model_cache_size', -1)
        >>> print(is_valid)  # False
    """
    # Define validation rules
    validation_rules = {
        "model_cache_size": lambda v: isinstance(v, int) and v >= 1,
        "session_timeout_minutes": lambda v: isinstance(v, (int, float)) and v > 0,
        "max_session_memory_gb": lambda v: isinstance(v, (int, float)) and v > 0,
        "memory_warning_threshold": lambda v: isinstance(v, (int, float)) and 0 < v <= 100,
        "monitoring_interval_sec": lambda v: isinstance(v, (int, float)) and v > 0,
        "gc_aggressiveness": lambda v: v in ["low", "moderate", "high"],
        "batch_size": lambda v: isinstance(v, int) and v >= 1,
        "num_workers": lambda v: isinstance(v, int) and v >= 1,
        "log_level": lambda v: v in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    }

    # Get validation rule for parameter
    validator = validation_rules.get(param_name)

    if validator is None:
        # No specific validation rule - accept any value
        return True

    try:
        return validator(value)
    except Exception:
        return False


def get_parameter_description(param_name: str) -> Optional[str]:
    """
    Get human-readable description of a parameter.

    Args:
        param_name: Name of the parameter

    Returns:
        Description string or None if parameter not found

    Example:
        >>> desc = get_parameter_description('model_cache_size')
        >>> print(desc)  # "Maximum number of models to keep in LRU cache"
    """
    # Get docstring from dataclass field
    defaults = MemoryManagementDefaults()
    if not hasattr(defaults, param_name):
        return None

    # Try to get docstring from class definition
    # (This is a simplified version - in practice you'd parse the docstring)
    descriptions = {
        "model_cache_size": "Maximum number of models to keep in LRU cache",
        "session_timeout_minutes": "Session timeout in minutes",
        "max_session_memory_gb": "Maximum total session memory in GB",
        "memory_warning_threshold": "Memory usage threshold for warnings in percent",
        "gc_aggressiveness": "GC aggressiveness level: 'low', 'moderate', 'high'",
        "batch_size": "Default batch size for inference",
    }

    return descriptions.get(param_name)


__all__ = [
    "MemoryManagementDefaults",
    "DEFAULT_VALUES",
    "get_default_value",
    "merge_with_defaults",
    "validate_parameter_value",
    "get_parameter_description",
]
