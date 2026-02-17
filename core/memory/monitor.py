"""
Memory Monitoring and Profiling Module

This module provides comprehensive memory monitoring, profiling, and emergency
cleanup capabilities for the TensorFlow/Keras-based medical image analysis application.

Typical Usage:
    >>> from core.memory import MemoryMonitor, MemoryConfig  # Recommended
    >>>
    >>> # Initialize monitor
    >>> config = MemoryConfig.from_env()
    >>> monitor = MemoryMonitor(config=config)
    >>>
    >>> # Start background monitoring
    >>> monitor.start_monitoring()
    >>>
    >>> # Profile memory usage
    >>> with monitor.profile_memory('model_inference'):
    ...     result = model.predict(data)
    >>>
    >>> # Handle OOM errors
    >>> with monitor.handle_oom('large_operation'):
    ...     process_large_data()

Import Paths:
    >>> from core.memory import MemoryMonitor, MemoryConfig  # Recommended
    >>> from core.memory.monitor import MemoryMonitor  # Also valid
"""

import gc
import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import psutil

from core.exceptions import EmergencyCleanupError, InsufficientMemoryError
from core.memory.config import MemoryConfig

logger = logging.getLogger("memory_management.monitor")


def log_memory_warning(
    logger: logging.Logger,
    message: str,
    current_percent: float,
    threshold_percent: float,
    rss_mb: float,
    include_stack: bool = True,
) -> None:
    """
    Log a memory threshold warning with detailed context.

    Args:
        logger: Logger instance to use
        message: Warning message
        current_percent: Current memory usage percentage
        threshold_percent: Configured threshold percentage
        rss_mb: Resident Set Size in MB
        include_stack: Include stack trace in log
    """
    import traceback

    extra_data = {
        "event_type": "memory_threshold_exceeded",
        "current_percent": current_percent,
        "threshold_percent": threshold_percent,
        "rss_mb": rss_mb,
    }

    if include_stack:
        extra_data["stack_trace"] = "".join(traceback.format_stack())

    logger.warning(message, extra=extra_data)


def log_emergency_cleanup(
    logger: logging.Logger,
    trigger_reason: str,
    memory_before_mb: float,
    memory_after_mb: float,
    models_unloaded: int = 0,
    sessions_cleaned: int = 0,
) -> None:
    """
    Log an emergency cleanup event with before/after metrics.

    Args:
        logger: Logger instance to use
        trigger_reason: Reason for emergency cleanup
        memory_before_mb: Memory usage before cleanup
        memory_after_mb: Memory usage after cleanup
        models_unloaded: Number of models unloaded
        sessions_cleaned: Number of sessions cleaned
    """
    logger.error(
        f"Emergency cleanup triggered: {trigger_reason}",
        extra={
            "event_type": "emergency_cleanup_triggered",
            "trigger_reason": trigger_reason,
            "memory_before_mb": memory_before_mb,
            "memory_after_mb": memory_after_mb,
            "memory_freed_mb": memory_before_mb - memory_after_mb,
            "models_unloaded": models_unloaded,
            "sessions_cleaned": sessions_cleaned,
        },
    )


@dataclass
class MemoryMetrics:
    """Memory usage metrics snapshot"""

    timestamp: datetime

    # Process metrics
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float  # System memory percentage

    # Python heap
    heap_mb: float
    objects_count: int

    # GPU (optional)
    gpu_allocated_mb: Optional[float] = None
    gpu_reserved_mb: Optional[float] = None
    gpu_free_mb: Optional[float] = None

    # GC statistics
    gc_collections: Dict[int, int] = field(default_factory=dict)

    # Application metrics
    model_cache_size: int = 0
    active_sessions: int = 0
    total_session_memory_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Prometheus/monitoring format"""
        result = {
            "timestamp": self.timestamp.isoformat(),
            "memory_rss_bytes": int(self.rss_mb * 1024 * 1024),
            "memory_vms_bytes": int(self.vms_mb * 1024 * 1024),
            "memory_percent": self.percent,
            "memory_heap_bytes": int(self.heap_mb * 1024 * 1024),
            "memory_objects_count": self.objects_count,
            "gc_collections_gen0": self.gc_collections.get(0, 0),
            "gc_collections_gen1": self.gc_collections.get(1, 0),
            "gc_collections_gen2": self.gc_collections.get(2, 0),
            "model_cache_size": self.model_cache_size,
            "active_sessions": self.active_sessions,
            "session_memory_bytes": int(self.total_session_memory_mb * 1024 * 1024),
        }

        if self.gpu_allocated_mb is not None:
            result["gpu_allocated_bytes"] = int(self.gpu_allocated_mb * 1024 * 1024)
        if self.gpu_reserved_mb is not None:
            result["gpu_reserved_bytes"] = int(self.gpu_reserved_mb * 1024 * 1024)
        if self.gpu_free_mb is not None:
            result["gpu_free_bytes"] = int(self.gpu_free_mb * 1024 * 1024)

        return result


class MemoryMonitor:
    """
    Memory monitoring and profiling system.

    Provides process-level memory tracking, Python heap monitoring, GPU memory tracking,
    profiling capabilities, and emergency cleanup mechanisms.
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        warning_threshold_percent: Optional[float] = None,
        sampling_interval_sec: Optional[float] = None,
    ):
        """
        Initialize memory monitor.

        Args:
            config: Memory configuration (uses defaults if None)
            warning_threshold_percent: Override warning threshold
            sampling_interval_sec: Override sampling interval
        """
        self.config = config or MemoryConfig()
        self.warning_threshold = warning_threshold_percent or self.config.memory_warning_threshold
        self.sampling_interval = sampling_interval_sec or self.config.monitoring_interval_sec

        # Process handle for memory tracking
        self._process = psutil.Process(os.getpid())

        # Background monitoring
        self._monitoring_thread: Optional[threading.Thread] = None
        self._monitoring_active = False
        self._monitoring_lock = threading.Lock()

        # Metrics history for profiling
        self._metrics_history: list = []
        self._max_history_size = 1000

        # Emergency cleanup callbacks
        self._cleanup_callbacks: list = []

        # Check for TensorFlow/GPU availability
        self._tf_available = False
        self._gpu_available = False
        try:
            import tensorflow as tf

            self._tf_available = True
            self._gpu_available = len(tf.config.list_physical_devices("GPU")) > 0
        except (ImportError, RuntimeError):
            pass

        logger.info(
            "MemoryMonitor initialized",
            extra={
                "warning_threshold": self.warning_threshold,
                "sampling_interval": self.sampling_interval,
                "tf_available": self._tf_available,
                "gpu_available": self._gpu_available,
            },
        )

    def get_process_memory(self) -> Dict[str, float]:
        """
        Get process-level memory usage.

        Returns:
            Dictionary with:
                - rss_mb: Resident Set Size in MB
                - vms_mb: Virtual Memory Size in MB
                - percent: Percentage of system memory used

        Validates: Requirement 5.1
        """
        try:
            mem_info = self._process.memory_info()
            mem_percent = self._process.memory_percent()

            return {
                "rss_mb": mem_info.rss / (1024 * 1024),
                "vms_mb": mem_info.vms / (1024 * 1024),
                "percent": mem_percent,
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.error(f"Failed to get process memory: {e}")
            return {"rss_mb": 0.0, "vms_mb": 0.0, "percent": 0.0}

    def get_python_memory(self) -> Dict[str, float]:
        """
        Get Python heap memory usage.

        Returns:
            Dictionary with:
                - heap_mb: Python heap size in MB
                - objects_count: Number of tracked objects

        Validates: Requirement 5.2
        """
        try:
            # Get all objects tracked by GC
            objects = gc.get_objects()
            objects_count = len(objects)

            # Estimate heap size (rough approximation)
            # This is not exact but gives a reasonable estimate
            import sys

            heap_size = sum(sys.getsizeof(obj) for obj in objects[:1000])  # Sample first 1000
            heap_size_mb = (heap_size * objects_count / 1000) / (1024 * 1024)

            return {
                "heap_mb": heap_size_mb,
                "objects_count": objects_count,
            }
        except Exception as e:
            logger.error(f"Failed to get Python memory: {e}")
            return {"heap_mb": 0.0, "objects_count": 0}

    def get_gpu_memory(self) -> Optional[Dict[str, float]]:
        """
        Get GPU memory usage (if GPU is available).

        Returns:
            Dictionary with GPU memory info or None if GPU not available:
                - allocated_mb: Allocated GPU memory in MB
                - reserved_mb: Reserved GPU memory in MB
                - free_mb: Free GPU memory in MB

        Validates: Requirement 5.8
        """
        if not self._tf_available or not self._gpu_available:
            return None

        try:
            import tensorflow as tf

            # Get GPU memory info
            gpu_devices = tf.config.list_physical_devices("GPU")
            if not gpu_devices:
                return None

            # TensorFlow 2.x memory stats
            try:
                memory_info = tf.config.experimental.get_memory_info("GPU:0")
                return {
                    "allocated_mb": memory_info["current"] / (1024 * 1024),
                    "reserved_mb": memory_info["peak"] / (1024 * 1024),
                    "free_mb": 0.0,  # TF doesn't provide free memory directly
                }
            except (AttributeError, RuntimeError):
                # Fallback for older TF versions or if memory info not available
                return {
                    "allocated_mb": 0.0,
                    "reserved_mb": 0.0,
                    "free_mb": 0.0,
                }

        except Exception as e:
            logger.warning(f"Failed to get GPU memory: {e}")
            return None

    def get_current_metrics(
        self,
        model_cache_size: int = 0,
        active_sessions: int = 0,
        total_session_memory_mb: float = 0.0,
    ) -> MemoryMetrics:
        """
        Get current memory metrics snapshot.

        Args:
            model_cache_size: Current model cache size
            active_sessions: Number of active sessions
            total_session_memory_mb: Total session memory usage

        Returns:
            MemoryMetrics object with current state
        """
        process_mem = self.get_process_memory()
        python_mem = self.get_python_memory()
        gpu_mem = self.get_gpu_memory()

        # Get GC stats
        gc.get_stats()  # noqa: F841
        gc_collections = {i: gc.get_count()[i] if i < len(gc.get_count()) else 0 for i in range(3)}

        metrics = MemoryMetrics(
            timestamp=datetime.utcnow(),
            rss_mb=process_mem["rss_mb"],
            vms_mb=process_mem["vms_mb"],
            percent=process_mem["percent"],
            heap_mb=python_mem["heap_mb"],
            objects_count=python_mem["objects_count"],
            gc_collections=gc_collections,
            model_cache_size=model_cache_size,
            active_sessions=active_sessions,
            total_session_memory_mb=total_session_memory_mb,
        )

        if gpu_mem:
            metrics.gpu_allocated_mb = gpu_mem["allocated_mb"]
            metrics.gpu_reserved_mb = gpu_mem["reserved_mb"]
            metrics.gpu_free_mb = gpu_mem["free_mb"]

        return metrics

    def check_memory_available(self, required_mb: float) -> bool:
        """
        Check if sufficient memory is available.

        Args:
            required_mb: Required memory in MB

        Returns:
            True if sufficient memory available, False otherwise
        """
        try:
            available_mb = psutil.virtual_memory().available / (1024 * 1024)
            return available_mb >= required_mb
        except Exception as e:
            logger.error(f"Failed to check available memory: {e}")
            return False

    def check_and_reject_if_insufficient(
        self, required_mb: float, operation_name: str = "operation", suggest_resolution: bool = True
    ) -> None:
        """
        Check memory availability and raise exception if insufficient.

        This method checks if sufficient memory is available for an operation.
        If memory is insufficient, it raises InsufficientMemoryError with a
        descriptive message and optional resolution suggestions.

        Args:
            required_mb: Required memory in MB
            operation_name: Name of the operation (for error message)
            suggest_resolution: Include resolution suggestions in error

        Raises:
            InsufficientMemoryError: If available memory < required_mb

        Validates: Requirements 7.1, 7.7
        """
        try:
            vm = psutil.virtual_memory()
            available_mb = vm.available / (1024 * 1024)

            # Check if we have enough memory
            if available_mb < required_mb:
                logger.warning(
                    f"Insufficient memory for {operation_name}",
                    extra={
                        "required_mb": required_mb,
                        "available_mb": available_mb,
                        "total_mb": vm.total / (1024 * 1024),
                        "percent_used": vm.percent,
                    },
                )

                # Create error with suggestions
                error = InsufficientMemoryError(required_mb, available_mb)

                if suggest_resolution:
                    # Add resolution suggestions to the error message
                    suggestions = []

                    # Suggest reducing image resolution
                    if "image" in operation_name.lower():
                        suggestions.append("reduce image resolution")

                    # Suggest closing other applications
                    if vm.percent > 80:
                        suggestions.append("close other applications")

                    # Suggest clearing cache
                    suggestions.append("clear model cache")

                    if suggestions:
                        suggestion_text = ", ".join(suggestions)
                        error.args = (f"{error.args[0]} Suggestions: {suggestion_text}.",)

                raise error

            # Log successful check
            logger.debug(
                f"Memory check passed for {operation_name}",
                extra={
                    "required_mb": required_mb,
                    "available_mb": available_mb,
                },
            )

        except InsufficientMemoryError:
            # Re-raise our custom exception
            raise
        except Exception as e:
            # Log error but don't block operation
            logger.error(f"Failed to check memory availability: {e}")
            # In case of check failure, we allow the operation to proceed
            # rather than blocking it with an error

    def get_available_memory_mb(self) -> float:
        """
        Get currently available memory in MB.

        Returns:
            Available memory in MB, or 0.0 if check fails
        """
        try:
            return psutil.virtual_memory().available / (1024 * 1024)
        except Exception as e:
            logger.error(f"Failed to get available memory: {e}")
            return 0.0

    def register_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """
        Register a callback for emergency cleanup.

        Args:
            callback: Function to call during emergency cleanup
        """
        self._cleanup_callbacks.append(callback)
        callback_name = getattr(callback, "__name__", repr(callback))
        logger.debug(f"Registered cleanup callback: {callback_name}")

    def trigger_emergency_cleanup(self) -> None:
        """
        Trigger emergency cleanup procedures.

        This method:
        1. Calls all registered cleanup callbacks
        2. Forces full garbage collection
        3. Logs the cleanup event with memory statistics

        Raises:
            EmergencyCleanupError: If cleanup fails critically

        Validates: Requirements 7.2, 7.3, 7.4
        """
        logger.warning("Emergency cleanup triggered")

        # Get memory before cleanup
        try:
            memory_before = self.get_process_memory()["rss_mb"]
        except Exception as e:
            logger.error(f"Failed to get memory before cleanup: {e}")
            memory_before = 0.0

        # Call all cleanup callbacks
        models_unloaded = 0
        sessions_cleaned = 0
        failed_callbacks = []

        for callback in self._cleanup_callbacks:
            try:
                result = callback()
                if isinstance(result, dict):
                    models_unloaded += result.get("models_unloaded", 0)
                    sessions_cleaned += result.get("sessions_cleaned", 0)
            except Exception as e:
                callback_name = getattr(callback, "__name__", repr(callback))
                logger.error(f"Cleanup callback '{callback_name}' failed: {e}", exc_info=True)
                failed_callbacks.append(callback_name)

        # Force full garbage collection
        try:
            collected = gc.collect(2)
            logger.debug(f"Garbage collection collected {collected} objects")
        except Exception as e:
            logger.error(f"Garbage collection failed: {e}")

        # Get memory after cleanup
        time.sleep(0.5)  # Allow GC to complete
        try:
            memory_after = self.get_process_memory()["rss_mb"]
        except Exception as e:
            logger.error(f"Failed to get memory after cleanup: {e}")
            memory_after = memory_before

        # Log cleanup event
        try:
            log_emergency_cleanup(
                logger,
                trigger_reason="emergency_cleanup_triggered",
                memory_before_mb=memory_before,
                memory_after_mb=memory_after,
                models_unloaded=models_unloaded,
                sessions_cleaned=sessions_cleaned,
            )
        except Exception as e:
            logger.error(f"Failed to log emergency cleanup: {e}")

        # Check if cleanup was successful
        memory_freed = memory_before - memory_after

        if memory_freed > 0:
            logger.info(f"Emergency cleanup completed: freed {memory_freed:.2f} MB")
        else:
            logger.warning(
                f"Emergency cleanup completed but no memory freed "
                f"(before: {memory_before:.2f} MB, after: {memory_after:.2f} MB)"
            )

        # If cleanup failed critically, raise error
        if failed_callbacks and memory_freed <= 0:
            raise EmergencyCleanupError(
                f"Emergency cleanup failed: {len(failed_callbacks)} callbacks failed "
                f"and no memory was freed",
                memory_before_mb=memory_before,
            )

    @contextmanager
    def profile_memory(self, operation_name: str):
        """
        Context manager for profiling memory usage of operations.

        Usage:
            with monitor.profile_memory('model_inference'):
                result = model.predict(data)

        Logs: Operation name, memory delta, execution time

        Validates: Requirements 5.4, 5.5
        """
        # Get initial memory
        start_time = time.time()
        start_memory = self.get_process_memory()["rss_mb"]

        try:
            yield
        finally:
            # Get final memory
            end_time = time.time()
            end_memory = self.get_process_memory()["rss_mb"]

            # Calculate deltas
            memory_delta = end_memory - start_memory
            duration = end_time - start_time

            # Log profiling data
            logger.info(
                f"Memory profile: {operation_name}",
                extra={
                    "operation": operation_name,
                    "memory_delta_mb": memory_delta,
                    "duration_sec": duration,
                    "start_memory_mb": start_memory,
                    "end_memory_mb": end_memory,
                },
            )

            # Store in history if profiling enabled
            if self.config.enable_profiling:
                self._metrics_history.append(
                    {
                        "timestamp": datetime.utcnow(),
                        "operation": operation_name,
                        "memory_delta_mb": memory_delta,
                        "duration_sec": duration,
                    }
                )

                # Trim history if too large
                if len(self._metrics_history) > self._max_history_size:
                    self._metrics_history = self._metrics_history[-self._max_history_size :]

    @contextmanager
    def handle_oom(self, operation_name: str = "operation"):
        """
        Context manager for handling OOM (Out of Memory) errors.

        This context manager catches MemoryError exceptions and attempts
        emergency cleanup before re-raising the error. It also logs the
        incident with memory statistics.

        Usage:
            with monitor.handle_oom('model_inference'):
                result = model.predict(large_data)

        Args:
            operation_name: Name of the operation (for logging)

        Raises:
            MemoryError: Re-raised after emergency cleanup attempt
            EmergencyCleanupError: If cleanup fails critically

        Validates: Requirements 7.2, 7.3, 7.4
        """
        try:
            yield
        except MemoryError as e:
            # Log OOM error
            logger.error(
                f"Out of Memory error during {operation_name}",
                extra={
                    "operation": operation_name,
                    "error": str(e),
                },
                exc_info=True,
            )

            # Attempt emergency cleanup
            try:
                logger.warning(f"Attempting emergency cleanup after OOM in {operation_name}")
                self.trigger_emergency_cleanup()
            except EmergencyCleanupError as cleanup_error:
                logger.critical(
                    f"Emergency cleanup failed after OOM: {cleanup_error}", exc_info=True
                )
                # Re-raise the cleanup error as it's more critical
                raise
            except Exception as cleanup_error:
                logger.error(
                    f"Unexpected error during emergency cleanup: {cleanup_error}", exc_info=True
                )

            # Re-raise the original OOM error
            raise

    def _check_threshold_and_warn(self) -> None:
        """
        Check memory threshold and log warning if exceeded.
        Also triggers proactive garbage collection if threshold is very high.

        Validates: Requirements 5.3, 7.6
        """
        try:
            process_mem = self.get_process_memory()
            current_percent = process_mem["percent"]

            # Check if we should trigger proactive GC (90% threshold)
            if current_percent >= 90.0:
                logger.warning(
                    f"Memory usage critical ({current_percent:.1f}%), triggering proactive GC"
                )
                self._trigger_proactive_gc()

            # Log warning if threshold exceeded
            if current_percent >= self.warning_threshold:
                log_memory_warning(
                    logger,
                    f"Memory threshold exceeded: {current_percent:.1f}% (threshold: {self.warning_threshold}%)",
                    current_percent=current_percent,
                    threshold_percent=self.warning_threshold,
                    rss_mb=process_mem["rss_mb"],
                    include_stack=True,
                )
        except Exception as e:
            logger.error(f"Failed to check memory threshold: {e}")

    def check_and_trigger_proactive_gc(self, threshold_percent: float = 90.0) -> bool:
        """
        Check memory usage and trigger proactive GC if threshold exceeded.

        This method can be called manually before memory-intensive operations
        to proactively free memory and reduce the risk of OOM errors.

        Args:
            threshold_percent: Memory usage percentage threshold (default: 90%)

        Returns:
            True if GC was triggered, False otherwise

        Validates: Requirement 7.6
        """
        try:
            current_percent = self.get_process_memory()["percent"]

            if current_percent >= threshold_percent:
                logger.info(f"Memory usage at {current_percent:.1f}%, triggering proactive GC")
                self._trigger_proactive_gc()
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to check/trigger proactive GC: {e}")
            return False

    def _trigger_proactive_gc(self) -> None:
        """
        Trigger proactive garbage collection.

        This method is called when memory usage exceeds 90% to prevent
        OOM errors by proactively freeing memory.

        Validates: Requirement 7.6
        """
        try:
            memory_before = self.get_process_memory()["rss_mb"]

            # Force full garbage collection
            collected = gc.collect(2)

            # Wait for GC to complete
            time.sleep(0.1)

            memory_after = self.get_process_memory()["rss_mb"]
            memory_freed = memory_before - memory_after

            logger.info(
                "Proactive garbage collection completed",
                extra={
                    "objects_collected": collected,
                    "memory_before_mb": memory_before,
                    "memory_after_mb": memory_after,
                    "memory_freed_mb": memory_freed,
                },
            )
        except Exception as e:
            logger.error(f"Proactive GC failed: {e}", exc_info=True)

    def _monitoring_loop(self) -> None:
        """
        Background monitoring loop.

        Runs in a separate thread and periodically checks memory usage.
        """
        logger.info("Memory monitoring thread started")

        while self._monitoring_active:
            try:
                # Check threshold
                self._check_threshold_and_warn()

                # Store metrics if profiling enabled
                if self.config.enable_profiling:
                    metrics = self.get_current_metrics()
                    self._metrics_history.append(
                        {
                            "timestamp": metrics.timestamp,
                            "rss_mb": metrics.rss_mb,
                            "percent": metrics.percent,
                        }
                    )

                    # Trim history
                    if len(self._metrics_history) > self._max_history_size:
                        self._metrics_history = self._metrics_history[-self._max_history_size :]

                # Sleep for sampling interval
                time.sleep(self.sampling_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                time.sleep(self.sampling_interval)

        logger.info("Memory monitoring thread stopped")

    def start_monitoring(self) -> None:
        """
        Start background memory monitoring thread.

        Validates: Requirement 5.6
        """
        with self._monitoring_lock:
            if self._monitoring_active:
                logger.warning("Monitoring already active")
                return

            self._monitoring_active = True
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop, daemon=True, name="MemoryMonitor"
            )
            self._monitoring_thread.start()

            logger.info("Background memory monitoring started")

    def stop_monitoring(self) -> None:
        """
        Stop background memory monitoring thread.
        """
        with self._monitoring_lock:
            if not self._monitoring_active:
                logger.warning("Monitoring not active")
                return

            self._monitoring_active = False

            # Wait for thread to finish
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                self._monitoring_thread.join(timeout=5.0)

            logger.info("Background memory monitoring stopped")

    def get_metrics_endpoint(self) -> Dict[str, Any]:
        """
        Get metrics in Prometheus/monitoring format.

        Returns:
            Dictionary with current metrics for external monitoring

        Validates: Requirement 5.6
        """
        metrics = self.get_current_metrics()
        return metrics.to_dict()

    def get_profiling_timeline(self) -> list:
        """
        Get memory profiling timeline data.

        Returns:
            List of historical metrics for timeline visualization

        Validates: Requirement 5.7
        """
        if not self.config.enable_profiling:
            logger.warning("Profiling not enabled, no timeline data available")
            return []

        return self._metrics_history.copy()

    def clear_profiling_history(self) -> None:
        """Clear profiling history data."""
        self._metrics_history.clear()
        logger.debug("Profiling history cleared")

    def __enter__(self):
        """Context manager entry - start monitoring."""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop monitoring."""
        self.stop_monitoring()
        return False
