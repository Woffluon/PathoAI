"""
Memory Management Configuration Module

This module provides configuration management for memory and resource optimization
in the TensorFlow/Keras-based medical image analysis application.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing critical values"""

    pass


@dataclass
class MemoryConfig:
    """
    Memory management configuration with environment variable support.

    This class centralizes all memory-related configuration parameters and provides
    validation, environment variable loading, and sensible defaults.

    Attributes:
        model_cache_size: Maximum number of models to keep in LRU cache (default: 2)
        model_paths: Dictionary mapping model names to file paths
        mmap_threshold_pixels: Pixel count threshold for memory-mapped file loading
        tile_size: Default tile size for large image processing (height, width)
        tile_overlap: Overlap between tiles in pixels
        session_timeout_minutes: Inactive session timeout duration
        max_session_memory_gb: Maximum total memory for all sessions
        enable_compression: Enable image compression in session storage
        memory_warning_threshold: Memory usage percentage to trigger warnings
        monitoring_interval_sec: Memory monitoring sampling interval
        enable_profiling: Enable detailed memory profiling
        gc_aggressiveness: Garbage collection aggressiveness level
        max_peak_memory_gb: Maximum peak memory usage target
        emergency_cleanup_threshold_mb: Available memory threshold for emergency cleanup
    """

    # Model management
    model_cache_size: int = 2
    model_paths: Dict[str, str] = field(default_factory=dict)

    # Image processing
    mmap_threshold_pixels: int = 2000 * 2000
    tile_size: Tuple[int, int] = (512, 512)
    tile_overlap: int = 64

    # Session management
    session_timeout_minutes: int = 30
    max_session_memory_gb: float = 10.0
    enable_compression: bool = False

    # Memory monitoring
    memory_warning_threshold: float = 90.0
    monitoring_interval_sec: float = 1.0
    enable_profiling: bool = False

    # GC configuration
    gc_aggressiveness: str = "moderate"  # 'low', 'moderate', 'high'

    # Memory limits
    max_peak_memory_gb: float = 1.5
    emergency_cleanup_threshold_mb: float = 500.0

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        """
        Load configuration from environment variables with fallback to defaults.

        Environment Variables:
            MODEL_CACHE_SIZE: Maximum number of models in cache
            SESSION_TIMEOUT_MIN: Session timeout in minutes
            MAX_SESSION_MEMORY_GB: Maximum session memory in GB
            MEMORY_WARNING_THRESHOLD: Warning threshold percentage (0-100)
            ENABLE_MEMORY_PROFILING: Enable profiling (true/false)
            GC_AGGRESSIVENESS: GC level (low/moderate/high)
            MMAP_THRESHOLD_PIXELS: Pixel threshold for memory-mapped loading
            TILE_SIZE_HEIGHT: Tile height in pixels
            TILE_SIZE_WIDTH: Tile width in pixels
            TILE_OVERLAP: Tile overlap in pixels
            ENABLE_COMPRESSION: Enable image compression (true/false)
            MONITORING_INTERVAL_SEC: Monitoring interval in seconds
            MAX_PEAK_MEMORY_GB: Maximum peak memory in GB
            EMERGENCY_CLEANUP_THRESHOLD_MB: Emergency cleanup threshold in MB

        Returns:
            MemoryConfig instance with values from environment or defaults

        Raises:
            ConfigurationError: If critical configuration values are invalid
        """
        try:
            # Parse integer values
            model_cache_size = int(os.getenv("MODEL_CACHE_SIZE", "2"))
            session_timeout_minutes = int(os.getenv("SESSION_TIMEOUT_MIN", "30"))
            mmap_threshold_pixels = int(os.getenv("MMAP_THRESHOLD_PIXELS", str(2000 * 2000)))
            tile_size_height = int(os.getenv("TILE_SIZE_HEIGHT", "512"))
            tile_size_width = int(os.getenv("TILE_SIZE_WIDTH", "512"))
            tile_overlap = int(os.getenv("TILE_OVERLAP", "64"))

            # Parse float values
            max_session_memory_gb = float(os.getenv("MAX_SESSION_MEMORY_GB", "10.0"))
            memory_warning_threshold = float(os.getenv("MEMORY_WARNING_THRESHOLD", "90.0"))
            monitoring_interval_sec = float(os.getenv("MONITORING_INTERVAL_SEC", "1.0"))
            max_peak_memory_gb = float(os.getenv("MAX_PEAK_MEMORY_GB", "1.5"))
            emergency_cleanup_threshold_mb = float(
                os.getenv("EMERGENCY_CLEANUP_THRESHOLD_MB", "500.0")
            )

            # Parse boolean values
            enable_profiling = os.getenv("ENABLE_MEMORY_PROFILING", "false").lower() == "true"
            enable_compression = os.getenv("ENABLE_COMPRESSION", "false").lower() == "true"

            # Parse string values
            gc_aggressiveness = os.getenv("GC_AGGRESSIVENESS", "moderate").lower()

            config = cls(
                model_cache_size=model_cache_size,
                session_timeout_minutes=session_timeout_minutes,
                max_session_memory_gb=max_session_memory_gb,
                memory_warning_threshold=memory_warning_threshold,
                enable_profiling=enable_profiling,
                gc_aggressiveness=gc_aggressiveness,
                mmap_threshold_pixels=mmap_threshold_pixels,
                tile_size=(tile_size_height, tile_size_width),
                tile_overlap=tile_overlap,
                enable_compression=enable_compression,
                monitoring_interval_sec=monitoring_interval_sec,
                max_peak_memory_gb=max_peak_memory_gb,
                emergency_cleanup_threshold_mb=emergency_cleanup_threshold_mb,
            )

            logger.info(
                "Memory configuration loaded from environment",
                extra={
                    "model_cache_size": config.model_cache_size,
                    "session_timeout_minutes": config.session_timeout_minutes,
                    "gc_aggressiveness": config.gc_aggressiveness,
                    "enable_profiling": config.enable_profiling,
                },
            )

            return config

        except ValueError as e:
            error_msg = f"Invalid configuration value in environment variables: {e}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg) from e

    def validate(self) -> None:
        """
        Validate configuration values and fail fast if critical values are invalid.

        Raises:
            ConfigurationError: If any configuration value is invalid
        """
        errors = []

        # Validate model cache size
        if self.model_cache_size < 1:
            errors.append("model_cache_size must be >= 1")

        # Validate session timeout
        if self.session_timeout_minutes < 1:
            errors.append("session_timeout_minutes must be >= 1")

        # Validate GC aggressiveness
        if self.gc_aggressiveness not in ["low", "moderate", "high"]:
            errors.append("gc_aggressiveness must be 'low', 'moderate', or 'high'")

        # Validate memory thresholds
        if not 0 < self.memory_warning_threshold <= 100:
            errors.append("memory_warning_threshold must be between 0 and 100")

        if self.max_session_memory_gb <= 0:
            errors.append("max_session_memory_gb must be > 0")

        if self.max_peak_memory_gb <= 0:
            errors.append("max_peak_memory_gb must be > 0")

        if self.emergency_cleanup_threshold_mb <= 0:
            errors.append("emergency_cleanup_threshold_mb must be > 0")

        # Validate monitoring interval
        if self.monitoring_interval_sec <= 0:
            errors.append("monitoring_interval_sec must be > 0")

        # Validate tile configuration
        if self.tile_size[0] <= 0 or self.tile_size[1] <= 0:
            errors.append("tile_size dimensions must be > 0")

        if self.tile_overlap < 0:
            errors.append("tile_overlap must be >= 0")

        if self.mmap_threshold_pixels <= 0:
            errors.append("mmap_threshold_pixels must be > 0")

        # If there are validation errors, raise exception
        if errors:
            error_msg = "Configuration validation failed: " + "; ".join(errors)
            logger.error(error_msg, extra={"validation_errors": errors})
            raise ConfigurationError(error_msg)

        logger.info("Memory configuration validated successfully")

    def apply_defaults_on_invalid(self) -> None:
        """
        Apply default values for invalid configuration and log warnings.

        This method is used when non-critical configuration values are invalid
        but the system should continue with defaults rather than failing.
        """
        warnings = []

        # Check and fix model cache size
        if self.model_cache_size < 1:
            warnings.append(f"Invalid model_cache_size={self.model_cache_size}, using default=2")
            self.model_cache_size = 2

        # Check and fix session timeout
        if self.session_timeout_minutes < 1:
            warnings.append(
                f"Invalid session_timeout_minutes={self.session_timeout_minutes}, using default=30"
            )
            self.session_timeout_minutes = 30

        # Check and fix GC aggressiveness
        if self.gc_aggressiveness not in ["low", "moderate", "high"]:
            warnings.append(
                f"Invalid gc_aggressiveness={self.gc_aggressiveness}, using default='moderate'"
            )
            self.gc_aggressiveness = "moderate"

        # Check and fix memory warning threshold
        if not 0 < self.memory_warning_threshold <= 100:
            warnings.append(
                f"Invalid memory_warning_threshold={self.memory_warning_threshold}, using default=90.0"
            )
            self.memory_warning_threshold = 90.0

        # Log all warnings
        for warning in warnings:
            logger.warning(warning)

        if warnings:
            logger.info("Applied default values for invalid configuration parameters")

    def to_dict(self) -> Dict:
        """
        Convert configuration to dictionary for logging and serialization.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "model_cache_size": self.model_cache_size,
            "mmap_threshold_pixels": self.mmap_threshold_pixels,
            "tile_size": self.tile_size,
            "tile_overlap": self.tile_overlap,
            "session_timeout_minutes": self.session_timeout_minutes,
            "max_session_memory_gb": self.max_session_memory_gb,
            "enable_compression": self.enable_compression,
            "memory_warning_threshold": self.memory_warning_threshold,
            "monitoring_interval_sec": self.monitoring_interval_sec,
            "enable_profiling": self.enable_profiling,
            "gc_aggressiveness": self.gc_aggressiveness,
            "max_peak_memory_gb": self.max_peak_memory_gb,
            "emergency_cleanup_threshold_mb": self.emergency_cleanup_threshold_mb,
        }
