"""
Performance monitoring utilities for PathoAI.

This module provides dataclasses and utilities for tracking performance metrics
during image analysis operations.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """
    Performance metrics for analysis operations.

    Tracks timing, memory usage, and optimization flags for monitoring
    and debugging performance.

    Attributes:
        preprocessing_time: Time spent on image preprocessing (seconds)
        classification_time: Time spent on classification inference (seconds)
        segmentation_time: Time spent on segmentation inference (seconds)
        postprocessing_time: Time spent on postprocessing (seconds)
        total_time: Total analysis time (seconds)
        peak_memory_mb: Peak memory usage during analysis (MB)
        memory_delta_mb: Change in memory usage (MB)
        cache_hit: Whether preprocessing cache was hit
        parallel_execution: Whether parallel execution was used
        gradcam_enabled: Whether Grad-CAM was generated

    Validates: Requirements 10.1, 10.2, 10.3
    """

    preprocessing_time: float = 0.0
    classification_time: float = 0.0
    segmentation_time: float = 0.0
    postprocessing_time: float = 0.0
    total_time: float = 0.0

    peak_memory_mb: float = 0.0
    memory_delta_mb: float = 0.0

    cache_hit: bool = False
    parallel_execution: bool = False
    gradcam_enabled: bool = True

    def log_summary(self) -> None:
        """
        Log performance metrics summary.

        Logs timing breakdown and memory usage for debugging and monitoring.
        """
        logger.info("=== Performance Metrics ===")
        logger.info(f"Total Time: {self.total_time:.2f}s")
        logger.info(f"  - Preprocessing: {self.preprocessing_time:.2f}s")
        logger.info(f"  - Classification: {self.classification_time:.2f}s")
        logger.info(f"  - Segmentation: {self.segmentation_time:.2f}s")
        logger.info(f"  - Postprocessing: {self.postprocessing_time:.2f}s")
        logger.info(
            f"Memory - Peak: {self.peak_memory_mb:.1f}MB, Delta: {self.memory_delta_mb:.1f}MB"
        )
        logger.info(
            f"Optimizations - Cache Hit: {self.cache_hit}, Parallel: {self.parallel_execution}, Grad-CAM: {self.gradcam_enabled}"
        )

    def to_dict(self) -> dict:
        """
        Convert metrics to dictionary for serialization.

        Returns:
            Dictionary containing all metrics
        """
        return {
            "timing": {
                "preprocessing": self.preprocessing_time,
                "classification": self.classification_time,
                "segmentation": self.segmentation_time,
                "postprocessing": self.postprocessing_time,
                "total": self.total_time,
            },
            "memory": {
                "peak_mb": self.peak_memory_mb,
                "delta_mb": self.memory_delta_mb,
            },
            "optimizations": {
                "cache_hit": self.cache_hit,
                "parallel_execution": self.parallel_execution,
                "gradcam_enabled": self.gradcam_enabled,
            },
        }
