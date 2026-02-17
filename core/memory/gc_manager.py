"""
Garbage Collection Manager for optimized memory management.

This module provides a GarbageCollectionManager class that optimizes Python's
garbage collection behavior based on configurable aggressiveness levels.
"""

import gc
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Literal

logger = logging.getLogger(__name__)

AggressivenessLevel = Literal["low", "moderate", "high"]


@dataclass
class GCThresholds:
    """Garbage collection threshold configuration."""

    gen0: int
    gen1: int
    gen2: int


class GarbageCollectionManager:
    """
    Manages Python garbage collection with configurable aggressiveness.

    This class provides optimized garbage collection strategies for different
    deployment scenarios:
    - 'low': Performance-oriented, less frequent collections
    - 'moderate': Balanced approach (default)
    - 'high': Memory-constrained environments, aggressive collection

    Attributes:
        aggressiveness: Current GC aggressiveness level
    """

    # Predefined threshold configurations
    THRESHOLDS = {
        "low": GCThresholds(gen0=1000, gen1=15, gen2=15),
        "moderate": GCThresholds(gen0=700, gen1=10, gen2=10),
        "high": GCThresholds(gen0=400, gen1=5, gen2=5),
    }

    def __init__(self, aggressiveness: AggressivenessLevel = "moderate"):
        """
        Initialize the GarbageCollectionManager.

        Args:
            aggressiveness: GC aggressiveness level ('low', 'moderate', 'high')

        Raises:
            ValueError: If aggressiveness is not a valid level
        """
        if aggressiveness not in self.THRESHOLDS:
            raise ValueError(
                f"Invalid aggressiveness level: {aggressiveness}. "
                f"Must be one of: {list(self.THRESHOLDS.keys())}"
            )

        self.aggressiveness = aggressiveness
        self._configured = False

        logger.info(
            "gc_manager_initialized",
            extra={
                "aggressiveness": aggressiveness,
                "thresholds": self.THRESHOLDS[aggressiveness].__dict__,
            },
        )

    def configure_gc(self) -> None:
        """
        Configure garbage collection with the specified aggressiveness level.

        This method sets the GC thresholds based on the aggressiveness level
        and enables automatic garbage collection.
        """
        thresholds = self.THRESHOLDS[self.aggressiveness]

        # Enable automatic garbage collection
        gc.enable()

        # Set thresholds for each generation
        gc.set_threshold(thresholds.gen0, thresholds.gen1, thresholds.gen2)

        self._configured = True

        logger.info(
            "gc_configured",
            extra={
                "aggressiveness": self.aggressiveness,
                "gen0_threshold": thresholds.gen0,
                "gen1_threshold": thresholds.gen1,
                "gen2_threshold": thresholds.gen2,
            },
        )

    def collect_with_stats(self, generation: int = 2) -> Dict[str, Any]:
        """
        Perform garbage collection and return statistics.

        Args:
            generation: GC generation to collect (0, 1, or 2)
                       2 = full collection (default)

        Returns:
            Dictionary containing:
                - collected: Number of objects collected
                - uncollectable: Number of uncollectable objects
                - duration_ms: Collection duration in milliseconds

        Raises:
            ValueError: If generation is not 0, 1, or 2
        """
        if generation not in (0, 1, 2):
            raise ValueError(f"Invalid generation: {generation}. Must be 0, 1, or 2")

        # Get counts before collection
        before_counts = gc.get_count()

        # Perform collection with timing
        start_time = time.perf_counter()
        collected = gc.collect(generation)
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Get uncollectable objects
        uncollectable = len(gc.garbage)

        stats = {
            "collected": collected,
            "uncollectable": uncollectable,
            "duration_ms": round(duration_ms, 2),
            "generation": generation,
            "before_counts": before_counts,
            "after_counts": gc.get_count(),
        }

        logger.debug("gc_collection_completed", extra=stats)

        return stats

    def force_full_collection(self) -> None:
        """
        Force a full garbage collection across all generations.

        This method performs collection on all three generations (0, 1, 2)
        in sequence, ensuring maximum memory reclamation.
        """
        logger.info("gc_full_collection_started")

        total_collected = 0
        total_duration_ms = 0.0

        # Collect all generations
        for generation in range(3):
            stats = self.collect_with_stats(generation)
            total_collected += stats["collected"]
            total_duration_ms += stats["duration_ms"]

        logger.info(
            "gc_full_collection_completed",
            extra={
                "total_collected": total_collected,
                "total_duration_ms": round(total_duration_ms, 2),
            },
        )

    def get_gc_stats(self) -> Dict[str, Any]:
        """
        Get current garbage collection statistics.

        Returns:
            Dictionary containing:
                - enabled: Whether GC is enabled
                - thresholds: Current GC thresholds (gen0, gen1, gen2)
                - counts: Current object counts per generation
                - collections: Number of collections per generation
                - configured: Whether configure_gc() has been called
                - aggressiveness: Current aggressiveness level
        """
        thresholds = gc.get_threshold()
        counts = gc.get_count()

        # Get collection counts per generation
        collections = {}
        for i in range(3):
            collections[f"gen{i}"] = gc.get_stats()[i].get("collections", 0)

        stats = {
            "enabled": gc.isenabled(),
            "thresholds": {
                "gen0": thresholds[0],
                "gen1": thresholds[1],
                "gen2": thresholds[2],
            },
            "counts": {
                "gen0": counts[0],
                "gen1": counts[1],
                "gen2": counts[2],
            },
            "collections": collections,
            "configured": self._configured,
            "aggressiveness": self.aggressiveness,
        }

        return stats
