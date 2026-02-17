"""
Session Manager Module

This module provides session state management for Streamlit applications with
memory-efficient storage, timeout mechanisms, and selective cleanup.
"""

import io
import logging
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Set

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

from core.memory.config import MemoryConfig

logger = logging.getLogger(__name__)


@dataclass
class SessionData:
    """
    Data structure for storing session information.

    Attributes:
        session_id: Unique session identifier
        last_access: Timestamp of last access
        data: Dictionary storing session data
        memory_usage_mb: Estimated memory usage in MB
        compressed_keys: Set of keys that have compressed data
    """

    session_id: str
    last_access: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    memory_usage_mb: float = 0.0
    compressed_keys: Set[str] = field(default_factory=set)

    def update_last_access(self) -> None:
        """Update the last access timestamp to current time."""
        self.last_access = datetime.now()

    def is_inactive(self, timeout_minutes: int) -> bool:
        """
        Check if session is inactive based on timeout.

        Args:
            timeout_minutes: Timeout duration in minutes

        Returns:
            True if session has been inactive longer than timeout
        """
        timeout_delta = timedelta(minutes=timeout_minutes)
        return datetime.now() - self.last_access > timeout_delta

    def estimate_memory_usage(self) -> float:
        """
        Estimate memory usage of session data in MB.

        Returns:
            Estimated memory usage in MB
        """
        total_bytes = 0

        for key, value in self.data.items():
            try:
                if isinstance(value, np.ndarray):
                    total_bytes += value.nbytes
                elif isinstance(value, (str, bytes)):
                    total_bytes += sys.getsizeof(value)
                elif isinstance(value, (list, tuple, dict)):
                    total_bytes += sys.getsizeof(value)
                else:
                    total_bytes += sys.getsizeof(value)
            except Exception as e:
                logger.warning(
                    f"Failed to estimate size for key '{key}': {e}",
                    extra={"session_id": self.session_id, "key": key},
                )

        self.memory_usage_mb = total_bytes / (1024 * 1024)
        return self.memory_usage_mb


class SessionManager:
    """
    Manages Streamlit session state with memory-efficient storage and cleanup.

    This class provides:
    - Session timeout mechanism
    - Memory usage tracking
    - Selective cleanup (large objects vs metadata)
    - Optional image compression
    - Background cleanup tasks

    Thread-safe for multi-user Streamlit applications.
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        timeout_minutes: Optional[int] = None,
        max_total_memory_gb: Optional[float] = None,
    ):
        """
        Initialize SessionManager.

        Args:
            config: Memory configuration object (if None, uses defaults)
            timeout_minutes: Session timeout in minutes (overrides config)
            max_total_memory_gb: Maximum total session memory in GB (overrides config)
        """
        # Load configuration
        if config is None:
            config = MemoryConfig()

        self.config = config
        self.timeout_minutes = timeout_minutes or config.session_timeout_minutes
        self.max_total_memory_gb = max_total_memory_gb or config.max_session_memory_gb
        self.enable_compression = config.enable_compression

        # Session storage (OrderedDict for LRU-like access)
        self._sessions: OrderedDict[str, SessionData] = OrderedDict()
        self._lock = threading.Lock()

        # Background cleanup task
        self._cleanup_timer: Optional[threading.Timer] = None
        self._cleanup_interval_sec = 300  # 5 minutes
        self._running = False

        logger.info(
            "SessionManager initialized",
            extra={
                "timeout_minutes": self.timeout_minutes,
                "max_total_memory_gb": self.max_total_memory_gb,
                "enable_compression": self.enable_compression,
            },
        )

    def store_result(
        self, session_id: str, key: str, value: Any, compress: Optional[bool] = None
    ) -> None:
        """
        Store data in session state.

        Args:
            session_id: Session identifier
            key: Data key
            value: Data to store
            compress: Enable compression for this value (overrides config)
                     Only applies to numpy arrays and PIL Images

        Raises:
            ValueError: If compression is requested but not supported for value type
        """
        with self._lock:
            # Get or create session
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionData(
                    session_id=session_id, last_access=datetime.now()
                )

            session = self._sessions[session_id]
            session.update_last_access()

            # Move to end (LRU)
            self._sessions.move_to_end(session_id)

            # Determine if compression should be used
            use_compression = compress if compress is not None else self.enable_compression

            # Compress if requested and supported
            if use_compression:
                compressed_value = self._compress_value(value)
                if compressed_value is not None:
                    session.data[key] = compressed_value
                    session.compressed_keys.add(key)
                    logger.debug(
                        f"Stored compressed data for key '{key}'",
                        extra={"session_id": session_id, "key": key},
                    )
                else:
                    # Compression not supported, store as-is
                    session.data[key] = value
                    session.compressed_keys.discard(key)
            else:
                session.data[key] = value
                session.compressed_keys.discard(key)

            # Update memory usage estimate
            session.estimate_memory_usage()

            logger.debug(
                f"Stored data for session '{session_id}', key '{key}'",
                extra={
                    "session_id": session_id,
                    "key": key,
                    "memory_mb": session.memory_usage_mb,
                    "compressed": key in session.compressed_keys,
                },
            )

    def get_result(self, session_id: str, key: str) -> Optional[Any]:
        """
        Retrieve data from session state.

        Args:
            session_id: Session identifier
            key: Data key

        Returns:
            Stored data or None if not found
        """
        with self._lock:
            if session_id not in self._sessions:
                return None

            session = self._sessions[session_id]
            session.update_last_access()

            # Move to end (LRU)
            self._sessions.move_to_end(session_id)

            if key not in session.data:
                return None

            value = session.data[key]

            # Decompress if needed
            if key in session.compressed_keys:
                value = self._decompress_value(value)

            logger.debug(
                f"Retrieved data for session '{session_id}', key '{key}'",
                extra={"session_id": session_id, "key": key},
            )

            return value

    def get_session_memory_usage(self, session_id: str) -> float:
        """
        Get memory usage for a specific session.

        Args:
            session_id: Session identifier

        Returns:
            Memory usage in MB, or 0.0 if session not found
        """
        with self._lock:
            if session_id not in self._sessions:
                return 0.0

            session = self._sessions[session_id]
            # Recalculate to ensure accuracy
            return session.estimate_memory_usage()

    def get_total_memory_usage(self) -> float:
        """
        Get total memory usage across all sessions.

        Returns:
            Total memory usage in MB
        """
        with self._lock:
            total_mb = sum(session.estimate_memory_usage() for session in self._sessions.values())
            return total_mb

    def get_active_session_count(self) -> int:
        """
        Get count of active sessions.

        Returns:
            Number of active sessions
        """
        with self._lock:
            return len(self._sessions)

    def _compress_value(self, value: Any) -> Optional[bytes]:
        """
        Compress a value if supported.

        Args:
            value: Value to compress

        Returns:
            Compressed bytes or None if compression not supported
        """
        try:
            # Compress numpy arrays as PNG images
            if isinstance(value, np.ndarray):
                if Image is None:
                    logger.warning("PIL not available, cannot compress numpy array")
                    return None

                # Convert to uint8 if needed
                if value.dtype != np.uint8:
                    # Normalize to 0-255 range
                    value_min = value.min()
                    value_max = value.max()
                    if value_max > value_min:
                        value_normalized = (
                            (value - value_min) / (value_max - value_min) * 255
                        ).astype(np.uint8)
                    else:
                        value_normalized = np.zeros_like(value, dtype=np.uint8)
                else:
                    value_normalized = value

                # Convert to PIL Image and compress
                if len(value_normalized.shape) == 2:
                    # Grayscale
                    img = Image.fromarray(value_normalized, mode="L")
                elif len(value_normalized.shape) == 3 and value_normalized.shape[2] == 3:
                    # RGB
                    img = Image.fromarray(value_normalized, mode="RGB")
                elif len(value_normalized.shape) == 3 and value_normalized.shape[2] == 4:
                    # RGBA
                    img = Image.fromarray(value_normalized, mode="RGBA")
                else:
                    logger.warning(
                        f"Unsupported array shape for compression: {value_normalized.shape}"
                    )
                    return None

                # Compress to PNG
                buffer = io.BytesIO()
                img.save(buffer, format="PNG", optimize=True)
                compressed = buffer.getvalue()

                logger.debug(
                    f"Compressed numpy array: {value.nbytes} -> {len(compressed)} bytes "
                    f"({len(compressed) / value.nbytes * 100:.1f}%)"
                )

                return compressed

            # Compress PIL Images
            elif Image is not None and isinstance(value, Image.Image):
                buffer = io.BytesIO()
                value.save(buffer, format="PNG", optimize=True)
                compressed = buffer.getvalue()

                logger.debug(f"Compressed PIL Image: {len(compressed)} bytes")

                return compressed

            else:
                # Compression not supported for this type
                return None

        except Exception as e:
            logger.warning(f"Failed to compress value: {e}")
            return None

    def _decompress_value(self, compressed: bytes) -> Any:
        """
        Decompress a compressed value.

        Args:
            compressed: Compressed bytes

        Returns:
            Decompressed value

        Raises:
            ValueError: If decompression fails
        """
        try:
            if Image is None:
                raise ValueError("PIL not available, cannot decompress")

            buffer = io.BytesIO(compressed)
            img = Image.open(buffer)

            # Convert to numpy array
            array = np.array(img)

            logger.debug(f"Decompressed image: {len(compressed)} bytes -> {array.nbytes} bytes")

            return array

        except Exception as e:
            logger.error(f"Failed to decompress value: {e}")
            raise ValueError(f"Decompression failed: {e}") from e

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop background tasks."""
        self.stop_background_cleanup()
        return False

    def cleanup_inactive_sessions(self) -> int:
        """
        Clean up sessions that have exceeded the timeout period.

        Returns:
            Number of sessions cleaned up
        """
        with self._lock:
            sessions_to_remove = []

            for session_id, session in self._sessions.items():
                if session.is_inactive(self.timeout_minutes):
                    sessions_to_remove.append(session_id)

            # Remove inactive sessions
            for session_id in sessions_to_remove:
                del self._sessions[session_id]
                logger.info(
                    f"Cleaned up inactive session '{session_id}'",
                    extra={
                        "session_id": session_id,
                        "timeout_minutes": self.timeout_minutes,
                    },
                )

            if sessions_to_remove:
                logger.info(
                    f"Cleaned up {len(sessions_to_remove)} inactive sessions",
                    extra={"count": len(sessions_to_remove)},
                )

            return len(sessions_to_remove)

    def cleanup_all_sessions(self) -> int:
        """
        Clean up all sessions regardless of activity status.
        Useful for memory pressure situations or shutdown.

        Returns:
            Number of sessions cleaned up
        """
        with self._lock:
            session_count = len(self._sessions)
            self._sessions.clear()

            if session_count > 0:
                logger.info(
                    f"Cleaned up all {session_count} sessions", extra={"count": session_count}
                )

            return session_count

    def start_background_cleanup(self) -> None:
        """
        Start background cleanup task that runs periodically.

        The cleanup task runs every 5 minutes and removes inactive sessions.
        """
        if self._running:
            logger.warning("Background cleanup already running")
            return

        self._running = True
        self._schedule_cleanup()

        logger.info(
            "Started background session cleanup", extra={"interval_sec": self._cleanup_interval_sec}
        )

    def stop_background_cleanup(self) -> None:
        """
        Stop the background cleanup task.
        """
        self._running = False

        if self._cleanup_timer is not None:
            self._cleanup_timer.cancel()
            self._cleanup_timer = None

        logger.info("Stopped background session cleanup")

    def _schedule_cleanup(self) -> None:
        """
        Schedule the next cleanup task.

        This method is called recursively to maintain periodic cleanup.
        """
        if not self._running:
            return

        # Run cleanup
        try:
            cleaned = self.cleanup_inactive_sessions()
            if cleaned > 0:
                logger.debug(
                    f"Background cleanup removed {cleaned} sessions",
                    extra={"cleaned_count": cleaned},
                )
        except Exception as e:
            logger.error(f"Error during background cleanup: {e}", exc_info=True)

        # Schedule next cleanup
        self._cleanup_timer = threading.Timer(self._cleanup_interval_sec, self._schedule_cleanup)
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()

    def reset_session(self, session_id: str, preserve_metadata: bool = True) -> None:
        """
        Reset a session by removing large data objects while optionally preserving metadata.

        Large objects include:
        - numpy arrays
        - PIL Images
        - Large strings/bytes (>1MB)
        - Lists/dicts containing large objects

        Metadata includes:
        - Small strings (<1KB)
        - Numbers, booleans
        - Small dictionaries with primitive values

        Args:
            session_id: Session identifier
            preserve_metadata: If True, keep small metadata objects
        """
        with self._lock:
            if session_id not in self._sessions:
                logger.warning(
                    f"Cannot reset session '{session_id}': not found",
                    extra={"session_id": session_id},
                )
                return

            session = self._sessions[session_id]
            keys_to_remove = []

            for key, value in session.data.items():
                if self._is_large_object(value):
                    keys_to_remove.append(key)

            # Remove large objects
            for key in keys_to_remove:
                del session.data[key]
                session.compressed_keys.discard(key)

            # Update memory usage
            old_memory = session.memory_usage_mb
            session.estimate_memory_usage()

            logger.info(
                f"Reset session '{session_id}': removed {len(keys_to_remove)} large objects",
                extra={
                    "session_id": session_id,
                    "removed_keys": len(keys_to_remove),
                    "memory_before_mb": old_memory,
                    "memory_after_mb": session.memory_usage_mb,
                    "memory_freed_mb": old_memory - session.memory_usage_mb,
                },
            )

    def _is_large_object(self, value: Any) -> bool:
        """
        Determine if a value is a large object that should be cleaned up.

        Args:
            value: Value to check

        Returns:
            True if value is considered a large object
        """
        try:
            # numpy arrays are always considered large
            if isinstance(value, np.ndarray):
                return True

            # PIL Images are always considered large
            if Image is not None and isinstance(value, Image.Image):
                return True

            # Large strings/bytes (>1MB)
            if isinstance(value, (str, bytes)):
                size_mb = sys.getsizeof(value) / (1024 * 1024)
                return size_mb > 1.0

            # Large lists/tuples (>1MB or containing arrays)
            if isinstance(value, (list, tuple)):
                # Check if contains numpy arrays
                if any(isinstance(item, np.ndarray) for item in value):
                    return True
                # Check size
                size_mb = sys.getsizeof(value) / (1024 * 1024)
                return size_mb > 1.0

            # Large dictionaries (>1MB or containing arrays)
            if isinstance(value, dict):
                # Check if contains numpy arrays
                if any(isinstance(v, np.ndarray) for v in value.values()):
                    return True
                # Check size
                size_mb = sys.getsizeof(value) / (1024 * 1024)
                return size_mb > 1.0

            # Everything else is considered small metadata
            return False

        except Exception as e:
            logger.warning(f"Error checking object size: {e}")
            # If we can't determine, assume it's large to be safe
            return True

    def cleanup_session_selective(self, session_id: str) -> float:
        """
        Selectively clean up large objects from a session.

        This is an alias for reset_session with preserve_metadata=True.

        Args:
            session_id: Session identifier

        Returns:
            Amount of memory freed in MB
        """
        with self._lock:
            if session_id not in self._sessions:
                return 0.0

            old_memory = self._sessions[session_id].memory_usage_mb

        self.reset_session(session_id, preserve_metadata=True)

        with self._lock:
            if session_id not in self._sessions:
                return old_memory

            new_memory = self._sessions[session_id].memory_usage_mb
            return old_memory - new_memory

    def cleanup_if_memory_exceeded(self) -> bool:
        """
        Check if total session memory exceeds limit and cleanup if needed.

        Cleanup strategy:
        1. Calculate total memory usage
        2. If exceeds limit, cleanup oldest inactive sessions first
        3. If still exceeds, cleanup oldest active sessions

        Returns:
            True if cleanup was performed, False otherwise
        """
        with self._lock:
            total_memory_mb = sum(
                session.estimate_memory_usage() for session in self._sessions.values()
            )
            max_memory_mb = self.max_total_memory_gb * 1024

            if total_memory_mb <= max_memory_mb:
                return False

            logger.warning(
                f"Total session memory ({total_memory_mb:.1f} MB) exceeds limit ({max_memory_mb:.1f} MB)",
                extra={
                    "total_memory_mb": total_memory_mb,
                    "max_memory_mb": max_memory_mb,
                    "session_count": len(self._sessions),
                },
            )

            # First, cleanup inactive sessions
            cleaned_count = 0
            sessions_to_cleanup = []

            # Sort sessions by last access (oldest first)
            sorted_sessions = sorted(self._sessions.items(), key=lambda x: x[1].last_access)

            # Identify inactive sessions first
            for session_id, session in sorted_sessions:
                if session.is_inactive(self.timeout_minutes):
                    sessions_to_cleanup.append(session_id)

            # Remove inactive sessions
            for session_id in sessions_to_cleanup:
                del self._sessions[session_id]
                cleaned_count += 1

                # Check if we're under limit now
                total_memory_mb = sum(s.estimate_memory_usage() for s in self._sessions.values())
                if total_memory_mb <= max_memory_mb:
                    logger.info(
                        f"Memory limit enforcement: removed {cleaned_count} inactive sessions",
                        extra={
                            "cleaned_count": cleaned_count,
                            "total_memory_mb": total_memory_mb,
                        },
                    )
                    return True

            # If still over limit, selectively cleanup oldest active sessions
            sessions_to_reset = []
            for session_id, session in sorted_sessions:
                if session_id in self._sessions:  # Still exists
                    sessions_to_reset.append(session_id)

        # Release lock before calling reset_session (which acquires its own lock)
        for session_id in sessions_to_reset:
            self.reset_session(session_id, preserve_metadata=True)
            cleaned_count += 1

            # Check if we're under limit now
            with self._lock:
                total_memory_mb = sum(s.estimate_memory_usage() for s in self._sessions.values())
                if total_memory_mb <= max_memory_mb:
                    logger.info(
                        f"Memory limit enforcement: cleaned {cleaned_count} sessions",
                        extra={
                            "cleaned_count": cleaned_count,
                            "total_memory_mb": total_memory_mb,
                        },
                    )
                    return True

        # Check final status
        with self._lock:
            total_memory_mb = sum(s.estimate_memory_usage() for s in self._sessions.values())

            # If we're still over limit, log warning
            if total_memory_mb > max_memory_mb:
                logger.warning(
                    f"Memory limit still exceeded after cleanup: {total_memory_mb:.1f} MB",
                    extra={
                        "total_memory_mb": total_memory_mb,
                        "max_memory_mb": max_memory_mb,
                        "cleaned_count": cleaned_count,
                    },
                )

            return True

    def get_memory_usage_stats(self) -> Dict[str, Any]:
        """
        Get detailed memory usage statistics.

        Returns:
            Dictionary with memory statistics
        """
        with self._lock:
            # Calculate total memory directly (avoid nested lock)
            total_memory_mb = sum(
                session.estimate_memory_usage() for session in self._sessions.values()
            )
            max_memory_mb = self.max_total_memory_gb * 1024

            session_stats = []
            for session_id, session in self._sessions.items():
                session_stats.append(
                    {
                        "session_id": session_id,
                        "memory_mb": session.memory_usage_mb,
                        "last_access": session.last_access.isoformat(),
                        "is_inactive": session.is_inactive(self.timeout_minutes),
                        "data_keys": len(session.data),
                        "compressed_keys": len(session.compressed_keys),
                    }
                )

            return {
                "total_memory_mb": total_memory_mb,
                "max_memory_mb": max_memory_mb,
                "memory_usage_percent": (
                    (total_memory_mb / max_memory_mb * 100) if max_memory_mb > 0 else 0
                ),
                "session_count": len(self._sessions),
                "sessions": session_stats,
            }
