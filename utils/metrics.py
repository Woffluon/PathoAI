"""
Prometheus metrics collection module for PathoAI.

This module defines and exposes Prometheus metrics for monitoring:
- Request counts and status
- Inference duration histograms
- Error counts by type
- Memory usage
- Active sessions
- Model predictions by class

Typical Usage:
    >>> from utils import track_inference_time, start_metrics_server  # Recommended
    >>> from utils import REQUEST_COUNT, ERROR_COUNT, MEMORY_USAGE
    >>>
    >>> # Start metrics server
    >>> start_metrics_server(port=8000)
    >>>
    >>> # Track inference time
    >>> @track_inference_time('classifier')
    >>> def run_inference(data):
    ...     return model.predict(data)
    >>>
    >>> # Update metrics
    >>> REQUEST_COUNT.labels(endpoint='/analyze', status='success').inc()
    >>> ERROR_COUNT.labels(error_type='validation_error').inc()

Import Paths:
    >>> from utils import track_inference_time, start_metrics_server  # Recommended
    >>> from utils.metrics import track_inference_time  # Also valid
"""

import functools
import logging
import time
from threading import Thread
from typing import Callable

import psutil
from flask import Flask, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

logger = logging.getLogger(__name__)

# Metric definitions

REQUEST_COUNT = Counter(
    "pathoai_requests_total", "Total number of requests", ["endpoint", "status"]
)

INFERENCE_DURATION = Histogram(
    "pathoai_inference_seconds",
    "Inference duration in seconds",
    ["model_type"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

ERROR_COUNT = Counter("pathoai_errors_total", "Total number of errors", ["error_type"])

MEMORY_USAGE = Gauge("pathoai_memory_bytes", "Memory usage in bytes")

ACTIVE_SESSIONS = Gauge("pathoai_active_sessions", "Number of active sessions")

MODEL_PREDICTIONS = Counter(
    "pathoai_predictions_total", "Total number of predictions", ["model", "predicted_class"]
)


def track_inference_time(model_type: str) -> Callable:
    """
    Decorator factory to track inference duration.

    Measures function execution time and records it in the INFERENCE_DURATION
    histogram. On exception, increments ERROR_COUNT and re-raises.

    Args:
        model_type: Type of model ("classifier" or "segmenter")

    Returns:
        Decorator function that wraps the target function

    Example:
        @track_inference_time(model_type="classifier")
        def predict_classification(self, img):
            # ... inference code ...
            return result
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                INFERENCE_DURATION.labels(model_type=model_type).observe(duration)
                return result
            except Exception as e:
                ERROR_COUNT.labels(error_type=type(e).__name__).inc()
                raise

        return wrapper

    return decorator


def update_memory_metrics() -> None:
    """
    Update memory usage gauge with current process memory.

    Uses psutil to get the current process's resident set size (RSS)
    and updates the MEMORY_USAGE gauge.
    """
    try:
        memory_bytes = psutil.Process().memory_info().rss
        MEMORY_USAGE.set(memory_bytes)
    except Exception as e:
        logger.warning(f"Failed to update memory metrics: {e}")


def start_metrics_server(port: int = 9090) -> None:
    """
    Start Flask metrics server in background daemon thread.

    Creates a Flask app with a /metrics endpoint that returns Prometheus
    text format metrics. The server runs in a daemon thread so it doesn't
    block application shutdown.

    Args:
        port: Port to bind the metrics server (default: 9090)

    Note:
        Binds to 0.0.0.0 to accept connections from any interface.
        Handles port conflicts gracefully by logging a warning.
        Uses singleton pattern to prevent multiple server instances.
    """
    # Singleton pattern - prevent multiple server instances
    if hasattr(start_metrics_server, '_server_started'):
        logger.debug(f"Metrics server already running on port {port}")
        return
    
    app = Flask(__name__)

    @app.route("/metrics")
    def metrics():
        """Prometheus metrics endpoint."""
        update_memory_metrics()
        return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

    def run_server():
        try:
            logger.info(f"Starting metrics server on port {port}")
            app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False)
        except OSError as e:
            logger.warning(f"Failed to start metrics server on port {port}: {e}")
        except Exception as e:
            logger.error(f"Metrics server error: {e}", exc_info=True)

    # Start server in daemon thread
    thread = Thread(target=run_server, daemon=True)
    thread.start()
    start_metrics_server._server_started = True
    logger.info("Metrics server thread started (daemon mode)")
