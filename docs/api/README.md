# API Reference

This section provides detailed documentation for all public APIs in PathoAI.

## Module Organization

PathoAI is organized into three main packages:

### Core Package (`core/`)

The core package contains the main business logic for image analysis, model inference, and memory management.

- [**inference**](core/inference.md): Model inference engine and prediction modules
  - `InferenceEngine`: Unified interface for classification and segmentation
  - `predict_classification()`: Tissue classification with Grad-CAM
  - `predict_segmentation()`: Nucleus segmentation with probability maps
  - `generate_gradcam()`: Grad-CAM heatmap generation

- [**analysis**](core/analysis.md): Image preprocessing and postprocessing
  - `ImageValidator`: Image validation and security checks
  - `ImageProcessor`: Image preprocessing (resizing, normalization)
  - `adaptive_watershed()`: Nucleus separation algorithm
  - `calculate_morphometrics()`: Cell feature extraction
  - `calculate_entropy()`: Uncertainty quantification

- [**memory**](core/memory.md): Memory management and session handling
  - `MemoryMonitor`: Real-time memory usage tracking
  - `MemoryConfig`: Memory management configuration
  - `SessionManager`: User session lifecycle management
  - `GarbageCollectionManager`: Automatic garbage collection

- [**models**](core/models.md): Model loading and caching
  - `ModelManager`: Model lifecycle management with LRU cache

- [**exceptions**](core/exceptions.md): Custom exception hierarchy
  - `PathoAIException`: Base exception class
  - `InferenceError`: Model inference failures
  - `ImageValidationError`: Image validation failures
  - `MemoryManagementError`: Memory-related errors

- [**performance_metrics**](core/performance_metrics.md): Performance tracking
  - `PerformanceMetrics`: Timing and resource usage tracking

### UI Package (`ui/`)

The UI package contains Streamlit dashboard components and visualization utilities.

- [**dashboard**](ui/dashboard.md): Streamlit UI components
  - `render_header()`: Application header and branding
  - `render_css()`: Custom CSS styling
  - `render_classification_panel()`: Classification results display
  - `render_segmentation_panel()`: Segmentation results display
  - `render_metric_card()`: Metric card component
  - `apply_heatmap_overlay()`: Heatmap visualization
  - `get_disease_info()`: Disease information lookup

### Utils Package (`utils/`)

The utils package provides cross-cutting utilities for logging, metrics, and auditing.

- [**audit_logger**](utils/audit_logger.md): Compliance-focused event logging
  - `AuditLogger`: Audit event tracking

- [**metrics**](utils/metrics.md): Prometheus metrics export
  - `track_inference_time()`: Inference duration tracking
  - `update_memory_metrics()`: Memory usage monitoring
  - `start_metrics_server()`: Metrics HTTP server

### Configuration (`config/`)

- [**settings**](config/settings.md): Application configuration
  - `Config`: Centralized configuration management
  - `JSONFormatter`: Structured JSON logging

## Quick Reference

### Common Tasks

**Load and run inference**:
```python
from core import InferenceEngine
import numpy as np

# Initialize engine
engine = InferenceEngine()

# Classification
img = np.random.rand(224, 224, 3).astype(np.float32)
class_idx, confidence, tensor = engine.predict_classification(img)

# Segmentation
mask, prob_map = engine.predict_segmentation(img)
```

**Validate and preprocess images**:
```python
from core.analysis import ImageValidator, ImageProcessor

# Validate uploaded image
img = ImageValidator.validate_image(file_bytes)

# Smart resize
resized, scale = ImageProcessor.smart_resize(img_array, max_dim=2048)

# Macenko normalization
normalized = ImageProcessor.macenko_normalize(img_array)
```

**Monitor memory usage**:
```python
from core.memory import MemoryMonitor, MemoryConfig

# Initialize monitor
config = MemoryConfig.from_env()
monitor = MemoryMonitor(config)

# Check available memory
metrics = monitor.get_current_metrics()
print(f"Memory usage: {metrics.percent_used}%")

# Check if operation is safe
if monitor.check_memory_available(required_mb=1500):
    # Proceed with operation
    pass
```

**Manage sessions**:
```python
from core.memory import SessionManager

# Initialize manager
manager = SessionManager(timeout_minutes=30)

# Create session
session_id = manager.create_session()

# Store data
manager.update_session(session_id, uploaded_image=img)

# Cleanup inactive sessions
cleaned = manager.cleanup_inactive_sessions()
```

**Track metrics**:
```python
from utils.metrics import track_inference_time, update_memory_metrics

# Track inference duration
@track_inference_time("classification")
def run_classification(img):
    return model.predict(img)

# Update memory metrics
update_memory_metrics()
```

**Log audit events**:
```python
from utils.audit_logger import AuditLogger

# Log image upload
AuditLogger.log_image_upload(
    session_id="abc123",
    file_size=1024000,
    image_format="PNG"
)

# Log analysis completion
AuditLogger.log_analysis_complete(
    session_id="abc123",
    analysis_type="classification",
    diagnosis="Benign",
    confidence=0.95
)
```

## Type Hints

PathoAI uses type hints throughout the codebase for better IDE support and type checking. Common types:

```python
from typing import Tuple, Optional, Dict, Any
from numpy.typing import NDArray
import numpy as np

# Image arrays
ImageArray = NDArray[np.uint8]  # 8-bit image
FloatArray = NDArray[np.float32]  # Float image

# Common return types
ClassificationResult = Tuple[int, float, NDArray[np.float32]]
SegmentationResult = Tuple[NDArray[np.uint8], NDArray[np.float32]]
```

## Error Handling

All public APIs raise specific exceptions for different error conditions:

```python
from core.exceptions import (
    ImageValidationError,
    InferenceError,
    InsufficientMemoryError,
    ModelNotFoundError
)

try:
    # Validate image
    img = ImageValidator.validate_image(file_bytes)
except ImageValidationError as e:
    print(f"Invalid image: {e}")

try:
    # Run inference
    result = engine.predict_classification(img)
except InferenceError as e:
    print(f"Inference failed: {e}")
except InsufficientMemoryError as e:
    print(f"Not enough memory: {e}")
```

## Module Index

- [core.inference](core/inference.md)
- [core.analysis](core/analysis.md)
- [core.memory](core/memory.md)
- [core.models](core/models.md)
- [core.exceptions](core/exceptions.md)
- [core.performance_metrics](core/performance_metrics.md)
- [ui.dashboard](ui/dashboard.md)
- [utils.audit_logger](utils/audit_logger.md)
- [utils.metrics](utils/metrics.md)
- [config.settings](config/settings.md)

---

*For architecture details, see the [Architecture Overview](../architecture.md)*
