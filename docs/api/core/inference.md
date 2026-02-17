# core.inference - Model Inference

The inference module provides a unified interface for running classification and segmentation models with automatic memory management and TensorFlow session cleanup.

## Table of Contents

- [InferenceEngine](#inferenceengine)
- [Classification Functions](#classification-functions)
- [Segmentation Functions](#segmentation-functions)
- [Grad-CAM Visualization](#grad-cam-visualization)

## InferenceEngine

**Class**: `core.inference.InferenceEngine`

Unified inference engine for classification and segmentation models with automatic model loading, caching, and memory cleanup.

### Constructor

```python
def __init__(
    self,
    cls_model_path: Optional[str] = None,
    seg_model_path: Optional[str] = None,
    memory_config: Optional[MemoryConfig] = None
) -> None
```

**Parameters**:
- `cls_model_path` (str, optional): Path to classification model. Defaults to `Config.CLS_MODEL_PATH`.
- `seg_model_path` (str, optional): Path to segmentation model. Defaults to `Config.SEG_MODEL_PATH`.
- `memory_config` (MemoryConfig, optional): Memory management configuration. Defaults to environment-based config.

**Example**:
```python
from core.inference import InferenceEngine

# Use default paths
engine = InferenceEngine()

# Custom paths
engine = InferenceEngine(
    cls_model_path="./models/custom_classifier.keras",
    seg_model_path="./models/custom_segmenter.keras"
)
```

### Methods

#### predict_classification

```python
def predict_classification(
    self,
    img: NDArray[np.float32],
    generate_heatmap: bool = True
) -> Tuple[int, float, Optional[NDArray[np.float32]]]
```

Predict tissue classification with optional Grad-CAM heatmap.

**Parameters**:
- `img` (NDArray[np.float32]): Preprocessed image array of shape (224, 224, 3), normalized to [0, 1]
- `generate_heatmap` (bool): Whether to generate Grad-CAM heatmap. Default: True

**Returns**:
- `class_idx` (int): Predicted class index (0=Benign, 1=Adenocarcinoma, 2=Squamous Cell Carcinoma)
- `confidence` (float): Prediction confidence score [0.0, 1.0]
- `heatmap` (NDArray[np.float32] | None): Grad-CAM heatmap of shape (224, 224) or None if not generated

**Raises**:
- `InferenceError`: If model loading or prediction fails
- `ModelNotFoundError`: If model file doesn't exist

**Example**:
```python
import numpy as np
from core.inference import InferenceEngine
from core.analysis import ImageProcessor

# Load and preprocess image
img = ImageProcessor.load_and_preprocess("sample.png")

# Run classification
engine = InferenceEngine()
class_idx, confidence, heatmap = engine.predict_classification(img)

print(f"Predicted class: {class_idx}")
print(f"Confidence: {confidence:.2%}")
print(f"Heatmap shape: {heatmap.shape if heatmap is not None else 'None'}")
```

**Source**: `core/inference/classifier.py:50-225`

#### predict_segmentation

```python
def predict_segmentation(
    self,
    img: NDArray[np.float32]
) -> Tuple[NDArray[np.uint8], NDArray[np.float32]]
```

Predict nucleus segmentation mask and probability map.

**Parameters**:
- `img` (NDArray[np.float32]): Preprocessed image array, normalized to [0, 1]

**Returns**:
- `mask` (NDArray[np.uint8]): Binary segmentation mask (0=background, 255=nucleus)
- `prob_map` (NDArray[np.float32]): Probability map [0.0, 1.0] for each pixel

**Raises**:
- `InferenceError`: If model loading or prediction fails
- `ModelNotFoundError`: If model file doesn't exist

**Example**:
```python
from core.inference import InferenceEngine
from core.analysis import ImageProcessor

# Load and preprocess image
img = ImageProcessor.load_and_preprocess("sample.png")

# Run segmentation
engine = InferenceEngine()
mask, prob_map = engine.predict_segmentation(img)

print(f"Mask shape: {mask.shape}")
print(f"Detected pixels: {(mask > 0).sum()}")
print(f"Mean probability: {prob_map.mean():.3f}")
```

**Source**: `core/inference/segmenter.py:47-120`

#### predict_with_cleanup

```python
def predict_with_cleanup(
    self,
    img: NDArray[np.float32],
    mode: str = "classification"
) -> Any
```

Run prediction with automatic TensorFlow session cleanup.

**Parameters**:
- `img` (NDArray[np.float32]): Preprocessed image array
- `mode` (str): Prediction mode ("classification" or "segmentation")

**Returns**:
- Classification: `(class_idx, confidence, heatmap)`
- Segmentation: `(mask, prob_map)`

**Raises**:
- `InferenceError`: If prediction fails
- `ValueError`: If mode is invalid

**Example**:
```python
engine = InferenceEngine()

# Classification with cleanup
result = engine.predict_with_cleanup(img, mode="classification")
class_idx, confidence, heatmap = result

# Segmentation with cleanup
result = engine.predict_with_cleanup(img, mode="segmentation")
mask, prob_map = result
```

**Source**: `core/inference/engine.py:229-280`

## Classification Functions

### predict_classification

**Function**: `core.inference.classifier.predict_classification`

```python
def predict_classification(
    img: NDArray[np.float32],
    model: Any,
    generate_heatmap: bool = True
) -> Tuple[int, float, Optional[NDArray[np.float32]]]
```

Low-level classification function (used internally by InferenceEngine).

**Parameters**:
- `img` (NDArray[np.float32]): Preprocessed image (224, 224, 3)
- `model`: Loaded Keras model
- `generate_heatmap` (bool): Generate Grad-CAM heatmap

**Returns**:
- `class_idx` (int): Predicted class index
- `confidence` (float): Confidence score
- `heatmap` (NDArray[np.float32] | None): Grad-CAM heatmap

**Note**: Prefer using `InferenceEngine.predict_classification()` for automatic model management.

**Source**: `core/inference/classifier.py:50-225`

## Segmentation Functions

### predict_segmentation

**Function**: `core.inference.segmenter.predict_segmentation`

```python
def predict_segmentation(
    img: NDArray[np.float32],
    model: Any,
    nuc_threshold: float = 0.4,
    con_threshold: float = 0.3
) -> Tuple[NDArray[np.uint8], NDArray[np.float32]]
```

Low-level segmentation function (used internally by InferenceEngine).

**Parameters**:
- `img` (NDArray[np.float32]): Preprocessed image
- `model`: Loaded Keras model
- `nuc_threshold` (float): Nucleus detection threshold [0.0, 1.0]. Default: 0.4
- `con_threshold` (float): Contour detection threshold [0.0, 1.0]. Default: 0.3

**Returns**:
- `mask` (NDArray[np.uint8]): Binary segmentation mask
- `prob_map` (NDArray[np.float32]): Probability map

**Note**: Prefer using `InferenceEngine.predict_segmentation()` for automatic model management.

**Source**: `core/inference/segmenter.py:47-120`

## Grad-CAM Visualization

### generate_gradcam

**Function**: `core.inference.classifier.generate_gradcam`

```python
def generate_gradcam(
    img: NDArray[np.float32],
    model: Any,
    class_idx: int,
    layer_name: Optional[str] = None
) -> NDArray[np.float32]
```

Generate Grad-CAM (Gradient-weighted Class Activation Mapping) heatmap for model interpretability.

**Parameters**:
- `img` (NDArray[np.float32]): Input image (224, 224, 3)
- `model`: Loaded Keras model
- `class_idx` (int): Target class index for visualization
- `layer_name` (str, optional): Convolutional layer name. Auto-detected if None.

**Returns**:
- `heatmap` (NDArray[np.float32]): Grad-CAM heatmap (224, 224) with values [0.0, 1.0]

**Raises**:
- `InferenceError`: If Grad-CAM generation fails

**Example**:
```python
from core.inference.classifier import generate_gradcam
import matplotlib.pyplot as plt

# Generate heatmap
heatmap = generate_gradcam(img, model, class_idx=1)

# Visualize
plt.imshow(heatmap, cmap='jet', alpha=0.5)
plt.colorbar()
plt.title("Grad-CAM Heatmap")
plt.show()
```

**Algorithm**:
1. Extract activations from last convolutional layer
2. Compute gradients of target class w.r.t. activations
3. Weight activations by gradients (global average pooling)
4. Apply ReLU to focus on positive contributions
5. Normalize to [0, 1] range

**Source**: `core/inference/classifier.py:226-427`

## Model Loading

Models are loaded lazily on first use and cached in an LRU cache (default: 2 models). Model integrity is verified using SHA256 checksums.

**Model Checksums** (defined in `config.settings.Config`):
- `efficientnetv2s_classification.keras`: `635f32b909057efe286a15fea16a80ec0adfede434dfafb9106fb0f6777f41d7`
- `cianet_segmentation.keras`: `74a8618c44734af796d35935c77bc94bef3aeb969be82c771fda03d0bb63e74f`

## Memory Management

The inference engine automatically:
- Cleans up TensorFlow sessions after prediction
- Releases GPU memory (if available)
- Triggers garbage collection after inference
- Monitors memory usage and prevents OOM errors

**Configuration** (via `MemoryConfig`):
- `enable_session_cleanup`: Enable TensorFlow session cleanup (default: True)
- `enable_gpu_memory_release`: Release GPU memory after inference (default: True)
- `enable_auto_gc`: Automatic garbage collection (default: True)

## Performance

**Typical Inference Times** (CPU, Intel i7):
- Classification: 2-5 seconds
- Segmentation: 5-10 seconds
- Grad-CAM generation: +0.5-1 second

**Memory Usage**:
- Classification model: ~200 MB
- Segmentation model: ~150 MB
- Inference overhead: ~100-500 MB (depends on image size)

## See Also

- [core.analysis](analysis.md): Image preprocessing and postprocessing
- [core.models](models.md): Model management and caching
- [core.memory](memory.md): Memory monitoring and cleanup
- [core.exceptions](exceptions.md): Exception handling

---

*Source files: `core/inference/engine.py`, `core/inference/classifier.py`, `core/inference/segmenter.py`*
