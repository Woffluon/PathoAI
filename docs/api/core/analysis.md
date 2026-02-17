# core.analysis - Image Processing

The analysis module provides image preprocessing, validation, and postprocessing utilities for histopathology image analysis.

## Modules

- **preprocessing**: Image validation and preprocessing
- **postprocessing**: Segmentation refinement and feature extraction

## Image Validation

### ImageValidator

**Class**: `core.analysis.preprocessing.ImageValidator`

Validates uploaded images for security and format compliance.

#### validate_image

```python
@staticmethod
def validate_image(file_bytes: bytes) -> Image.Image
```

Validate and load image from bytes.

**Parameters**:
- `file_bytes` (bytes): Raw image file bytes

**Returns**:
- `Image.Image`: Validated PIL Image object

**Raises**:
- `ImageValidationError`: If validation fails (format, size, dimensions)

**Validation Checks**:
- File size ≤ 50 MB
- Image dimensions ≤ 100 megapixels
- Format in {PNG, JPEG, TIFF}
- Valid image data (not corrupted)

**Example**:
```python
from core.analysis import ImageValidator

with open("sample.png", "rb") as f:
    file_bytes = f.read()

try:
    img = ImageValidator.validate_image(file_bytes)
    print(f"Valid image: {img.size}")
except ImageValidationError as e:
    print(f"Validation failed: {e}")
```

**Source**: `core/analysis/preprocessing.py:52-110`

## Image Preprocessing

### ImageProcessor

**Class**: `core.analysis.preprocessing.ImageProcessor`

Image preprocessing utilities including resizing and normalization.

#### smart_resize

```python
@staticmethod
def smart_resize(
    img: np.ndarray,
    max_dim: Optional[int] = None
) -> Tuple[np.ndarray, float]
```

Resize image while preserving aspect ratio.

**Parameters**:
- `img` (np.ndarray): Input image array
- `max_dim` (int, optional): Maximum dimension. Default: 2048

**Returns**:
- `resized` (np.ndarray): Resized image
- `scale` (float): Scaling factor applied

**Example**:
```python
from core.analysis import ImageProcessor
import numpy as np

img = np.random.rand(4000, 3000, 3)
resized, scale = ImageProcessor.smart_resize(img, max_dim=2048)
print(f"Original: {img.shape}, Resized: {resized.shape}, Scale: {scale:.3f}")
```

**Source**: `core/analysis/preprocessing.py:120-148`

#### macenko_normalize

```python
@staticmethod
def macenko_normalize(img: np.ndarray) -> np.ndarray
```

Apply Macenko stain normalization for consistent color appearance.

**Parameters**:
- `img` (np.ndarray): Input RGB image

**Returns**:
- `normalized` (np.ndarray): Stain-normalized image

**Algorithm**:
1. Convert RGB to optical density (OD)
2. Estimate stain matrix using SVD
3. Normalize stain concentrations
4. Reconstruct RGB image

**Note**: Falls back to original image if normalization fails.

**Source**: `core/analysis/preprocessing.py` (inferred from imports)

## Segmentation Postprocessing

### adaptive_watershed

```python
def adaptive_watershed(
    prob_map: NDArray[np.float32],
    threshold: float = 0.4
) -> NDArray[np.int32]
```

Apply adaptive watershed segmentation to separate overlapping nuclei.

**Parameters**:
- `prob_map` (NDArray[np.float32]): Probability map from segmentation model
- `threshold` (float): Detection threshold [0.0, 1.0]. Default: 0.4

**Returns**:
- `labels` (NDArray[np.int32]): Labeled segmentation mask (0=background, 1+=cells)

**Algorithm**:
1. Threshold probability map
2. Compute distance transform
3. Detect local maxima as seeds
4. Apply marker-based watershed
5. Remove small objects (noise filtering)

**Example**:
```python
from core.analysis.postprocessing import adaptive_watershed
import numpy as np

prob_map = np.random.rand(512, 512).astype(np.float32)
labels = adaptive_watershed(prob_map, threshold=0.4)
print(f"Detected {labels.max()} cells")
```

**Source**: `core/analysis/postprocessing.py:31-73`

### calculate_morphometrics

```python
def calculate_morphometrics(
    label_mask: NDArray[np.int32]
) -> pd.DataFrame
```

Calculate morphometric features for each segmented cell.

**Parameters**:
- `label_mask` (NDArray[np.int32]): Labeled segmentation mask

**Returns**:
- `df` (pd.DataFrame): DataFrame with columns:
  - `cell_id`: Cell identifier
  - `area`: Cell area (pixels)
  - `perimeter`: Cell perimeter (pixels)
  - `circularity`: Shape circularity [0.0, 1.0]
  - `centroid_x`: X coordinate of centroid
  - `centroid_y`: Y coordinate of centroid

**Formulas**:
- Circularity = 4π × area / perimeter²
- Perfect circle: circularity = 1.0
- Irregular shape: circularity < 1.0

**Example**:
```python
from core.analysis.postprocessing import calculate_morphometrics

df = calculate_morphometrics(labels)
print(f"Mean area: {df['area'].mean():.1f} pixels")
print(f"Mean circularity: {df['circularity'].mean():.3f}")
```

**Source**: `core/analysis/postprocessing.py:75-109`

### calculate_entropy

```python
def calculate_entropy(
    prob_map: NDArray[np.float32]
) -> NDArray[np.float32]
```

Calculate entropy-based uncertainty for each pixel.

**Parameters**:
- `prob_map` (NDArray[np.float32]): Probability map [0.0, 1.0]

**Returns**:
- `entropy` (NDArray[np.float32]): Entropy map [0.0, 1.0]

**Formula**:
- Entropy = -p × log(p) - (1-p) × log(1-p)
- High entropy: uncertain prediction
- Low entropy: confident prediction

**Example**:
```python
from core.analysis.postprocessing import calculate_entropy
import numpy as np

prob_map = np.random.rand(512, 512).astype(np.float32)
entropy = calculate_entropy(prob_map)
print(f"Mean uncertainty: {entropy.mean():.3f}")
```

**Source**: `core/analysis/postprocessing.py:111-140`

## Complete Preprocessing Pipeline

**Example**: Full preprocessing pipeline

```python
from core.analysis import ImageValidator, ImageProcessor
import numpy as np

# 1. Validate uploaded image
with open("sample.png", "rb") as f:
    file_bytes = f.read()

img = ImageValidator.validate_image(file_bytes)

# 2. Convert to numpy array
img_array = np.array(img)

# 3. Smart resize
resized, scale = ImageProcessor.smart_resize(img_array, max_dim=2048)

# 4. Macenko normalization
normalized = ImageProcessor.macenko_normalize(resized)

# 5. Resize to model input size
final = cv2.resize(normalized, (224, 224))
final = final.astype(np.float32) / 255.0

# Ready for inference
print(f"Preprocessed image shape: {final.shape}")
```

## Complete Postprocessing Pipeline

**Example**: Full postprocessing pipeline

```python
from core.analysis.postprocessing import (
    adaptive_watershed,
    calculate_morphometrics,
    calculate_entropy
)

# 1. Apply watershed segmentation
labels = adaptive_watershed(prob_map, threshold=0.4)

# 2. Calculate morphometric features
df = calculate_morphometrics(labels)

# 3. Calculate uncertainty
entropy = calculate_entropy(prob_map)

# 4. Analyze results
print(f"Total cells: {labels.max()}")
print(f"Mean area: {df['area'].mean():.1f} pixels")
print(f"Mean circularity: {df['circularity'].mean():.3f}")
print(f"Mean uncertainty: {entropy.mean():.3f}")
```

## Configuration

**Image Processing Parameters** (from `config.settings.Config`):

```python
MIN_CELL_AREA_PIXELS = 30              # Minimum cell area for noise filtering
MORPHOLOGY_KERNEL_SIZE = (3, 3)        # Morphological operations kernel
PEAK_DETECTION_FOOTPRINT = (5, 5)     # Peak detection footprint
PEAK_MIN_DISTANCE = 5                  # Minimum distance between peaks
MACENKO_IO = 240                       # Transmitted light intensity
MACENKO_ALPHA = 1.0                    # Percentile for stain estimation
MACENKO_BETA = 0.15                    # OD threshold
SMART_RESIZE_MAX_DIM = 2048            # Maximum dimension for resizing
MACENKO_DOWNSAMPLE_MAX = 1024          # Maximum dimension for SVD
```

## See Also

- [core.inference](inference.md): Model inference
- [core.memory](memory.md): Memory management
- [config.settings](../config/settings.md): Configuration

---

*Source files: `core/analysis/preprocessing.py`, `core/analysis/postprocessing.py`*
