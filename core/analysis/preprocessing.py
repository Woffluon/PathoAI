"""
Image Preprocessing Module

This module provides memory-efficient image preprocessing utilities for medical
histopathology images. It includes validation, loading, resizing, normalization,
and stain normalization capabilities.

Typical Usage:
    >>> from core.analysis import ImageProcessor, ImageValidator  # Recommended
    >>>
    >>> # Validate uploaded image
    >>> validator = ImageValidator()
    >>> validated_img = validator.validate_image(file_bytes)
    >>>
    >>> # Process image
    >>> processor = ImageProcessor()
    >>> normalized = processor.macenko_normalize(image)
    >>> resized, scale = processor.smart_resize(image, max_dim=2048)

Import Paths:
    >>> from core.analysis import ImageProcessor, ImageValidator  # Recommended
    >>> from core.analysis.preprocessing import ImageProcessor  # Also valid
"""

import io
import logging
import os
import tempfile
from typing import Generator, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from config import Config
from core.exceptions import ImageValidationError
from core.memory import GarbageCollectionManager, MemoryConfig

logger = logging.getLogger(__name__)


class ImageValidator:
    """
    Security-focused validator for uploaded medical images.

    This class provides static methods for validating and sanitizing uploaded
    image files to prevent security vulnerabilities and ensure data integrity.
    """

    @staticmethod
    def validate_image(file_bytes: bytes) -> Image.Image:
        """
        Validate and sanitize uploaded medical image for security and integrity.

        Args:
            file_bytes: Raw bytes of the uploaded image file.

        Returns:
            PIL.Image.Image: Validated and sanitized image in RGB color space.

        Raises:
            ValueError: If any validation check fails.
        """
        # 1. Check file size
        if len(file_bytes) > Config.MAX_FILE_SIZE:
            error_msg = (
                f"File too large: {len(file_bytes)/1024/1024:.1f}MB exceeds "
                f"maximum allowed size of {Config.MAX_FILE_SIZE/1024/1024:.0f}MB"
            )
            logger.warning(f"Image validation failed: {error_msg}")
            raise ImageValidationError(error_msg)

        # 2. Verify image format and integrity
        try:
            img = Image.open(io.BytesIO(file_bytes))
            img.verify()
        except ValueError as e:
            error_msg = f"Invalid or corrupted image file: {str(e)}"
            logger.warning(f"Image validation failed: {error_msg}")
            raise ImageValidationError(error_msg) from e

        # Re-open image after verify (verify closes the file)
        img = Image.open(io.BytesIO(file_bytes))

        # Check format
        if img.format not in Config.ALLOWED_FORMATS:
            error_msg = (
                f"Unsupported format: {img.format}. "
                f"Allowed formats: {', '.join(Config.ALLOWED_FORMATS)}"
            )
            logger.warning(f"Image validation failed: {error_msg}")
            raise ImageValidationError(error_msg)

        # 3. Check pixel dimensions
        total_pixels = img.width * img.height
        if total_pixels > Config.MAX_PIXELS:
            error_msg = (
                f"Image too large: {img.width}x{img.height} "
                f"({total_pixels/1000000:.1f} megapixels) exceeds "
                f"maximum of {Config.MAX_PIXELS/1000000:.0f} megapixels"
            )
            logger.warning(f"Image validation failed: {error_msg}")
            raise ImageValidationError(error_msg)

        # 4. Remove EXIF and convert to RGB
        img_clean = img.convert("RGB")

        return img_clean


class ImageProcessor:
    """
    Memory-efficient image preprocessing utilities for medical histopathology images.
    """

    MAX_DIMENSION = Config.SMART_RESIZE_MAX_DIM

    @staticmethod
    def smart_resize(img: np.ndarray, max_dim: Optional[int] = None) -> Tuple[np.ndarray, float]:
        """
        Intelligently downsample large images to reduce memory and computation time.

        Args:
            img: Input image as numpy array with shape (H, W, C).
            max_dim: Maximum allowed dimension in pixels.

        Returns:
            Tuple[np.ndarray, float]: Resized image and scale factor.
        """
        if max_dim is None:
            max_dim = ImageProcessor.MAX_DIMENSION

        h, w = img.shape[:2]

        # Check if resize needed
        if h <= max_dim and w <= max_dim:
            return img, 1.0

        # Calculate scale to fit within max_dim
        scale = max_dim / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Use INTER_AREA for downsampling (best quality)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return resized, scale

    @staticmethod
    def load_image_mmap(path: str, memory_config: Optional[MemoryConfig] = None) -> np.ndarray:
        """
        Load image using memory-mapped file access for efficient large image handling.

        Args:
            path: Path to the image file.
            memory_config: Optional MemoryConfig instance.

        Returns:
            np.ndarray: Image array (regular or memory-mapped).

        Raises:
            FileNotFoundError: If the image file does not exist.
            ValueError: If the image cannot be loaded or decoded.
        """
        if not os.path.exists(path):
            logger.error(f"Image file not found: {path}")
            raise FileNotFoundError(f"Image file not found: {path}")

        if memory_config is None:
            memory_config = MemoryConfig()

        try:
            # Load image to check dimensions
            img = Image.open(path)
            width, height = img.size
            total_pixels = width * height

            # Check if image exceeds threshold for memory-mapped loading
            if total_pixels > memory_config.mmap_threshold_pixels:
                # Convert to numpy array and save to temporary file for mmap
                img_array = np.array(img)

                # Create temporary file for memory mapping
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
                temp_path = temp_file.name
                temp_file.close()

                # Save array to temp file
                np.save(temp_path, img_array)

                # Load as memory-mapped array
                mmap_array = np.load(temp_path, mmap_mode="r")

                # Store temp path as attribute for cleanup
                mmap_array._temp_path = temp_path

                return mmap_array
            else:
                # Small image - load normally
                return np.array(img)

        except FileNotFoundError:
            raise
        except ValueError as e:
            logger.error(f"Failed to load or decode image from {path}: {e}")
            raise ValueError(f"Failed to load image from {path}: {e}") from e

    @staticmethod
    def generate_tiles(
        image: np.ndarray, tile_size: Tuple[int, int], overlap: int = 0
    ) -> Generator[Tuple[np.ndarray, Tuple[int, int]], None, None]:
        """
        Generate tiles from large image using memory-efficient generator pattern.

        Args:
            image: Source image array.
            tile_size: Tuple of (tile_height, tile_width) in pixels.
            overlap: Overlap between adjacent tiles in pixels.

        Yields:
            Tuple[np.ndarray, Tuple[int, int]]: Tile array and position.

        Raises:
            ValueError: If overlap is larger than or equal to tile size.
        """
        height, width = image.shape[:2]
        tile_h, tile_w = tile_size

        # Calculate step size (tile size minus overlap)
        step_h = tile_h - overlap
        step_w = tile_w - overlap

        # Ensure step size is positive
        if step_h <= 0 or step_w <= 0:
            raise ValueError("Overlap must be smaller than tile size")

        # Generate tiles
        for row in range(0, height, step_h):
            for col in range(0, width, step_w):
                # Calculate tile boundaries
                row_end = min(row + tile_h, height)
                col_end = min(col + tile_w, width)

                # Extract tile
                tile = image[row:row_end, col:col_end]

                # Yield tile and its position
                yield tile, (row, col)

    @staticmethod
    def normalize_inplace(
        image: NDArray[np.uint8], mean: NDArray[np.float32], std: NDArray[np.float32]
    ) -> NDArray[np.uint8]:
        """
        Normalize image in-place to reduce memory overhead.

        Args:
            image: Image array to normalize (modified in-place).
            mean: Mean values for normalization.
            std: Standard deviation values.

        Returns:
            NDArray[np.uint8]: Same array reference (modified in-place).

        Raises:
            ValueError: If array is not writeable.
        """
        if not image.flags.writeable:
            raise ValueError("Cannot normalize in-place: array is not writeable")

        # Perform in-place normalization
        image -= mean
        image /= std

        return image

    @staticmethod
    def convert_colorspace_inplace(image: NDArray[np.uint8], conversion: str) -> NDArray[np.uint8]:
        """
        Convert image color space with minimal memory overhead.

        Args:
            image: Image array to convert (may be modified in-place).
            conversion: OpenCV conversion code string (e.g., 'RGB2LAB').

        Returns:
            NDArray[np.uint8]: Converted image array.

        Raises:
            ValueError: If array is not writeable or conversion code is invalid.
        """
        if not image.flags.writeable:
            raise ValueError("Cannot convert in-place: array is not writeable")

        # Get OpenCV conversion code
        conversion_code = getattr(cv2, f"COLOR_{conversion}", None)
        if conversion_code is None:
            raise ValueError(f"Invalid conversion: {conversion}")

        # Perform conversion
        converted = cv2.cvtColor(image, conversion_code)

        # If dimensions match, copy back to original array
        if converted.shape == image.shape:
            np.copyto(image, converted)
            return image
        else:
            # Dimension changed, return new array
            return converted

    @staticmethod
    def resize_inplace(image: NDArray[np.uint8], target_size: Tuple[int, int]) -> NDArray[np.uint8]:
        """
        Resize image with minimal memory overhead.

        Args:
            image: Image array to resize.
            target_size: Target dimensions as (height, width) tuple.

        Returns:
            NDArray[np.uint8]: Resized image array.
        """
        # OpenCV resize (efficient implementation)
        resized = cv2.resize(image, (target_size[1], target_size[0]))

        return resized

    @staticmethod
    def cleanup_mmap(image: NDArray) -> None:
        """
        Clean up memory-mapped array and delete its temporary backing file.

        Args:
            image: Memory-mapped numpy array with `_temp_path` attribute.
        """
        if hasattr(image, "_temp_path"):
            temp_path = image._temp_path
            # Delete the array reference
            del image
            # Remove temporary file
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception:
                pass  # Ignore cleanup errors

    @staticmethod
    def process_with_cleanup(
        operation_func, *args, gc_manager: Optional[GarbageCollectionManager] = None, **kwargs
    ):
        """
        Execute image processing operation with automatic garbage collection.

        Args:
            operation_func: Callable function to execute.
            *args: Positional arguments to pass to operation_func.
            gc_manager: Optional GarbageCollectionManager instance.
            **kwargs: Keyword arguments to pass to operation_func.

        Returns:
            Any: The return value of operation_func.
        """
        try:
            result = operation_func(*args, **kwargs)
            return result
        finally:
            # Trigger garbage collection
            if gc_manager is not None:
                gc_manager.collect_with_stats(generation=0)
            else:
                import gc

                gc.collect()

    @staticmethod
    def macenko_normalize(
        img: NDArray[np.uint8],
        Io: int = Config.MACENKO_IO,
        alpha: float = Config.MACENKO_ALPHA,
        beta: float = Config.MACENKO_BETA,
    ) -> NDArray[np.uint8]:
        """
        Normalize H&E staining appearance using Macenko method with memory optimization.

        Args:
            img: Input RGB image as numpy array.
            Io: Transmitted light intensity (default: 240).
            alpha: Percentile for stain vector estimation (default: 1).
            beta: Optical density threshold (default: 0.15).

        Returns:
            NDArray[np.uint8]: Normalized RGB image.
        """
        try:
            # Downsample for SVD computation if image is large
            img_small, scale = ImageProcessor.smart_resize(
                img, max_dim=Config.MACENKO_DOWNSAMPLE_MAX
            )

            # Perform normalization on smaller image
            HER = np.array([[0.650, 0.704, 0.286], [0.072, 0.990, 0.105], [0.268, 0.570, 0.776]])
            h, w, c = img_small.shape

            # Use memory-efficient reshape
            img_flat = img_small.reshape((-1, 3))
            OD = -np.log((img_flat.astype(float) + 1) / Io)

            ODhat = OD[np.all(OD > beta, axis=1)]
            if len(ODhat) < 10:
                return img

            eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
            That = ODhat.dot(eigvecs[:, 1:3])
            phi = np.arctan2(That[:, 1], That[:, 0])
            minPhi = np.percentile(phi, alpha)
            maxPhi = np.percentile(phi, 100 - alpha)

            vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
            vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

            if vMin[0] > vMax[0]:
                HE = np.array((vMin[:, 0], vMax[:, 0])).T
            else:
                HE = np.array((vMax[:, 0], vMin[:, 0])).T

            Y = np.reshape(OD, (-1, 3)).T
            C = np.linalg.lstsq(HE, Y, rcond=None)[0]
            np.array([1.9705, 1.0308])  # maxC reference values  # noqa: F841

            Inorm = Io * np.exp(-HER[:, 0:2].dot(C))
            normalized_small = np.clip(np.reshape(Inorm.T, (h, w, c)), 0, 255).astype(np.uint8)

            # Upscale back to original size if needed
            if scale < 1.0:
                orig_h, orig_w = img.shape[:2]
                normalized = cv2.resize(
                    normalized_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
                )
            else:
                normalized = normalized_small

            # Trigger garbage collection after processing
            gc_manager = GarbageCollectionManager()
            gc_manager.collect_with_stats(generation=0)

            return normalized
        except np.linalg.LinAlgError as e:
            logger.warning(f"Macenko normalization failed due to linear algebra error: {e}")
            return img
        except ValueError as e:
            logger.warning(f"Macenko normalization failed due to invalid value: {e}")
            return img
        except Exception as e:
            logger.error(f"Unexpected error in Macenko normalization: {e}", exc_info=True)
            return img
