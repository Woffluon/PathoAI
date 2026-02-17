"""
Image Analysis Module

This module provides comprehensive image analysis capabilities for medical
histopathology images, including preprocessing, postprocessing, validation,
and feature extraction.

Main Components:
    - ImageValidator: Security-focused image validation and sanitization
    - ImageProcessor: Unified interface for preprocessing and postprocessing

Preprocessing Functions (via ImageProcessor):
    - smart_resize(): Intelligent downsampling for large images
    - load_image_mmap(): Memory-mapped file loading
    - generate_tiles(): Generator for tile-based processing
    - normalize_inplace(): In-place normalization
    - convert_colorspace_inplace(): Color space conversion
    - resize_inplace(): Image resizing
    - macenko_normalize(): H&E stain normalization
    - cleanup_mmap(): Memory-mapped array cleanup
    - process_with_cleanup(): Processing with garbage collection

Postprocessing Functions (via ImageProcessor):
    - adaptive_watershed(): Cell segmentation using probability maps
    - calculate_morphometrics(): Biological feature extraction
    - calculate_entropy(): Uncertainty quantification

Typical Usage:
    >>> from core.analysis import ImageValidator, ImageProcessor
    >>>
    >>> # Validate uploaded image
    >>> img = ImageValidator.validate_image(file_bytes)
    >>>
    >>> # Preprocess image
    >>> img_array = np.array(img)
    >>> normalized = ImageProcessor.macenko_normalize(img_array)
    >>>
    >>> # Postprocess segmentation results
    >>> mask = ImageProcessor.adaptive_watershed(nuc_prob, con_prob)
    >>> stats = ImageProcessor.calculate_morphometrics(mask)

See Also:
    - core.analysis.preprocessing: Preprocessing utilities
    - core.analysis.postprocessing: Postprocessing utilities
"""

from core.analysis.postprocessing import ImageProcessor as PostprocessingImageProcessor
from core.analysis.preprocessing import ImageProcessor as PreprocessingImageProcessor
from core.analysis.preprocessing import ImageValidator


# Create a unified ImageProcessor class that combines both preprocessing and postprocessing
class ImageProcessor(PreprocessingImageProcessor, PostprocessingImageProcessor):
    """
    Unified ImageProcessor combining preprocessing and postprocessing methods.

    This class provides a single interface for all image processing operations,
    including validation, preprocessing, normalization, segmentation, and
    feature extraction.

    Preprocessing Methods:
        - smart_resize(): Intelligent downsampling
        - load_image_mmap(): Memory-mapped loading
        - generate_tiles(): Tile-based processing
        - normalize_inplace(): In-place normalization
        - convert_colorspace_inplace(): Color space conversion
        - resize_inplace(): Image resizing
        - macenko_normalize(): H&E stain normalization
        - cleanup_mmap(): Memory cleanup
        - process_with_cleanup(): Processing with GC

    Postprocessing Methods:
        - adaptive_watershed(): Cell segmentation
        - calculate_morphometrics(): Feature extraction
        - calculate_entropy(): Uncertainty quantification
    """

    pass


__all__ = [
    "ImageValidator",
    "ImageProcessor",
]
