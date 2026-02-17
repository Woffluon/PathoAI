"""
Custom exception classes for memory management and error handling.

This module defines the exception hierarchy for memory management operations,
providing descriptive error messages and context for different failure scenarios.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class PathoAIException(Exception):
    """
    Base exception for all PathoAI application errors.

    This is the root exception class for the PathoAI system, providing a common
    parent for all custom exceptions. It allows catching any PathoAI-specific
    error while distinguishing from standard Python exceptions.

    Examples:
        >>> try:
        ...     # PathoAI operation
        ...     process_image(img)
        ... except PathoAIException as e:
        ...     logger.error(f"PathoAI error: {e}")
        ...     # Handle application-specific error
        ... except Exception as e:
        ...     logger.error(f"Unexpected error: {e}")
        ...     # Handle unexpected error

    Notes:
        - All custom PathoAI exceptions should inherit from this class
        - Provides consistent error handling across the application
        - Distinguishes application errors from system/library errors

    See Also:
        - MemoryManagementError: Memory-related errors
        - ImageProcessingError: Image processing failures
        - InferenceError: Model inference failures
    """

    pass


class MemoryManagementError(PathoAIException):
    """
    Base exception for all memory management related errors.

    This is the parent class for all custom exceptions in the memory management
    system. It provides a common interface for catching any memory-related error.
    """

    pass


class ImageValidationError(PathoAIException):
    """
    Raised when image validation fails.

    This exception is raised when an uploaded or provided image fails validation
    checks such as format verification, size limits, or content requirements.

    Common causes include:
    - Unsupported image format
    - Image dimensions too large or too small
    - Corrupted image data
    - Invalid color space or bit depth
    - File size exceeds limits

    Examples:
        >>> from core.image_processing import ImageValidator
        >>> from core.exceptions import ImageValidationError
        >>> try:
        ...     validated_img = ImageValidator.validate_image(img_bytes)
        ... except ImageValidationError as e:
        ...     logger.warning(f"Image validation failed: {e}")
        ...     return {"error": "Invalid image", "details": str(e)}

    Notes:
        - Validation errors are user-facing and should have clear messages
        - Should be logged at WARNING level (user error, not system error)
        - Prevents invalid data from entering the processing pipeline
        - Message should guide user on how to fix the issue

    See Also:
        - ImageValidator: Image validation utilities
        - ImageProcessingError: Processing failures after validation
    """

    pass


class ImageProcessingError(PathoAIException):
    """
    Raised when image processing operations fail.

    This exception is raised when image processing operations such as
    normalization, segmentation, or feature extraction encounter errors
    that prevent successful completion.

    Common causes include:
    - Invalid image data or dimensions
    - Numerical instability in algorithms
    - Insufficient memory for processing
    - Corrupted intermediate results

    Examples:
        >>> from core.image_processing import ImageProcessor
        >>> from core.exceptions import ImageProcessingError
        >>> try:
        ...     normalized = ImageProcessor.macenko_normalize(img)
        ... except ImageProcessingError as e:
        ...     logger.warning(f"Normalization failed: {e}")
        ...     # Fall back to original image
        ...     normalized = img

    Notes:
        - Often includes the original exception as context
        - May be recoverable by falling back to simpler processing
        - Should include descriptive error message for debugging
        - Logged at WARNING or ERROR level depending on severity

    See Also:
        - ImageProcessor: Image processing utilities
        - ImageValidator: Image validation
    """

    pass


class InferenceError(PathoAIException):
    """
    Raised when model inference operations fail.

    This exception is raised when model prediction operations encounter errors
    that prevent successful inference. This includes both classification and
    segmentation failures.

    Common causes include:
    - Model not loaded or unavailable
    - Invalid input dimensions or format
    - TensorFlow/Keras runtime errors
    - Insufficient memory for inference
    - Model architecture incompatibility

    Examples:
        >>> from core.inference import InferenceEngine
        >>> from core.exceptions import InferenceError
        >>> try:
        ...     engine = InferenceEngine()
        ...     class_idx, conf, tensor = engine.predict_classification(img)
        ... except InferenceError as e:
        ...     logger.error(f"Inference failed: {e}")
        ...     return {"error": "Prediction unavailable", "status": "error"}

    Notes:
        - Critical error that prevents prediction results
        - Should be logged with full stack trace for debugging
        - May indicate model corruption or version mismatch
        - Requires investigation and cannot be silently ignored

    See Also:
        - InferenceEngine: Model inference engine
        - ModelLoadError: Model loading failures
    """

    pass


class InsufficientMemoryError(MemoryManagementError):
    """
    Raised when insufficient memory is available for an operation.

    This exception is raised when a memory check determines that there is not
    enough available memory to safely perform a requested operation.

    Attributes:
        required_mb: Amount of memory required for the operation (in MB)
        available_mb: Amount of memory currently available (in MB)
    """

    def __init__(self, required_mb: float, available_mb: float):
        """
        Initialize InsufficientMemoryError.

        Args:
            required_mb: Memory required for the operation (MB)
            available_mb: Memory currently available (MB)
        """
        self.required_mb = required_mb
        self.available_mb = available_mb
        super().__init__(
            f"Insufficient memory: required {required_mb:.2f} MB, "
            f"available {available_mb:.2f} MB. "
            f"Consider reducing image resolution or closing other applications."
        )


class EmergencyCleanupError(MemoryManagementError):
    """
    Raised when emergency cleanup fails or encounters an error.

    This exception indicates that the emergency cleanup mechanism was triggered
    but failed to complete successfully. This is a critical error that may
    require process restart.

    Attributes:
        reason: Description of why the cleanup failed
        memory_before_mb: Memory usage before cleanup attempt (MB)
    """

    def __init__(self, reason: str, memory_before_mb: Optional[float] = None):
        """
        Initialize EmergencyCleanupError.

        Args:
            reason: Description of the failure
            memory_before_mb: Memory usage before cleanup (MB), if available
        """
        self.reason = reason
        self.memory_before_mb = memory_before_mb

        message = f"Emergency cleanup failed: {reason}"
        if memory_before_mb is not None:
            message += f" (memory before cleanup: {memory_before_mb:.2f} MB)"

        super().__init__(message)


class SessionExpiredError(MemoryManagementError):
    """
    Raised when attempting to access an expired session.

    This exception is raised when a user tries to access session data that has
    been cleaned up due to timeout or memory pressure.

    Attributes:
        session_id: ID of the expired session
        reason: Reason for session expiration (timeout, memory pressure, etc.)
    """

    def __init__(self, session_id: str, reason: str = "timeout"):
        """
        Initialize SessionExpiredError.

        Args:
            session_id: ID of the expired session
            reason: Reason for expiration (default: "timeout")
        """
        self.session_id = session_id
        self.reason = reason
        super().__init__(
            f"Session '{session_id}' has expired due to {reason}. "
            f"Please refresh the page to start a new session."
        )


class ModelNotFoundError(MemoryManagementError):
    """
    Raised when a requested model cannot be found.

    This exception is raised when attempting to load a model that doesn't exist
    at the expected path.

    Attributes:
        model_name: Name of the model that was not found
        model_path: Path where the model was expected
    """

    def __init__(self, model_name: str, model_path: Optional[str] = None):
        """
        Initialize ModelNotFoundError.

        Args:
            model_name: Name of the model
            model_path: Expected path of the model file
        """
        self.model_name = model_name
        self.model_path = model_path

        message = f"Model '{model_name}' not found"
        if model_path:
            message += f" at path: {model_path}"

        super().__init__(message)


class ModelLoadError(MemoryManagementError):
    """
    Raised when model loading fails.

    This exception is raised when a model file exists but cannot be loaded,
    typically due to corruption, incompatibility, or insufficient memory.

    Attributes:
        model_name: Name of the model that failed to load
        original_error: The original exception that caused the failure
    """

    def __init__(self, model_name: str, original_error: Optional[Exception] = None):
        """
        Initialize ModelLoadError.

        Args:
            model_name: Name of the model
            original_error: Original exception that caused the failure
        """
        self.model_name = model_name
        self.original_error = original_error

        message = f"Failed to load model '{model_name}'"
        if original_error:
            message += f": {str(original_error)}"

        super().__init__(message)


class ModelIntegrityError(MemoryManagementError):
    """
    Raised when model checksum verification fails.

    This exception is raised when a model file's checksum does not match the
    expected value, indicating potential corruption or tampering.

    Common causes include:
    - Incomplete file download
    - File corruption during storage
    - Model file tampering
    - Version mismatch between model and checksum

    Attributes:
        model_name: Name of the model that failed integrity check
        expected_hash: Expected checksum value
        actual_hash: Actual computed checksum value

    Examples:
        >>> from core.inference import verify_model_integrity
        >>> from core.exceptions import ModelIntegrityError
        >>> try:
        ...     verify_model_integrity(model_path, expected_hash)
        ... except ModelIntegrityError as e:
        ...     logger.error(f"Model integrity check failed: {e}")
        ...     return {"error": "Corrupted model file", "status": "error"}

    Notes:
        - Critical security and reliability check
        - Should be logged at ERROR level with full details
        - Requires re-downloading or replacing the model file
        - Prevents use of potentially corrupted models

    See Also:
        - verify_model_integrity: Model checksum verification
        - ModelLoadError: Model loading failures
    """

    def __init__(
        self,
        model_name: str,
        expected_hash: Optional[str] = None,
        actual_hash: Optional[str] = None,
    ):
        """
        Initialize ModelIntegrityError.

        Args:
            model_name: Name of the model
            expected_hash: Expected checksum value
            actual_hash: Actual computed checksum value
        """
        self.model_name = model_name
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash

        message = f"Model integrity check failed for '{model_name}'"
        if expected_hash and actual_hash:
            message += f": expected {expected_hash}, got {actual_hash}"

        super().__init__(message)


class ConfigurationError(MemoryManagementError):
    """
    Raised when configuration is invalid or missing.

    This exception is raised during startup when critical configuration values
    are invalid or missing.

    Attributes:
        parameter: Name of the invalid configuration parameter
        value: The invalid value
        reason: Description of why the value is invalid
    """

    def __init__(self, parameter: str, value: Any, reason: str):
        """
        Initialize ConfigurationError.

        Args:
            parameter: Name of the configuration parameter
            value: The invalid value
            reason: Why the value is invalid
        """
        self.parameter = parameter
        self.value = value
        self.reason = reason
        super().__init__(f"Invalid configuration for '{parameter}': {value}. {reason}")
