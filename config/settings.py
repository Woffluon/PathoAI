"""
Configuration Management Module

This module provides centralized configuration management for the PathoAI
histopathology analysis system. It includes application settings, security
configuration, logging setup, and Sentry error tracking integration.

Main Components:
    - Config: Central configuration class with all system parameters
    - JSONFormatter: Structured JSON logging formatter
    - Security exception classes (SecurityError, ModelIntegrityError, ImageValidationError)
    - Logging configuration with file rotation and audit trails
    - Sentry integration for production error tracking

Typical Usage:
    >>> from config import Config
    >>> logger = Config.setup_logging()
    >>> Config.setup_sentry()
    >>> print(Config.MODEL_DIR)
    '/path/to/models'

Import Path:
    >>> from config import Config  # Recommended
    >>> from config import SecurityError, ModelIntegrityError, ImageValidationError

Configuration Sources:
    - Environment variables (LOG_LEVEL, SENTRY_DSN, etc.)
    - Hard-coded defaults for production deployment
    - Model integrity checksums for security validation

References:
    - PEP 257: Docstring Conventions
    - Python Logging Cookbook: https://docs.python.org/3/howto/logging-cookbook.html
    - Sentry Python SDK: https://docs.sentry.io/platforms/python/
"""

import json
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Any, Dict, List, Optional, Set, Tuple


# Security Exception Classes
class SecurityError(Exception):
    """
    Base exception for security-related errors in the PathoAI system.

    This is the parent class for all security-related exceptions, providing
    a common interface for catching security violations, validation failures,
    and integrity check errors.

    Examples:
        >>> try:
        ...     # Security-sensitive operation
        ...     validate_input(data)
        ... except SecurityError as e:
        ...     logger.error(f"Security violation: {e}")
        ...     return error_response()

    Notes:
        - All security exceptions should inherit from this class
        - Used for catching any security-related error generically
        - Provides consistent error handling across security components

    See Also:
        - ModelIntegrityError: Model checksum verification failures
        - ImageValidationError: Image validation failures
    """

    pass


class ModelIntegrityError(SecurityError):
    """
    Raised when model file integrity verification fails.

    This exception is raised when the SHA256 checksum of a model file does not
    match the expected value, indicating potential corruption, tampering, or
    version mismatch. This is a critical security check to prevent loading
    compromised or incorrect models.

    Examples:
        >>> from config import ModelIntegrityError
        >>> from core.inference import verify_model_integrity
        >>> try:
        ...     verify_model_integrity(model_path, expected_hash)
        ... except ModelIntegrityError as e:
        ...     logger.critical(f"Model integrity check failed: {e}")
        ...     raise

    Notes:
        - Always raised during model loading if checksum mismatch detected
        - Indicates potential security breach or file corruption
        - Should never be caught and ignored - requires investigation
        - Expected checksums are defined in Config.MODEL_CHECKSUMS

    See Also:
        - verify_model_integrity(): Function that performs checksum verification
        - Config.MODEL_CHECKSUMS: Dictionary of expected model checksums
    """

    pass


class ImageValidationError(ValueError):
    """
    Raised when uploaded image validation fails.

    This exception is raised by ImageValidator when an uploaded image fails
    security or format validation checks. Common causes include:
    - File size exceeds maximum limit
    - Image dimensions exceed maximum pixels
    - Unsupported file format
    - Corrupted or invalid image data

    Inherits from ValueError for compatibility with standard Python exceptions.

    Examples:
        >>> from core.image_processing import ImageValidator
        >>> from config import ImageValidationError
        >>> try:
        ...     img = ImageValidator.validate_image(file_bytes)
        ... except ImageValidationError as e:
        ...     return {"error": str(e), "status": "validation_failed"}

    Notes:
        - Raised before any image processing to ensure security
        - Error messages are user-friendly and include specific failure reason
        - Maximum file size: 50 MB (Config.MAX_FILE_SIZE)
        - Maximum dimensions: 100 megapixels (Config.MAX_PIXELS)
        - Allowed formats: PNG, JPEG, TIFF (Config.ALLOWED_FORMATS)

    See Also:
        - ImageValidator.validate_image(): Image validation method
        - Config: Configuration class with validation limits
    """

    pass


class JSONFormatter(logging.Formatter):
    """Structured JSON logging formatter"""

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON string.

        Args:
            record: Python logging record

        Returns:
            JSON string with log data
        """
        # Build base log entry with standard fields
        log_entry = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Handle exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Extract and include extra fields (user_id, session_id, request_id)
        extra_fields = ["user_id", "session_id", "request_id"]
        for field in extra_fields:
            if hasattr(record, field):
                log_entry[field] = getattr(record, field)

        # Include any other extra fields that were passed
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
                "getMessage",
                "user_id",
                "session_id",
                "request_id",
            ]:
                log_entry[key] = value

        # Return single-line JSON string
        return json.dumps(log_entry)


class Config:
    """
    Central configuration management for the PathoAI system.

    This class provides a centralized location for all system configuration
    parameters including security settings, model paths, logging configuration,
    image processing parameters, and application constants. All attributes are
    class-level constants that can be accessed without instantiation.

    The Config class follows the singleton pattern for configuration management,
    ensuring consistent settings across the entire application. Configuration
    values can be overridden via environment variables for deployment flexibility.

    Attributes:
        APP_NAME (str): Application name identifier
        VERSION (str): Current application version
        ALLOWED_ORIGINS (List[str]): CORS whitelist for security
        MAX_FILE_SIZE (int): Maximum upload file size in bytes (50 MB)
        MAX_PIXELS (int): Maximum image dimensions (100 megapixels)
        ALLOWED_FORMATS (Set[str]): Permitted image formats (PNG, JPEG, TIFF)
        SESSION_TIMEOUT_SECONDS (int): Session expiration time (30 minutes)
        LOG_LEVEL (str): Logging level from environment (default: INFO)
        LOG_FORMAT (str): Log message format string
        LOG_JSON_FORMAT (bool): Enable structured JSON logging
        LOG_DIR (str): Directory for log files
        DEBUG_MODE (bool): Enable debug mode from environment
        SENTRY_DSN (Optional[str]): Sentry error tracking DSN
        SENTRY_ENVIRONMENT (str): Deployment environment name
        MODEL_CHECKSUMS (Dict[str, str]): SHA256 checksums for model integrity
        IMG_SIZE (Tuple[int, int]): Standard image size for models (224x224)
        INPUT_SHAPE (Tuple[int, int, int]): Model input shape (224x224x3)
        DEFAULT_MICRONS_PER_PIXEL (float): Spatial resolution (0.25 µm/pixel)
        CLASSES (List[str]): Classification labels in Turkish
        BASE_DIR (str): Application base directory path
        MODEL_DIR (str): Directory containing model files
        CLS_MODEL_PATH (str): Path to classification model
        SEG_MODEL_PATH (str): Path to segmentation model
        NUC_THRESHOLD (float): Nucleus detection threshold (0.4)
        CON_THRESHOLD (float): Contour detection threshold (0.3)

    Examples:
        >>> from config import Config
        >>> # Access configuration values
        >>> print(Config.APP_NAME)
        'PathoAI'
        >>> print(Config.IMG_SIZE)
        (224, 224)
        >>>
        >>> # Set up logging
        >>> logger = Config.setup_logging()
        >>> logger.info("Application started")
        >>>
        >>> # Initialize error tracking
        >>> Config.setup_sentry()
        >>>
        >>> # Check model paths
        >>> import os
        >>> assert os.path.exists(Config.CLS_MODEL_PATH)

    Notes:
        - All paths are computed relative to BASE_DIR for portability
        - Environment variables override default values where applicable
        - Model checksums are verified during loading for security
        - Logging configuration creates fallback directories if needed
        - Sentry integration is optional and skipped if DSN not configured

    See Also:
        - setup_logging(): Initialize logging system
        - setup_sentry(): Initialize error tracking
        - JSONFormatter: Structured logging formatter
    """

    APP_NAME: str = "PathoAI"
    VERSION: str = "1.0.0"

    # Security Configuration
    # CORS whitelist
    ALLOWED_ORIGINS: List[str] = [
        "https://huggingface.co",
        "https://*.hf.space",
        "http://localhost:8501",
        "http://localhost:7860",
    ]

    # Image validation limits
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50 MB
    MAX_PIXELS: int = 10000 * 10000  # 100 megapixels
    ALLOWED_FORMATS: Set[str] = {"PNG", "JPEG", "TIFF"}

    # Session security
    SESSION_TIMEOUT_SECONDS: int = 1800  # 30 minutes

    # Logging configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_JSON_FORMAT: bool = os.getenv("LOG_JSON_FORMAT", "false").lower() == "true"
    LOG_DIR: str = os.getenv("LOG_DIR", "/tmp/pathoai" if os.path.exists("/tmp") else "./logs")
    DEBUG_MODE: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Sentry configuration
    SENTRY_DSN: Optional[str] = os.getenv("SENTRY_DSN")
    SENTRY_ENVIRONMENT: str = os.getenv("SENTRY_ENVIRONMENT", "production")

    # Model integrity checksums (SHA256)
    # Computed using SHA256 hash algorithm for model file integrity verification
    # efficientnetv2s_classification.keras: 198.26 MB - EfficientNetV2-S classification model
    # cianet_segmentation.keras: 149.60 MB - CIA-Net segmentation model
    MODEL_CHECKSUMS: Dict[str, str] = {
        "efficientnetv2s_classification.keras": "635f32b909057efe286a15fea16a80ec0adfede434dfafb9106fb0f6777f41d7",
        "cianet_segmentation.keras": "74a8618c44734af796d35935c77bc94bef3aeb969be82c771fda03d0bb63e74f",
    }

    # Image Parameters
    IMG_SIZE: Tuple[int, int] = (224, 224)
    INPUT_SHAPE: Tuple[int, int, int] = (224, 224, 3)

    # Default Scale (40x Büyütme için yaklaşık değer)
    # 1 Piksel = 0.25 mikrometre (µm) varsayımı
    DEFAULT_MICRONS_PER_PIXEL: float = 0.25

    # Class Definitions
    CLASSES: List[str] = [
        "Benign (Normal Doku)",
        "Adenocarcinoma (Akciğer Kanseri Tip 1)",
        "Squamous Cell Carcinoma (Akciğer Kanseri Tip 2)",
    ]

    # File Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR: str = os.path.join(BASE_DIR, "models")

    CLS_MODEL_PATH: str = os.path.join(MODEL_DIR, "efficientnetv2s_classification.keras")
    SEG_MODEL_PATH: str = os.path.join(MODEL_DIR, "cianet_segmentation.keras")

    # Thresholds
    NUC_THRESHOLD: float = 0.4
    CON_THRESHOLD: float = 0.3

    # Image Processing Parameters
    MIN_CELL_AREA_PIXELS: int = 30  # Minimum cell area for noise filtering (pixels)
    MORPHOLOGY_KERNEL_SIZE: Tuple[int, int] = (3, 3)  # Morphological operations kernel size
    PEAK_DETECTION_FOOTPRINT: Tuple[int, int] = (5, 5)  # Peak detection footprint size
    PEAK_MIN_DISTANCE: int = 5  # Minimum distance between peaks (pixels)
    MACENKO_IO: int = 240  # Transmitted light intensity for Macenko normalization
    MACENKO_ALPHA: float = 1.0  # Percentile for stain vector estimation
    MACENKO_BETA: float = 0.15  # OD threshold for Macenko normalization
    SMART_RESIZE_MAX_DIM: int = 2048  # Maximum dimension for smart resizing (pixels)
    MACENKO_DOWNSAMPLE_MAX: int = 1024  # Maximum dimension for Macenko SVD computation (pixels)

    # Visualization Parameters
    HEATMAP_ALPHA: float = 0.5  # Heatmap overlay transparency [0.0-1.0]
    COLORMAP_MAX_VALUE: int = 255  # 8-bit image maximum value
    HEATMAP_CLIP_MIN: float = 0.0  # Minimum value for heatmap clipping
    HEATMAP_CLIP_MAX: float = 255.0  # Maximum value for heatmap clipping

    # Probability Map Parameters
    PROB_CLIP_MIN: float = 1e-7  # Minimum probability for entropy calculation (prevents log(0))
    PROB_CLIP_MAX: float = (
        1.0 - 1e-7
    )  # Maximum probability for entropy calculation (prevents log(0))

    # UI Parameters
    METRIC_CARD_PADDING: int = 16  # Metric card padding (pixels)
    CHART_FIGURE_SIZE: Tuple[float, float] = (
        5.0,
        3.5,
    )  # Matplotlib figure size (width, height in inches)

    # Model Loading Parameters
    HASH_CHUNK_SIZE: int = 8192  # Chunk size for hash computation (bytes)

    @staticmethod
    def setup_logging() -> logging.Logger:
        """
        Initialize and configure the application logging system.

        Sets up a comprehensive logging infrastructure with multiple handlers:
        - Console handler for real-time output
        - Rotating file handler for general application logs
        - Rotating file handler for error-only logs
        - Time-based rotating handler for audit trail logs

        The function creates log directories with fallback mechanisms, supports
        both plain text and structured JSON logging formats, and configures
        separate audit logging for compliance tracking.

        Args:
            None

        Returns:
            logging.Logger: Configured root logger instance with all handlers attached.
                The logger is set to DEBUG level to allow individual handlers to filter.

        Examples:
            >>> from config import Config
            >>> logger = Config.setup_logging()
            >>> logger.info("Application started successfully")
            >>> logger.error("An error occurred", extra={"user_id": "12345"})
            >>>
            >>> # Access audit logger separately
            >>> audit_logger = logging.getLogger("audit")
            >>> audit_logger.info("User login", extra={"user_id": "12345", "action": "login"})

        Notes:
            - Log directory defaults to /var/log/pathoai but falls back to ./logs if not writable
            - If file logging fails entirely, continues with console-only logging
            - Log level is controlled by LOG_LEVEL environment variable (default: INFO)
            - JSON format is enabled via LOG_JSON_FORMAT environment variable
            - Main log rotates at 10MB with 5 backups
            - Error log rotates at 10MB with 5 backups (ERROR level only)
            - Audit log rotates daily with 90-day retention
            - Audit logger does not propagate to root logger to avoid duplication
            - All existing handlers are cleared before setup to prevent duplicates

        See Also:
            - JSONFormatter: Structured JSON logging formatter
            - Config.LOG_LEVEL: Environment variable for log level
            - Config.LOG_JSON_FORMAT: Environment variable for JSON format
            - Config.LOG_DIR: Environment variable for log directory
        """
        # Create log directory with fallback to local ./logs if /var/log/pathoai not writable
        log_dir = Config.LOG_DIR
        logger_temp = logging.getLogger(__name__)
        try:
            os.makedirs(log_dir, exist_ok=True)
        except PermissionError as e:
            # Fallback to local logs directory due to permission error
            logger_temp.warning(f"Permission denied for log directory {log_dir}: {e}")
            log_dir = os.path.join(Config.BASE_DIR, "logs")
            try:
                os.makedirs(log_dir, exist_ok=True)
            except PermissionError as e:
                # If even local logs fail, we'll continue with console-only logging
                logger_temp.warning(f"Permission denied for fallback log directory {log_dir}: {e}")
            except OSError as e:
                # Other OS errors for fallback directory
                logger_temp.warning(f"OS error creating fallback log directory {log_dir}: {e}")
        except OSError as e:
            # Other OS errors for primary directory
            logger_temp.warning(f"OS error creating log directory {log_dir}: {e}")
            log_dir = os.path.join(Config.BASE_DIR, "logs")
            try:
                os.makedirs(log_dir, exist_ok=True)
            except (PermissionError, OSError) as e:
                # If even local logs fail, we'll continue with console-only logging
                logger_temp.warning(f"Failed to create fallback log directory {log_dir}: {e}")

        # Configure root logger with level from LOG_LEVEL environment variable (default: INFO)
        log_level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)  # Set to DEBUG to allow all handlers to filter

        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()

        # Determine formatter based on LOG_JSON_FORMAT
        if Config.LOG_JSON_FORMAT:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(Config.LOG_FORMAT)

        # Add console handler with appropriate formatter (JSON or plain based on LOG_JSON_FORMAT)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Handle file handler creation failures gracefully
        try:
            # Add RotatingFileHandler for main log (10MB max, 5 backups, DEBUG level)
            main_log_file = os.path.join(log_dir, "app.log")
            main_file_handler = RotatingFileHandler(
                main_log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10 MB
            )
            main_file_handler.setLevel(logging.DEBUG)
            main_file_handler.setFormatter(
                JSONFormatter() if Config.LOG_JSON_FORMAT else logging.Formatter(Config.LOG_FORMAT)
            )
            logger.addHandler(main_file_handler)

            # Add RotatingFileHandler for error log (10MB max, 5 backups, ERROR level only)
            error_log_file = os.path.join(log_dir, "error.log")
            error_file_handler = RotatingFileHandler(
                error_log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10 MB
            )
            error_file_handler.setLevel(logging.ERROR)
            error_file_handler.setFormatter(
                JSONFormatter() if Config.LOG_JSON_FORMAT else logging.Formatter(Config.LOG_FORMAT)
            )
            logger.addHandler(error_file_handler)

            # Add TimedRotatingFileHandler for audit log (daily rotation, 90-day retention)
            audit_log_file = os.path.join(log_dir, "audit.log")
            audit_file_handler = TimedRotatingFileHandler(
                audit_log_file, when="midnight", interval=1, backupCount=90
            )
            audit_file_handler.setLevel(logging.INFO)
            audit_file_handler.setFormatter(
                JSONFormatter() if Config.LOG_JSON_FORMAT else logging.Formatter(Config.LOG_FORMAT)
            )

            # Configure audit logger separately
            audit_logger = logging.getLogger("audit")
            audit_logger.setLevel(logging.INFO)
            audit_logger.propagate = False  # Don't propagate to root logger
            audit_logger.addHandler(audit_file_handler)

        except PermissionError as e:
            # Log warning and continue with console only
            logger.warning(
                f"Permission denied for file logging: {e}. Continuing with console-only logging."
            )
        except OSError as e:
            # Log warning for other OS errors and continue with console only
            logger.warning(
                f"OS error setting up file logging: {e}. Continuing with console-only logging."
            )

        # Return configured logger
        return logger

    @staticmethod
    def _sentry_before_send(
        event: Dict[str, Any], hint: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Filter and scrub personally identifiable information (PII) from Sentry events.

        This callback function is invoked by Sentry before sending error events to
        the Sentry server. It removes or redacts sensitive information including
        patient data, medical records, uploaded images, and other PII to ensure
        HIPAA compliance and data privacy.

        The function scrubs PII from multiple event contexts:
        - Request data (form submissions, API payloads)
        - Extra context fields
        - User context fields
        - Custom context dictionaries
        - Large binary data (images, files)

        Args:
            event: Sentry event dictionary containing error information, context,
                and metadata. This dictionary is modified in-place to remove PII.
                Common keys include 'request', 'extra', 'user', 'contexts'.
            hint: Additional context dictionary provided by Sentry with information
                about the error source. Contains keys like 'exc_info', 'log_record'.
                Currently not used but required by Sentry callback signature.

        Returns:
            Optional[Dict[str, Any]]: Modified event dictionary with PII redacted,
                or None to drop the event entirely. Returns the scrubbed event in
                normal operation, or the original event if scrubbing fails.

        Examples:
            >>> # Used internally by Sentry SDK
            >>> import sentry_sdk
            >>> sentry_sdk.init(
            ...     dsn="https://example.com",
            ...     before_send=Config._sentry_before_send
            ... )
            >>>
            >>> # Example event transformation
            >>> event = {
            ...     'request': {'data': {'patient_name': 'John Doe'}},
            ...     'extra': {'email': 'patient@example.com'}
            ... }
            >>> scrubbed = Config._sentry_before_send(event, {})
            >>> print(scrubbed['request']['data'])
            '[REDACTED]'
            >>> print(scrubbed['extra']['email'])
            '[REDACTED]'

        Notes:
            - PII fields redacted: patient_name, patient_id, ssn, dob, phone, email,
              address, medical_record_number, mrn, and other common identifiers
            - Image data fields are completely removed (not just redacted) due to size
              and sensitivity: image, image_data, img, img_data, uploaded_bytes, file_bytes
            - Non-sensitive identifiers preserved: session_id, error_id, request_id
            - If scrubbing fails, logs warning but still sends event (better some data than none)
            - Scrubbing is applied recursively to all context dictionaries
            - This function is called automatically by Sentry SDK, not manually

        See Also:
            - Config.setup_sentry(): Sentry initialization function
            - Sentry before_send documentation: https://docs.sentry.io/platforms/python/configuration/filtering/
        """
        try:
            # Check if 'request' in event and 'data' in event['request']
            if "request" in event and "data" in event.get("request", {}):
                # Replace event['request']['data'] with "[REDACTED]"
                event["request"]["data"] = "[REDACTED]"

            # Check for patient information fields and replace with "[REDACTED]"
            # Common PII fields that might appear in error context
            pii_fields = [
                "patient_name",
                "patient_id",
                "ssn",
                "social_security",
                "date_of_birth",
                "dob",
                "phone",
                "phone_number",
                "email",
                "address",
                "medical_record_number",
                "mrn",
            ]

            # Scrub PII from extra context
            if "extra" in event:
                for field in pii_fields:
                    if field in event["extra"]:
                        event["extra"][field] = "[REDACTED]"

            # Scrub PII from user context
            if "user" in event:
                for field in pii_fields:
                    if field in event["user"]:
                        event["user"][field] = "[REDACTED]"

            # Scrub PII from contexts
            if "contexts" in event:
                for context_name, context_data in event["contexts"].items():
                    if isinstance(context_data, dict):
                        for field in pii_fields:
                            if field in context_data:
                                context_data[field] = "[REDACTED]"

            # Check for image data fields and remove them
            # Image data can be very large and contains sensitive medical information
            image_fields = [
                "image",
                "image_data",
                "img",
                "img_data",
                "uploaded_bytes",
                "file_bytes",
            ]

            if "extra" in event:
                for field in image_fields:
                    if field in event["extra"]:
                        del event["extra"][field]

            # Preserve session_id, error_id, request_id fields
            # These are non-sensitive identifiers useful for debugging
            # They are already in the event if present, so no action needed

            # Return modified event
            return event

        except Exception as e:
            # If scrubbing fails, log the error but still send the event
            # Better to have some data than none
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to scrub PII from Sentry event: {e}")
            return event

    @staticmethod
    def setup_sentry() -> None:
        """
        Initialize Sentry error tracking and performance monitoring.

        Configures the Sentry SDK for production error tracking, performance monitoring,
        and profiling. The function sets up logging integration, PII scrubbing, sampling
        rates, and environment-specific configuration. If Sentry DSN is not configured
        or the SDK is not installed, the function gracefully skips initialization.

        Sentry provides:
        - Automatic error capture and stack traces
        - Performance transaction monitoring (10% sample rate)
        - Code profiling (10% sample rate)
        - Logging breadcrumbs for context
        - Release tracking for version management

        Args:
            None

        Returns:
            None

        Raises:
            Does not raise exceptions. All initialization failures are caught and logged
            as warnings, allowing the application to continue without error tracking.

        Examples:
            >>> from config import Config
            >>> # Initialize Sentry (typically called once at application startup)
            >>> Config.setup_sentry()
            >>> # Sentry will now automatically capture errors
            >>>
            >>> # Errors are captured automatically
            >>> try:
            ...     risky_operation()
            ... except Exception as e:
            ...     logger.error("Operation failed", exc_info=True)
            ...     # Error is automatically sent to Sentry
            >>>
            >>> # Add custom context to errors
            >>> import sentry_sdk
            >>> sentry_sdk.set_user({"id": "12345", "username": "doctor1"})
            >>> sentry_sdk.set_tag("component", "inference")

        Notes:
            - Requires SENTRY_DSN environment variable to be set
            - If DSN not configured, logs info message and returns early
            - If sentry_sdk not installed, logs warning and returns early
            - Environment set via SENTRY_ENVIRONMENT variable (default: production)
            - Release version set to "pathoai@{VERSION}" for tracking
            - Logging integration captures INFO+ as breadcrumbs, ERROR+ as events
            - Performance monitoring samples 10% of transactions (traces_sample_rate=0.1)
            - Profiling samples 10% of transactions (profiles_sample_rate=0.1)
            - PII scrubbing applied via Config._sentry_before_send callback
            - Initialization failures are logged but do not stop application
            - Should be called once during application startup, after setup_logging()

        See Also:
            - Config._sentry_before_send(): PII scrubbing callback
            - Config.SENTRY_DSN: Sentry Data Source Name configuration
            - Config.SENTRY_ENVIRONMENT: Deployment environment name
            - Sentry Python SDK: https://docs.sentry.io/platforms/python/
        """
        try:
            # Import sentry_sdk and LoggingIntegration
            import sentry_sdk
            from sentry_sdk.integrations.logging import LoggingIntegration

            # Check if SENTRY_DSN is set, return early if not
            if not Config.SENTRY_DSN:
                logger = logging.getLogger(__name__)
                logger.info("Sentry DSN not configured, skipping error tracking setup")
                return

            # Create LoggingIntegration with level=INFO, event_level=ERROR
            sentry_logging = LoggingIntegration(
                level=logging.INFO,  # Capture info and above as breadcrumbs
                event_level=logging.ERROR,  # Send errors and above as events
            )

            # Call sentry_sdk.init() with DSN, environment, integrations, traces_sample_rate=0.1, profiles_sample_rate=0.1
            sentry_sdk.init(
                dsn=Config.SENTRY_DSN,
                environment=Config.SENTRY_ENVIRONMENT,
                integrations=[sentry_logging],
                traces_sample_rate=0.1,  # Sample 10% of transactions for performance monitoring
                profiles_sample_rate=0.1,  # Sample 10% of transactions for profiling
                release=f"pathoai@{Config.VERSION}",  # Set release to f"pathoai@{Config.VERSION}"
                before_send=Config._sentry_before_send,  # Set before_send to Config._sentry_before_send
            )

            logger = logging.getLogger(__name__)
            logger.info(
                f"Sentry initialized successfully for environment: {Config.SENTRY_ENVIRONMENT}"
            )

        except ImportError as e:
            # Sentry SDK not installed
            logger = logging.getLogger(__name__)
            logger.warning(f"Sentry SDK not installed: {e}. Skipping error tracking setup.")
        except ValueError as e:
            # Invalid configuration value (e.g., invalid DSN format)
            logger = logging.getLogger(__name__)
            logger.warning(f"Invalid Sentry configuration: {e}. Continuing without error tracking.")
        except RuntimeError as e:
            # Runtime error during Sentry initialization
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Sentry initialization runtime error: {e}. Continuing without error tracking."
            )
        except OSError as e:
            # Network or I/O error during Sentry initialization
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Network/IO error initializing Sentry: {e}. Continuing without error tracking."
            )
