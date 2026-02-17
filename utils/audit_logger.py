"""
Audit logging module for compliance requirements.

This module provides audit logging functionality with extended retention
for HIPAA and KVKK compliance. Audit logs are stored separately from
application logs with 90-day retention.

Typical Usage:
    >>> from utils import AuditLogger  # Recommended
    >>>
    >>> # Log events
    >>> AuditLogger.log_image_upload(session_id, filename, file_size)
    >>> AuditLogger.log_inference(session_id, model_name, duration)
    >>> AuditLogger.log_report_generation(session_id, report_id)

Import Paths:
    >>> from utils import AuditLogger  # Recommended
    >>> from utils.audit_logger import AuditLogger  # Also valid
"""

import datetime
import logging
from typing import Optional

# Module-level logger
logger = logging.getLogger(__name__)

# Create module-level audit logger
audit_logger = logging.getLogger("audit")

# Prevent audit logs from propagating to root logger
audit_logger.propagate = False


class AuditLogger:
    """
    Audit logging for compliance requirements.

    Provides static methods to log compliance-relevant events with
    extended retention. All methods wrap logging calls in try/except
    to prevent interrupting application flow.
    """

    @staticmethod
    def log_image_upload(
        session_id: str, filename: str, file_size: int, user_id: Optional[str] = None
    ) -> None:
        """
        Log image upload event.

        Args:
            session_id: Unique session identifier
            filename: Name of uploaded file
            file_size: Size of file in bytes
            user_id: Optional user identifier
        """
        try:
            audit_logger.info(
                "Image uploaded",
                extra={
                    "event_type": "image_upload",
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "session_id": session_id,
                    "uploaded_filename": filename,  # Changed from 'filename' to avoid conflict
                    "file_size": file_size,
                    "user_id": user_id,
                },
            )
        except Exception as e:
            # Prevent audit logging failures from interrupting application
            logging.getLogger(__name__).error(
                f"Audit logging failed for image_upload: {e}", exc_info=True
            )

    @staticmethod
    def log_analysis_start(
        session_id: str, model_version: str, user_id: Optional[str] = None
    ) -> None:
        """
        Log analysis start event.

        Args:
            session_id: Unique session identifier
            model_version: Version of model being used
            user_id: Optional user identifier
        """
        try:
            audit_logger.info(
                "Analysis started",
                extra={
                    "event_type": "analysis_start",
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "session_id": session_id,
                    "model_version": model_version,
                    "user_id": user_id,
                },
            )
        except Exception as e:
            logging.getLogger(__name__).error(
                f"Audit logging failed for analysis_start: {e}", exc_info=True
            )

    @staticmethod
    def log_analysis_complete(
        session_id: str, diagnosis: str, confidence: float, user_id: Optional[str] = None
    ) -> None:
        """
        Log analysis completion event.

        Args:
            session_id: Unique session identifier
            diagnosis: Predicted diagnosis/class
            confidence: Confidence score (0-1)
            user_id: Optional user identifier
        """
        try:
            audit_logger.info(
                "Analysis completed",
                extra={
                    "event_type": "analysis_complete",
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "session_id": session_id,
                    "diagnosis": diagnosis,
                    "confidence": confidence,
                    "user_id": user_id,
                },
            )
        except Exception as e:
            logging.getLogger(__name__).error(
                f"Audit logging failed for analysis_complete: {e}", exc_info=True
            )

    @staticmethod
    def log_report_download(session_id: str, diagnosis: str, user_id: Optional[str] = None) -> None:
        """
        Log report download event.

        Args:
            session_id: Unique session identifier
            diagnosis: Diagnosis included in report
            user_id: Optional user identifier
        """
        try:
            audit_logger.info(
                "Report downloaded",
                extra={
                    "event_type": "report_download",
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "session_id": session_id,
                    "diagnosis": diagnosis,
                    "user_id": user_id,
                },
            )
        except Exception as e:
            logging.getLogger(__name__).error(
                f"Audit logging failed for report_download: {e}", exc_info=True
            )
