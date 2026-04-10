"""
errors.py
=========
Custom exception hierarchy for CRAG with context-aware error handling.

Purpose:
  Provides specific exception types for different failure modes to enable
  intelligent error recovery and graceful degradation.

Exception hierarchy:
  CRAGError (base)
  ├── RetrievalError (vector search or embedding failed)
  ├── GraderError (grading/evaluation API call failed)
  ├── GenerationError (answer generation API call failed)
  ├── CorrectionError (query correction failed)
  └── VerificationError (answer verification failed)
"""

from typing import Optional


class CRAGError(Exception):
    """Base exception for all CRAG errors."""

    def __init__(self, message: str, retryable: bool = False, context: Optional[dict] = None):
        self.message = message
        self.retryable = retryable  # Can retry this operation
        self.context = context or {}  # Additional context for debugging
        super().__init__(self.message)

    def __str__(self):
        msg = f"{self.__class__.__name__}: {self.message}"
        if self.context:
            msg += f" (context: {self.context})"
        return msg


class RetrievalError(CRAGError):
    """Raised when document retrieval (vector search, embedding) fails."""

    def __init__(self, message: str, retryable: bool = True, context: Optional[dict] = None):
        super().__init__(message, retryable=retryable, context=context)


class GraderError(CRAGError):
    """Raised when document relevance grading fails."""

    def __init__(self, message: str, retryable: bool = True, context: Optional[dict] = None):
        super().__init__(message, retryable=retryable, context=context)


class GenerationError(CRAGError):
    """Raised when answer generation fails."""

    def __init__(self, message: str, retryable: bool = True, context: Optional[dict] = None):
        super().__init__(message, retryable=retryable, context=context)


class CorrectionError(CRAGError):
    """Raised when query correction fails."""

    def __init__(self, message: str, retryable: bool = True, context: Optional[dict] = None):
        super().__init__(message, retryable=retryable, context=context)


class VerificationError(CRAGError):
    """Raised when answer verification fails."""

    def __init__(self, message: str, retryable: bool = True, context: Optional[dict] = None):
        super().__init__(message, retryable=retryable, context=context)


class APIError(CRAGError):
    """Raised when OpenAI API returns an error."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        retryable: bool = False,
        context: Optional[dict] = None,
    ):
        self.status_code = status_code
        # Determine if error is retryable based on status code
        if status_code:
            # 5xx errors and 429 (rate limit) are retryable
            retryable = status_code >= 500 or status_code == 429
        super().__init__(message, retryable=retryable, context=context or {})
        if status_code:
            self.context["status_code"] = status_code


class RateLimitError(APIError):
    """Raised when OpenAI API rate limit is hit (429)."""

    def __init__(self, message: str, retry_after: Optional[float] = None, context: Optional[dict] = None):
        self.retry_after = retry_after
        ctx = context or {}
        if retry_after:
            ctx["retry_after"] = retry_after
        super().__init__(message, status_code=429, retryable=True, context=ctx)


class TimeoutError(CRAGError):
    """Raised when API call times out."""

    def __init__(self, message: str, timeout_seconds: Optional[float] = None, context: Optional[dict] = None):
        self.timeout_seconds = timeout_seconds
        ctx = context or {}
        if timeout_seconds:
            ctx["timeout_seconds"] = timeout_seconds
        super().__init__(message, retryable=True, context=ctx)
