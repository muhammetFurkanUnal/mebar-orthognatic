"""Domain-specific errors raised by :mod:`prompting.llm_handler`."""

from __future__ import annotations

from typing import Optional


class LLMCallError(Exception):
    """Base class for expected failures during one LLM call."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        provider_code: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.provider_code = provider_code


class LLMRequestPreparationError(LLMCallError):
    """The request could not be built from the supplied local inputs."""


class LLMTransportError(LLMCallError):
    """The SDK could not connect to the LLM provider or timed out."""


class LLMProviderError(LLMCallError):
    """The LLM provider rejected the request or returned an HTTP error."""


class LLMAuthenticationError(LLMProviderError):
    """The provider rejected the credentials or access permissions."""


class LLMRateLimitError(LLMProviderError):
    """The provider refused the request because of rate limiting."""


class LLMProviderServerError(LLMProviderError):
    """The provider reported an internal server-side failure."""


class LLMResponseError(LLMCallError):
    """The provider response did not contain the expected message content."""


class LLMInvalidJsonError(LLMCallError):
    """The model response was present but was not valid JSON."""
