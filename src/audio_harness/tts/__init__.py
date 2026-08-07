"""Text-to-speech provider adapters.

Importing this package registers every adapter, so :func:`create` can resolve
any key that appears in a benchmark configuration.
"""

from __future__ import annotations

from . import azure, cartesia, deepgram, elevenlabs, gemini, inworld, mistral, openai, openrouter, xai
from .base import TtsProvider, available, create, family_of, register


__all__ = [
    "TtsProvider",
    "available",
    "azure",
    "cartesia",
    "create",
    "deepgram",
    "elevenlabs",
    "family_of",
    "gemini",
    "inworld",
    "mistral",
    "openai",
    "openrouter",
    "register",
    "xai",
]
