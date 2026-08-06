"""Text-to-speech provider adapters.

Importing this package registers every adapter, so :func:`create` can resolve
any key that appears in a benchmark configuration.
"""

from __future__ import annotations

from . import cartesia, deepgram, elevenlabs, gemini, inworld
from .base import TtsProvider, available, create, family_of, register


__all__ = [
    "TtsProvider",
    "available",
    "cartesia",
    "create",
    "deepgram",
    "elevenlabs",
    "family_of",
    "gemini",
    "inworld",
    "register",
]
