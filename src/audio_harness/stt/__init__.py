"""Speech-to-text provider adapters.

Importing this package registers every adapter, so :func:`create` can resolve
any key that appears in a benchmark configuration.
"""

from __future__ import annotations

from . import (
    apple_speech,
    assemblyai,
    azure,
    cartesia,
    deepgram,
    elevenlabs,
    gladia,
    google,
    openai,
    openrouter,
    parakeet_ane,
    soniox,
    speechmatics,
    voxtral,
    whisper_local,
    xai,
)
from .base import StreamTimeline, SttProvider, available, create, family_of, register


__all__ = [
    "StreamTimeline",
    "SttProvider",
    "apple_speech",
    "assemblyai",
    "available",
    "azure",
    "cartesia",
    "create",
    "deepgram",
    "elevenlabs",
    "family_of",
    "gladia",
    "google",
    "openai",
    "openrouter",
    "parakeet_ane",
    "register",
    "soniox",
    "speechmatics",
    "voxtral",
    "whisper_local",
    "xai",
]
