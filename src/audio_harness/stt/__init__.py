"""Speech-to-text provider adapters.

Importing this package registers every adapter, so :func:`create` can resolve
any key that appears in a benchmark configuration.
"""

from __future__ import annotations

from . import (
    assemblyai,
    deepgram,
    elevenlabs,
    google,
    soniox,
    speechmatics,
    whisper_local,
    xai,
)
from .base import StreamTimeline, SttProvider, available, create, family_of, register

__all__ = [
    "StreamTimeline",
    "SttProvider",
    "assemblyai",
    "available",
    "create",
    "deepgram",
    "elevenlabs",
    "family_of",
    "google",
    "register",
    "soniox",
    "speechmatics",
    "whisper_local",
    "xai",
]
