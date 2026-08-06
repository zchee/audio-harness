"""Cross-vendor STT/TTS benchmark harness.

Measures recognition accuracy, streaming latency and cost across speech
vendors under identical audio, identical pacing and identical normalization.
"""

from __future__ import annotations

import warnings


# google-genai trips a Python 3.14 typing deprecation the moment it is
# imported, which happens whenever the tts/judge modules load. The message is
# upstream noise; filtering it here — before any submodule import — keeps real
# warnings visible everywhere without sprinkling filters per entry point.
warnings.filterwarnings(  # ruff: ignore[non-empty-init-module] -- must run before any submodule imports google-genai
    "ignore", message="'_UnionGenericAlias' is deprecated", category=DeprecationWarning
)

__version__ = "0.1.0"

__all__ = ["__version__"]
