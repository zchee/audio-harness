"""Credential and connectivity checks.

Each check is the cheapest request that proves a credential will actually work
— usually listing projects, voices or models. No audio is ever sent, so running
``doctor`` costs essentially nothing and can be repeated freely.

"Authenticates" is not the same as "can spend", and the difference is what
turns up halfway through a long benchmark. Two vendors need more than a read:
xAI answers its identity endpoint for a key with no credit, and Soniox answers
every REST endpoint for an organization with an empty balance. Both are probed
for spending capability as well — xAI with a second request, Soniox by opening
a realtime session and sending no audio.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
import subprocess
import sys
from typing import Any

import httpx
import orjson
from websockets.asyncio.client import connect


TIMEOUT_S = 20.0


@dataclass(slots=True, frozen=True)
class CheckResult:
    """Outcome of one credential check.

    Attributes:
        provider: Human-readable vendor name.
        env_var: Environment variable the check read.
        ok: Whether the credential authenticated successfully.
        detail: Status text, error message, or the reason it was skipped.
        skipped: Whether the credential was absent, so nothing was attempted.
    """

    provider: str
    env_var: str
    ok: bool
    detail: str
    skipped: bool = False


@dataclass(slots=True, frozen=True)
class _HttpCheck:
    """Declarative description of a vendor's cheapest authenticated call.

    Attributes:
        provider: Human-readable vendor name.
        env_var: Environment variable holding the credential.
        url: Endpoint that proves the credential authenticates.
        header: Header the credential is sent in.
        template: Format string wrapping the credential, e.g. ``Bearer {key}``.
        extra_headers: Additional required headers, such as an API version.
        quota_url: Optional second endpoint that consumes quota. When ``url``
            authenticates but this one is refused, the account is out of
            credit rather than misconfigured — two very different fixes.
    """

    provider: str
    env_var: str
    url: str
    header: str
    template: str = "{key}"
    extra_headers: dict[str, str] | None = None
    quota_url: str | None = None


_CHECKS: tuple[_HttpCheck, ...] = (
    _HttpCheck(
        provider="Deepgram",
        env_var="DEEPGRAM_API_KEY",
        url="https://api.deepgram.com/v1/projects",
        header="Authorization",
        template="Token {key}",
    ),
    _HttpCheck(
        provider="ElevenLabs",
        env_var="ELEVENLABS_API_KEY",
        url="https://api.elevenlabs.io/v1/user",
        header="xi-api-key",
    ),
    _HttpCheck(
        provider="AssemblyAI",
        env_var="ASSEMBLYAI_API_KEY",
        url="https://api.assemblyai.com/v2/transcript?limit=1",
        header="Authorization",
    ),
    _HttpCheck(
        provider="Speechmatics",
        env_var="SPEECHMATICS_API_KEY",
        url="https://asr.api.speechmatics.com/v2/jobs?limit=1",
        header="Authorization",
        template="Bearer {key}",
    ),
    _HttpCheck(
        provider="Gladia",
        env_var="GLADIA_API_KEY",
        url="https://api.gladia.io/v2/pre-recorded?limit=1",
        header="x-gladia-key",
    ),
    _HttpCheck(
        provider="Mistral",
        env_var="MISTRAL_API_KEY",
        url="https://api.mistral.ai/v1/models",
        header="Authorization",
        template="Bearer {key}",
    ),
    _HttpCheck(
        provider="OpenAI",
        env_var="OPENAI_API_KEY",
        url="https://api.openai.com/v1/models",
        header="Authorization",
        template="Bearer {key}",
    ),
    # Soniox is checked by opening a realtime session instead; see
    # _check_soniox_session for why its REST endpoints cannot reveal the problem.
    _HttpCheck(
        provider="xAI",
        env_var="XAI_API_KEY",
        url="https://api.x.ai/v1/api-key",
        header="Authorization",
        template="Bearer {key}",
        quota_url="https://api.x.ai/v1/models",
    ),
    _HttpCheck(
        provider="Cartesia",
        env_var="CARTESIA_API_KEY",
        url="https://api.cartesia.ai/voices",
        header="X-API-Key",
        extra_headers={"Cartesia-Version": "2026-03-01"},
    ),
    _HttpCheck(
        provider="Gemini",
        env_var="GEMINI_API_KEY",
        url="https://generativelanguage.googleapis.com/v1beta/models",
        header="x-goog-api-key",
    ),
    _HttpCheck(
        provider="Inworld",
        env_var="INWORLD_API_KEY",
        url="https://api.inworld.ai/voices/v1/voices?pageSize=1",
        header="Authorization",
        template="Basic {key}",
    ),
)


async def _run_http_check(client: httpx.AsyncClient, check: _HttpCheck) -> CheckResult:
    """Issue one vendor's authenticated probe and classify the response."""
    key = os.environ.get(check.env_var)
    if not key:
        return CheckResult(
            provider=check.provider,
            env_var=check.env_var,
            ok=False,
            detail="not set — see docs/ACCOUNTS.md",
            skipped=True,
        )

    headers = {check.header: check.template.format(key=key)}
    if check.extra_headers:
        headers.update(check.extra_headers)

    try:
        response = await client.get(check.url, headers=headers)
    except httpx.HTTPError as exc:
        return CheckResult(
            provider=check.provider,
            env_var=check.env_var,
            ok=False,
            detail=f"unreachable: {type(exc).__name__}: {exc}",
        )

    if response.status_code in {401, 403}:
        return CheckResult(
            provider=check.provider,
            env_var=check.env_var,
            ok=False,
            detail=(f"rejected (HTTP {response.status_code}) — key invalid or scoped out"),
        )
    if response.status_code >= 400:
        return CheckResult(
            provider=check.provider,
            env_var=check.env_var,
            ok=False,
            detail=f"HTTP {response.status_code}: {response.text[:120]}",
        )

    if check.quota_url is not None:
        return await _check_quota(client, check, headers)

    return CheckResult(
        provider=check.provider,
        env_var=check.env_var,
        ok=True,
        detail=f"authenticated (HTTP {response.status_code})",
    )


async def _check_quota(client: httpx.AsyncClient, check: _HttpCheck, headers: dict[str, str]) -> CheckResult:
    """Confirm an authenticated key can also spend, not just identify itself.

    A key that authenticates but has no credit will pass every naive check and
    then fail on the first billable request, halfway through a benchmark.
    """
    assert check.quota_url is not None
    try:
        response = await client.get(check.quota_url, headers=headers)
    except httpx.HTTPError as exc:
        return CheckResult(
            provider=check.provider,
            env_var=check.env_var,
            ok=False,
            detail=f"quota probe unreachable: {type(exc).__name__}: {exc}",
        )

    if response.status_code in {402, 403, 429}:
        return CheckResult(
            provider=check.provider,
            env_var=check.env_var,
            ok=False,
            detail=(
                "key valid but billable requests are refused "
                f"(HTTP {response.status_code}) — out of credit or over the "
                "spending limit; top up before benchmarking"
            ),
        )
    if response.status_code >= 400:
        return CheckResult(
            provider=check.provider,
            env_var=check.env_var,
            ok=False,
            detail=f"quota probe HTTP {response.status_code}: {response.text[:100]}",
        )
    return CheckResult(
        provider=check.provider,
        env_var=check.env_var,
        ok=True,
        detail="authenticated, quota available",
    )


OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"


async def _check_openrouter(client: httpx.AsyncClient) -> CheckResult:
    """Fetch OpenRouter's current-key metadata, a non-billable authenticated call.

    An HTTP 200 alone is not proof the key can run inference: a management
    key authenticates here yet cannot call inference endpoints, and a key
    whose per-key spending limit is exhausted authenticates while every
    audio request would be refused. Both conditions are read from the key
    metadata and rejected explicitly.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return CheckResult(
            provider="OpenRouter",
            env_var="OPENROUTER_API_KEY",
            ok=False,
            detail="not set — see docs/ACCOUNTS.md",
            skipped=True,
        )

    try:
        response = await client.get(OPENROUTER_KEY_URL, headers={"Authorization": f"Bearer {api_key}"})
    except httpx.HTTPError as exc:
        return CheckResult(
            provider="OpenRouter",
            env_var="OPENROUTER_API_KEY",
            ok=False,
            detail=f"unreachable: {type(exc).__name__}: {exc}",
        )

    if response.status_code in {401, 403}:
        return CheckResult(
            provider="OpenRouter",
            env_var="OPENROUTER_API_KEY",
            ok=False,
            detail=f"rejected (HTTP {response.status_code})",
        )
    if response.status_code >= 400:
        return CheckResult(
            provider="OpenRouter",
            env_var="OPENROUTER_API_KEY",
            ok=False,
            detail=f"HTTP {response.status_code}: {response.text[:120]}",
        )

    data = response.json().get("data") or {}
    if data.get("is_management_key"):
        return CheckResult(
            provider="OpenRouter",
            env_var="OPENROUTER_API_KEY",
            ok=False,
            detail="management key — cannot call inference endpoints",
        )
    remaining = data.get("limit_remaining")
    if isinstance(remaining, int | float) and remaining <= 0:
        return CheckResult(
            provider="OpenRouter",
            env_var="OPENROUTER_API_KEY",
            ok=False,
            detail=f"per-key spending limit exhausted (limit_remaining={remaining})",
        )
    detail = f"authenticated (HTTP {response.status_code})"
    if isinstance(remaining, int | float):
        detail += f", ${remaining:.2f} key limit remaining"
    return CheckResult(
        provider="OpenRouter",
        env_var="OPENROUTER_API_KEY",
        ok=True,
        detail=detail,
    )


AZURE_TOKEN_URL = "https://{region}.api.cognitive.microsoft.com/sts/v1.0/issueToken"


async def _check_azure_speech(client: httpx.AsyncClient) -> CheckResult:
    """Issue a Speech Services auth token, the cheapest authenticated call.

    The endpoint is region-scoped (``{region}.api.cognitive.microsoft.com``),
    so unlike every other vendor here this probe cannot be expressed as a
    static :class:`_HttpCheck` entry -- the URL itself depends on a second
    environment variable.
    """
    key = os.environ.get("AZURE_SPEECH_KEY")
    region = os.environ.get("AZURE_SPEECH_REGION")
    if not key or not region:
        return CheckResult(
            provider="Azure Speech",
            env_var="AZURE_SPEECH_KEY" if not key else "AZURE_SPEECH_REGION",
            ok=False,
            detail="not set — see docs/ACCOUNTS.md",
            skipped=True,
        )

    url = AZURE_TOKEN_URL.format(region=region)
    try:
        response = await client.post(url, headers={"Ocp-Apim-Subscription-Key": key})
    except httpx.HTTPError as exc:
        return CheckResult(
            provider="Azure Speech",
            env_var="AZURE_SPEECH_KEY",
            ok=False,
            detail=f"unreachable: {type(exc).__name__}: {exc}",
        )

    if response.status_code in {401, 403}:
        return CheckResult(
            provider="Azure Speech",
            env_var="AZURE_SPEECH_KEY",
            ok=False,
            detail=f"rejected (HTTP {response.status_code}) — key invalid, region wrong, or scoped out",
        )
    if response.status_code >= 400:
        return CheckResult(
            provider="Azure Speech",
            env_var="AZURE_SPEECH_KEY",
            ok=False,
            detail=f"HTTP {response.status_code}: {response.text[:120]}",
        )
    return CheckResult(
        provider="Azure Speech",
        env_var="AZURE_SPEECH_KEY",
        ok=True,
        detail=f"authenticated (HTTP {response.status_code}), region {region}",
    )


SONIOX_STREAM_URL = "wss://stt-rt.soniox.com/transcribe-websocket"


async def _check_soniox_session() -> CheckResult:
    """Open a Soniox realtime session without sending any audio.

    Soniox's REST endpoints answer 200 for a key whose organization has no
    balance — ``/v1/models`` and ``/v1/transcriptions`` both succeed — so an
    HTTP probe reports a healthy credential and the refusal only appears once
    a benchmark is already running. Opening a session surfaces it up front,
    and since no audio is sent there is nothing to bill.
    """
    key = os.environ.get("SONIOX_API_KEY")
    if not key:
        return CheckResult(
            provider="Soniox",
            env_var="SONIOX_API_KEY",
            ok=False,
            detail="not set — see docs/ACCOUNTS.md",
            skipped=True,
        )

    try:
        async with connect(SONIOX_STREAM_URL, open_timeout=15.0) as socket:
            await socket.send(
                orjson.dumps({
                    "api_key": key,
                    "model": "stt-rt-v5",
                    "audio_format": "pcm_s16le",
                    "sample_rate": 16000,
                    "num_channels": 1,
                }).decode()
            )
            try:
                raw = await asyncio.wait_for(socket.recv(), timeout=10.0)
            except TimeoutError:
                # Silence after configuration means the session was accepted.
                return CheckResult(
                    provider="Soniox",
                    env_var="SONIOX_API_KEY",
                    ok=True,
                    detail="realtime session accepted",
                )
            payload = orjson.loads(raw)
    except Exception as exc:
        return CheckResult(
            provider="Soniox",
            env_var="SONIOX_API_KEY",
            ok=False,
            detail=f"session refused: {type(exc).__name__}: {exc}",
        )

    code = payload.get("error_code")
    if code:
        return CheckResult(
            provider="Soniox",
            env_var="SONIOX_API_KEY",
            ok=False,
            detail=f"{code}: {payload.get('error_message', 'session refused')}",
        )
    return CheckResult(
        provider="Soniox",
        env_var="SONIOX_API_KEY",
        ok=True,
        detail="realtime session accepted",
    )


def _check_google_cloud() -> CheckResult:
    """Verify application default credentials and a configured project."""
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        return CheckResult(
            provider="Google Cloud",
            env_var="GOOGLE_CLOUD_PROJECT",
            ok=False,
            detail="not set — see docs/ACCOUNTS.md",
            skipped=True,
        )
    try:
        import google.auth

        credentials, discovered = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    except Exception as exc:
        return CheckResult(
            provider="Google Cloud",
            env_var="GOOGLE_CLOUD_PROJECT",
            ok=False,
            detail=(
                f"no application default credentials ({type(exc).__name__}); run: gcloud auth application-default login"
            ),
        )

    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us")
    principal = getattr(credentials, "service_account_email", None) or "ADC user"
    return CheckResult(
        provider="Google Cloud",
        env_var="GOOGLE_CLOUD_PROJECT",
        ok=True,
        detail=(
            f"{principal} on project {project}"
            f"{'' if discovered == project else f' (ADC default: {discovered})'}"
            f", location {location}"
        ),
    )


def _check_apple_speech() -> CheckResult:
    """Probe on-device Speech availability without triggering any TCC prompt.

    File-based recognition needs no speech-recognition grant (verified
    2026-08-08), so this never calls ``requestAuthorization_`` — it only
    proves the framework imports and the en-US recognizer reports on-device
    support. Bills nothing: the whole lane runs on this machine.
    """
    if sys.platform != "darwin":
        return CheckResult(
            provider="Apple Speech",
            env_var="",
            ok=False,
            detail="requires macOS",
            skipped=True,
        )
    try:
        import Foundation
        import Speech
    except ImportError:
        return CheckResult(
            provider="Apple Speech",
            env_var="",
            ok=False,
            detail="pyobjc-framework-Speech not installed — run: uv sync --extra apple-speech",
            skipped=True,
        )

    # PyObjC populates framework members dynamically, so the type checker
    # cannot see them; the Any views mirror stt/apple_speech.py's guard.
    foundation: Any = Foundation
    speech: Any = Speech
    locale = foundation.NSLocale.localeWithLocaleIdentifier_("en-US")
    recognizer = speech.SFSpeechRecognizer.alloc().initWithLocale_(locale)
    if recognizer is None or not recognizer.isAvailable():
        return CheckResult(
            provider="Apple Speech",
            env_var="",
            ok=False,
            detail="SFSpeechRecognizer is unavailable for en-US",
        )
    if not recognizer.supportsOnDeviceRecognition():
        return CheckResult(
            provider="Apple Speech",
            env_var="",
            ok=False,
            detail="on-device recognition is unsupported for en-US",
        )
    return CheckResult(
        provider="Apple Speech",
        env_var="",
        ok=True,
        detail="on-device recognition available (en-US)",
    )


def _check_parakeet_ane() -> CheckResult:
    """Verify the Swift sidecar binary exists and its CoreML model loads.

    ``selfcheck`` is the cheapest call that proves the whole lane: with a warm
    model cache it answers in well under a second while exercising the binary,
    FluidAudio and the CoreML model together. The first-ever call instead
    downloads ~3 GB, so a timeout reports that condition rather than hanging.
    """
    from audio_harness.stt.parakeet_ane import BUILD_HINT, resolve_binary

    binary = resolve_binary()
    if not binary.is_file():
        return CheckResult(
            provider="Parakeet ANE",
            env_var="PARAKEET_ANE_BINARY",
            ok=False,
            detail=f"sidecar binary not found — build it with: {BUILD_HINT}",
            skipped=True,
        )
    try:
        completed = subprocess.run([str(binary), "selfcheck"], capture_output=True, timeout=120, check=False)
    except subprocess.TimeoutExpired:
        return CheckResult(
            provider="Parakeet ANE",
            env_var="PARAKEET_ANE_BINARY",
            ok=False,
            detail="selfcheck timed out after 120s (the first run downloads ~3 GB; run selfcheck manually once)",
        )
    if completed.returncode != 0:
        detail = (completed.stdout or completed.stderr).decode(errors="replace").strip()[:120]
        return CheckResult(
            provider="Parakeet ANE",
            env_var="PARAKEET_ANE_BINARY",
            ok=False,
            detail=f"selfcheck failed: {detail or '<no output>'}",
        )
    payload = orjson.loads(completed.stdout)
    return CheckResult(
        provider="Parakeet ANE",
        env_var="PARAKEET_ANE_BINARY",
        ok=True,
        detail=(
            f"model {payload.get('model_id', '?')} loads in "
            f"{float(payload.get('load_s', 0.0)):.2f}s on {payload.get('compute_units', '?')}"
        ),
    )


async def run_checks() -> list[CheckResult]:
    """Probe every configured credential concurrently.

    Returns:
        One result per provider, in a stable order for display.
    """
    async with httpx.AsyncClient(timeout=TIMEOUT_S, follow_redirects=True) as client:
        results = await asyncio.gather(
            *(_run_http_check(client, check) for check in _CHECKS),
            _check_openrouter(client),
            _check_azure_speech(client),
        )
    soniox = await _check_soniox_session()
    return [*results, soniox, _check_google_cloud(), _check_apple_speech(), _check_parakeet_ane()]
