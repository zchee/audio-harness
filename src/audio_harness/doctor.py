"""Credential and connectivity checks.

Each check is the cheapest authenticated request the vendor exposes — listing
projects, voices or models. No audio is sent, so running ``doctor`` costs
essentially nothing and can be repeated freely before a long benchmark.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import httpx

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
        provider="Soniox",
        env_var="SONIOX_API_KEY",
        url="https://api.soniox.com/v1/models",
        header="Authorization",
        template="Bearer {key}",
    ),
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
            detail=(
                f"rejected (HTTP {response.status_code}) — key invalid or scoped out"
            ),
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


async def _check_quota(
    client: httpx.AsyncClient, check: _HttpCheck, headers: dict[str, str]
) -> CheckResult:
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

        credentials, discovered = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
    except Exception as exc:
        return CheckResult(
            provider="Google Cloud",
            env_var="GOOGLE_CLOUD_PROJECT",
            ok=False,
            detail=(
                f"no application default credentials ({type(exc).__name__}); "
                f"run: gcloud auth application-default login"
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


async def run_checks() -> list[CheckResult]:
    """Probe every configured credential concurrently.

    Returns:
        One result per provider, in a stable order for display.
    """
    async with httpx.AsyncClient(timeout=TIMEOUT_S, follow_redirects=True) as client:
        http_results = await asyncio.gather(
            *(_run_http_check(client, check) for check in _CHECKS)
        )
    return [*http_results, _check_google_cloud()]
