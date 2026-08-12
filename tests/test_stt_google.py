"""Endpoint-selection tests for the Google Cloud STT v2 (Chirp 3) adapter.

Chirp 3 is served from regional endpoints, so the pre-existing ``location``
option must swap the client's ``api_endpoint`` to the v2 regional host
pattern (``{location}-speech.googleapis.com``). The SDK client is replaced
with a recording fake — no credentials, no network.
"""

from __future__ import annotations

from dataclasses import dataclass

from google.api_core.client_options import ClientOptions
import pytest

from audio_harness import stt
from audio_harness.stt import google


@dataclass
class FakeSpeechClient:
    """Records the client options the adapter constructs the SDK client with."""

    client_options: ClientOptions | None = None


def make_adapter(options: dict[str, object] | None = None) -> google.GoogleChirp3:
    """Instantiate the registered adapter with its concrete type asserted."""
    adapter = stt.create("google-chirp3", options)
    assert isinstance(adapter, google.GoogleChirp3)
    return adapter


def endpoint_of(adapter: google.GoogleChirp3) -> str | None:
    """Return the ``api_endpoint`` the adapter built its client with."""
    client = adapter._speech_client()
    assert isinstance(client, FakeSpeechClient)
    if client.client_options is None:
        return None
    return client.client_options.api_endpoint


@pytest.fixture
def patched_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap the SDK client for the fake and clear ambient location config."""
    monkeypatch.setattr(google, "SpeechAsyncClient", FakeSpeechClient)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")


class TestRegionalEndpoint:
    """The ``location`` option drives the regional ``api_endpoint``."""

    def test_default_location_is_us_regional_endpoint(self, patched_client: None) -> None:
        adapter = make_adapter()

        assert endpoint_of(adapter) == "us-speech.googleapis.com"
        assert adapter._recognizer() == "projects/test-project/locations/us/recognizers/_"

    def test_location_option_switches_the_endpoint_host(self, patched_client: None) -> None:
        adapter = make_adapter({"location": "eu"})

        assert endpoint_of(adapter) == "eu-speech.googleapis.com"
        assert adapter._recognizer() == "projects/test-project/locations/eu/recognizers/_"

    def test_global_location_uses_the_default_endpoint(self, patched_client: None) -> None:
        adapter = make_adapter({"location": "global"})

        assert endpoint_of(adapter) is None, "global must not override the SDK's default endpoint"
        assert adapter._recognizer() == "projects/test-project/locations/global/recognizers/_"

    def test_env_var_sets_location_and_option_wins_over_it(
        self,
        patched_client: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "asia-southeast1")

        assert endpoint_of(make_adapter()) == "asia-southeast1-speech.googleapis.com"
        assert endpoint_of(make_adapter({"location": "eu"})) == "eu-speech.googleapis.com"

    def test_client_is_constructed_once_and_cached(self, patched_client: None) -> None:
        adapter = make_adapter({"location": "eu"})

        assert adapter._speech_client() is adapter._speech_client()
