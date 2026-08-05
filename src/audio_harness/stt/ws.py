"""Generic WebSocket streaming driver for speech-to-text adapters.

Every vendor's realtime protocol differs only in four places: the message that
opens a session, how audio bytes are framed, the message that ends input, and
how transcript events are shaped. This driver owns everything else — pacing,
timing, concurrent send/receive and bounded shutdown — so each adapter reduces
to those four callbacks.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from typing import Any

import orjson
import websockets
from websockets.asyncio.client import ClientConnection, connect

from ..audio import pace_chunks
from ..types import AudioClip
from .base import StreamTimeline

OnOpen = Callable[[ClientConnection], Awaitable[None]]
"""Sends the session-configuration message, if the vendor needs one."""

EncodeChunk = Callable[[bytes], str | bytes]
"""Frames one PCM chunk as the payload the vendor expects."""

OnInputDone = Callable[[ClientConnection], Awaitable[None]]
"""Signals end of input, e.g. Deepgram's CloseStream or a JSON terminator."""

HandleMessage = Callable[[Any, StreamTimeline], bool]
"""Records transcript events; returns ``True`` when the session is finished."""

DEFAULT_FINALIZE_TIMEOUT_S = 15.0
"""Grace period for a final transcript after input ends."""

_CONNECT_SLACK_S = 30.0
"""Extra budget covering handshake and non-realtime upload time."""

CLOSE_TIMEOUT_S = 5.0
"""How long to wait for the closing handshake before abandoning the socket."""


class StreamProtocolError(RuntimeError):
    """Raised when a vendor reports an error over the WebSocket."""


async def run_stream(
    *,
    url: str,
    headers: dict[str, str],
    clip: AudioClip,
    chunk_ms: int,
    realtime: bool,
    timeline: StreamTimeline,
    handle_message: HandleMessage,
    on_open: OnOpen | None = None,
    encode_chunk: EncodeChunk | None = None,
    on_input_done: OnInputDone | None = None,
    finalize_timeout_s: float = DEFAULT_FINALIZE_TIMEOUT_S,
) -> None:
    """Stream a clip over a WebSocket and record transcript events.

    Audio is written by a background task while the calling task drains the
    socket, so a provider that emits interim results mid-stream is timed as it
    responds rather than only after the upload finishes.

    Args:
        url: Fully-qualified ``wss://`` endpoint including query parameters.
        headers: Handshake headers, typically carrying the API key.
        clip: Audio to stream.
        chunk_ms: Chunk size written to the socket, in milliseconds.
        realtime: Whether to pace chunks at playback speed.
        timeline: Recorder that receives every transcript event.
        handle_message: Parses one decoded message; returns ``True`` to stop.
        on_open: Sends the session-configuration message, if any.
        encode_chunk: Frames a PCM chunk; defaults to sending raw binary.
        on_input_done: Signals end of input, if the vendor requires it.
        finalize_timeout_s: Grace period for the final transcript after input
            ends. Exceeding it ends the session rather than raising, because a
            provider that never finalizes has still been measured.

    Raises:
        StreamProtocolError: If the vendor reports an error message.
    """
    encode = encode_chunk or (lambda chunk: chunk)

    async with connect(
        url,
        additional_headers=headers,
        max_size=None,
        open_timeout=30.0,
        ping_interval=None,
    ) as socket:
        if on_open is not None:
            await on_open(socket)

        timeline.start()
        sender = asyncio.create_task(
            _send_audio(
                socket=socket,
                clip=clip,
                chunk_ms=chunk_ms,
                realtime=realtime,
                encode=encode,
                timeline=timeline,
                on_input_done=on_input_done,
            )
        )
        try:
            await _receive(
                socket=socket,
                clip=clip,
                realtime=realtime,
                timeline=timeline,
                handle_message=handle_message,
                finalize_timeout_s=finalize_timeout_s,
            )
        finally:
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await sender
            # Complete the closing handshake before returning. Vendors count a
            # session as open until they see it, so returning early makes the
            # next clip in the same lane collide with a session that is still
            # winding down and get refused for excess concurrency.
            with contextlib.suppress(Exception):
                await asyncio.wait_for(socket.close(), timeout=CLOSE_TIMEOUT_S)


async def _send_audio(
    *,
    socket: ClientConnection,
    clip: AudioClip,
    chunk_ms: int,
    realtime: bool,
    encode: EncodeChunk,
    timeline: StreamTimeline,
    on_input_done: OnInputDone | None,
) -> None:
    """Write paced audio, then mark input complete and signal end of stream.

    ``audio_complete`` is stamped before the terminator is sent so that
    finalization latency measures the provider's tail, not our own write.
    """
    try:
        async for chunk in pace_chunks(clip, chunk_ms, realtime=realtime):
            await socket.send(encode(chunk))
    except websockets.ConnectionClosed:
        timeline.audio_complete()
        return

    timeline.audio_complete()
    if on_input_done is not None:
        with contextlib.suppress(websockets.ConnectionClosed):
            await on_input_done(socket)


async def _receive(
    *,
    socket: ClientConnection,
    clip: AudioClip,
    realtime: bool,
    timeline: StreamTimeline,
    handle_message: HandleMessage,
    finalize_timeout_s: float,
) -> None:
    """Drain the socket until the handler signals completion or time runs out."""
    while True:
        deadline = _deadline(
            timeline=timeline,
            clip=clip,
            realtime=realtime,
            finalize_timeout_s=finalize_timeout_s,
        )
        remaining = deadline - timeline.elapsed()
        if remaining <= 0:
            return
        try:
            raw = await asyncio.wait_for(socket.recv(), timeout=remaining)
        except TimeoutError, websockets.ConnectionClosed:
            return

        payload = _decode(raw)
        if payload is None:
            continue
        if handle_message(payload, timeline):
            return


def _deadline(
    *,
    timeline: StreamTimeline,
    clip: AudioClip,
    realtime: bool,
    finalize_timeout_s: float,
) -> float:
    """Return the elapsed-time budget for the receive loop.

    While audio is still being written the budget covers the whole clip; once
    input has finished it tightens to the finalization grace period, so a
    provider that goes silent is not waited on for the full clip duration.
    """
    if timeline.audio_end_s is not None:
        return timeline.audio_end_s + finalize_timeout_s
    playback = clip.duration_s if realtime else 0.0
    return playback + finalize_timeout_s + _CONNECT_SLACK_S


def _decode(raw: str | bytes) -> Any | None:
    """Decode one frame as JSON, or return ``None`` if it is not JSON."""
    try:
        return orjson.loads(raw)
    except orjson.JSONDecodeError:
        return None
