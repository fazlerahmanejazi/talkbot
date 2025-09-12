import asyncio
import time
from typing import Any, Dict, List

import pytest

from server.pipeline import VoicePipeline, PipelineConfig
from server.vad import StreamingVAD


class FakeSession:
    def __init__(self):
        self.sent: List[Dict[str, Any]] = []

    async def send_json(self, obj: Dict[str, Any]):
        # Record monotonic time for precision
        self.sent.append({"t": time.monotonic(), **obj})


class HotVAD(StreamingVAD):
    """
    A VAD that returns speech=True after N frames to simulate barge-in.
    """
    def __init__(self, trip_after: int = 2):
        super().__init__()
        self.trip_after = trip_after
        self.count = 0

    def update(self, frame_bytes: bytes) -> bool:
        # Count frames; after trip_after, report speech
        self.count += 1
        # Set raw_is_speech True from the moment we start "speaking"
        self.raw_is_speech = self.count >= self.trip_after
        return self.raw_is_speech


@pytest.mark.asyncio
async def test_bargein_cut_under_80ms(monkeypatch):
    session = FakeSession()
    pipe = VoicePipeline(session, cfg=PipelineConfig())

    # Monkeypatch the VAD to our hot VAD that triggers on 2nd frame
    hot = HotVAD(trip_after=2)
    pipe.vad = hot

    # Pretend we are already speaking (agent TTS in progress)
    pipe.state = "SPEAK"

    # Build a 20ms dummy frame (size isn't validated here)
    frame = b"\x00" * 640

    # Mark "speech start" time just before feeding the first frames
    t0 = time.monotonic()

    # Feed two frames: first = no speech, second = triggers barge-in
    await pipe.handle_audio(frame)
    await pipe.handle_audio(frame)

    # Allow event loop to run the cut handler
    await asyncio.sleep(0.01)  # 10ms

    # Find the cut_playback message
    cuts = [m for m in session.sent if m.get("type") == "cut_playback"]
    assert cuts, "Pipeline did not send cut_playback"

    t_cut = cuts[0]["t"]
    delta_ms = (t_cut - t0) * 1000.0
    assert delta_ms < 80.0, f"Barge-in cut too slow: {delta_ms:.2f} ms"
