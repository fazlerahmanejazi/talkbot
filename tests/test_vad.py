import math
import struct

import pytest

from server.vad import StreamingVAD, VADConfig


def pcm16_silence(frame_samples: int) -> bytes:
    return struct.pack("<" + "h" * frame_samples, *([0] * frame_samples))


def pcm16_sine(frame_samples: int, freq_hz: float, sample_rate: int, amp: float = 0.2) -> bytes:
    # amp in [0,1]; 0.2 ~= -14 dBFS
    vals = []
    for n in range(frame_samples):
        t = n / sample_rate
        s = amp * math.sin(2 * math.pi * freq_hz * t)
        vals.append(int(max(-1.0, min(1.0, s)) * 32767))
    return struct.pack("<" + "h" * frame_samples, *vals)


def make_frames(bytes_per_frame: int, blob: bytes):
    for i in range(0, len(blob), bytes_per_frame):
        yield blob[i : i + bytes_per_frame]


def test_expected_frame_size():
    vad = StreamingVAD()
    assert vad.expected_frame_bytes == 640  # 20ms @ 16kHz mono PCM16 = 320 samples * 2


def test_silence_is_not_speech():
    cfg = VADConfig(aggressiveness=2, min_consecutive_speech=2)
    vad = StreamingVAD(cfg)
    samples = int(cfg.sample_rate * cfg.frame_ms / 1000)
    frame = pcm16_silence(samples)

    # Feed several silent frames
    results = [vad.update(frame) for _ in range(10)]
    assert not any(results), "Silence should never be classified as speech"
    assert vad.consecutive_speech == 0
    assert vad.raw_is_speech is False


@pytest.mark.parametrize("freq", [200.0, 400.0, 1000.0])
def test_pure_tone_edge_case(freq):
    """
    Pure tones may sometimes fool VADs depending on aggressiveness.
    We assert that at moderate aggressiveness and small amplitude, it does NOT instantly trip.
    """
    cfg = VADConfig(aggressiveness=2, min_consecutive_speech=2)
    vad = StreamingVAD(cfg)
    samples = int(cfg.sample_rate * cfg.frame_ms / 1000)

    # Create a few consecutive frames of a low-amplitude sine wave
    frames = [pcm16_sine(samples, freq, cfg.sample_rate, amp=0.15) for _ in range(5)]
    results = [vad.update(f) for f in frames]

    # With min_consecutive_speech=2, even if a rare single frame flags, the threshold should protect us.
    assert not any(results), f"Pure tone at {freq}Hz should not reach 'speech' threshold"
    assert vad.consecutive_speech in (0, 1)  # allow a single raw positive at worst


def test_invalid_frame_size_raises():
    vad = StreamingVAD()
    with pytest.raises(ValueError):
        vad.update(b"\x00" * (vad.expected_frame_bytes - 2))
