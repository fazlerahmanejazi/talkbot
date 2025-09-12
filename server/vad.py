from __future__ import annotations

import struct
from dataclasses import dataclass

import webrtcvad


@dataclass
class VADConfig:
    sample_rate: int = 16000         # Hz
    frame_ms: int = 20               # ms
    aggressiveness: int = 2          # 0..3 (3 is most aggressive)
    min_consecutive_speech: int = 2  # consecutive frames to treat as "speech"


class StreamingVAD:
    """
    Thin, streaming wrapper over WebRTC VAD for fixed-size frames.

    Usage:
        vad = StreamingVAD()
        is_speaking = vad.update(frame_bytes)  # frame_bytes = 20ms PCM16@16k mono

    Notes:
      - Returns True only when current frame is speech AND we have hit the
        configured min_consecutive_speech threshold.
      - Provides `raw_is_speech` (per-frame VAD) and `consecutive_speech` counters.
    """

    def __init__(self, cfg: VADConfig | None = None):
        self.cfg = cfg or VADConfig()
        if self.cfg.sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError("WebRTC VAD supports 8000/16000/32000/48000 Hz only")
        if self.cfg.frame_ms not in (10, 20, 30):
            raise ValueError("WebRTC VAD supports frame sizes of 10, 20, or 30 ms")

        self._vad = webrtcvad.Vad(self.cfg.aggressiveness)
        self._frame_bytes = int(self.cfg.sample_rate * self.cfg.frame_ms / 1000) * 2  # PCM16 mono
        self.consecutive_speech = 0
        self.raw_is_speech = False

    @property
    def expected_frame_bytes(self) -> int:
        return self._frame_bytes

    def reset(self):
        self.consecutive_speech = 0
        self.raw_is_speech = False

    def _validate(self, frame_bytes: bytes):
        if not isinstance(frame_bytes, (bytes, bytearray)):
            raise TypeError("frame_bytes must be bytes-like")
        if len(frame_bytes) != self._frame_bytes:
            raise ValueError(
                f"Expected {self._frame_bytes} bytes per frame, got {len(frame_bytes)}"
            )
        # Quick sanity: ensure it looks like 16-bit aligned data
        if len(frame_bytes) % 2 != 0:
            raise ValueError("PCM16 frames must have even length")

    def update(self, frame_bytes: bytes) -> bool:
        """
        Feed one 20ms frame; returns True when min_consecutive_speech reached.
        """
        self._validate(frame_bytes)

        # WebRTC expects little-endian PCM16 mono
        try:
            self.raw_is_speech = self._vad.is_speech(
                frame_bytes, sample_rate=self.cfg.sample_rate
            )
        except Exception as e:
            # If VAD throws (rare), treat as non-speech for robustness
            self.raw_is_speech = False

        # Tone guard: suppress false positives on pure tones
        if self.raw_is_speech and self._looks_like_pure_tone(frame_bytes):
            self.raw_is_speech = False

        if self.raw_is_speech:
            self.consecutive_speech += 1
        else:
            self.consecutive_speech = 0

        return self.consecutive_speech >= self.cfg.min_consecutive_speech

    def _looks_like_pure_tone(self, frame_bytes: bytes) -> bool:
        import numpy as np
        x = np.frombuffer(frame_bytes, dtype="<i2").astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(x * x)) + 1e-9)
        if rms > 0.35:  # loud, don’t suppress
            return False
        x = x - np.mean(x)
        mag = np.abs(np.fft.rfft(x))
        if mag.size > 1:
            mag[0] = 0.0
        total = float(np.sum(mag) + 1e-9)
        peak = float(np.max(mag))
        peak_ratio = peak / total
        return peak_ratio > 0.90 and rms < 0.25


    # Convenience: quick energy gate (optional) to avoid obvious false positives on DC/zeros
    @staticmethod
    def rms(frame_bytes: bytes) -> float:
        """Root-mean-square as a tiny helper (not used in decision by default)."""
        samples = struct.unpack("<" + "h" * (len(frame_bytes) // 2), frame_bytes)
        acc = 0.0
        for s in samples:
            acc += (s / 32768.0) ** 2
        return (acc / len(samples)) ** 0.5
