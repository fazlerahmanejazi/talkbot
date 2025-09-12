from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Literal, Optional

import numpy as np

from .settings import settings

# Backend imports are optional; guarded so you can run without installing both.
try:
    from faster_whisper import WhisperModel  # type: ignore
except Exception:
    WhisperModel = None  # type: ignore


def _check_faster_whisper_model(model_name: str) -> bool:
    """
    Check if Faster-Whisper model is available without downloading.
    Returns True if model is cached/available, False otherwise.
    """
    if WhisperModel is None:
        return False
    
    try:
        # Try to initialize the model - this will use cached version if available
        # but won't download if not cached (we'll catch that exception)
        model = WhisperModel(model_name, compute_type="int8")
        return True
    except Exception:
        # Model not available/cached
        return False


ASREventType = Literal["partial", "final", "streaming"]


@dataclass
class ASRConfig:
    sample_rate: int = 16000            # Hz (expected input)
    frame_ms: int = 20                  # ms (input frame size)
    # Simplified: only decode on flush, no partials
    max_window_ms: int = 4000           # cap context to limit decode cost
    # Backend
    faster_model: str = settings.faster_whisper_model   # e.g. "base.en"
    # Misc
    language: Optional[str] = "en"
    beam_size: int = 1                   # keep small for latency
    
    # Partial ASR support
    enable_partials: bool = False        # Feature flag for partial results
    partial_interval_ms: int = 500       # Generate partials every 500ms
    partial_threshold_ms: int = 200      # Minimum speech before partials
    
    # Streaming ASR support
    enable_streaming: bool = False       # Feature flag for streaming ASR
    streaming_interval_ms: int = 200     # Generate streaming results every 200ms
    streaming_min_chars: int = 3         # Minimum characters before streaming


@dataclass
class ASREvent:
    type: ASREventType
    text: str
    t: float


class _FWBackend:
    """
    Enhanced ASR backend with optional partial results support.
    Accumulates audio in a buffer and can generate partials or final results.
    """
    def __init__(self, cfg: ASRConfig):
        if WhisperModel is None:
            raise RuntimeError("faster-whisper not installed")
        self.cfg = cfg
        # Compute sample sizes
        self.samples_per_frame = int(cfg.sample_rate * cfg.frame_ms / 1000)
        self.samples_max = int(cfg.sample_rate * cfg.max_window_ms / 1000)

        # Log model initialization
        print(f"[ASR] 🎯 Initializing Faster-Whisper with model: {cfg.faster_model}")
        print(f"[ASR] 📊 Sample rate: {cfg.sample_rate}Hz, Frame size: {cfg.frame_ms}ms")
        if cfg.enable_partials:
            print(f"[ASR] ⚡ Enhanced mode: partials every {cfg.partial_interval_ms}ms, max={cfg.max_window_ms}ms")
        else:
            print(f"[ASR] ⚡ Simplified mode: decode only on flush, max={cfg.max_window_ms}ms")

        # Model init: integer compute_type keeps it lean; adjust if you have GPU.
        self.model = WhisperModel(cfg.faster_model, compute_type="int8")
        print(f"[ASR] ✅ Faster-Whisper model '{cfg.faster_model}' loaded successfully")
        self._buf = np.zeros(0, dtype=np.int16)
        self._closed = False
        
        # Partial results tracking
        self._last_partial_time = 0.0
        self._partial_threshold_samples = int(cfg.sample_rate * cfg.partial_threshold_ms / 1000)
        
        # Streaming results tracking
        self._last_streaming_time = 0.0
        self._streaming_threshold_samples = int(cfg.sample_rate * cfg.streaming_interval_ms / 1000)

    def _int16_bytes_to_array(self, chunk: bytes) -> np.ndarray:
        arr = np.frombuffer(chunk, dtype="<i2")  # little-endian int16
        return arr

    def _trim_context(self):
        # Keep only the last max_window region
        if self._buf.shape[0] > self.samples_max:
            self._buf = self._buf[-self.samples_max :].copy()

    def _now(self) -> float:
        return time.time()

    def _decode_text(self, f32: np.ndarray) -> str:
        # Keep options latency-friendly; avoid heavy VAD to keep determinism.
        segments, _info = self.model.transcribe(
            f32,
            language=self.cfg.language,
            vad_filter=False,
            beam_size=self.cfg.beam_size,
            best_of=1,
            condition_on_previous_text=False,
            no_speech_threshold=0.2,
            temperature=0.0,
            patience=1.0,
            suppress_blank=True,
        )
        text = "".join(seg.text for seg in segments).strip()
        return text

    async def feed(self, frame_bytes: bytes) -> Optional[ASREvent]:
        """
        Feed a 20ms PCM16 frame. Accumulates audio and optionally generates partials or streaming results.
        Returns ASREvent if partials or streaming are enabled and conditions are met.
        """
        if self._closed:
            return None
        # Append audio
        arr = self._int16_bytes_to_array(frame_bytes)
        if arr.size == 0:
            return None
        self._buf = np.concatenate([self._buf, arr])
        self._trim_context()
        
        # Generate partial results if enabled
        if self.cfg.enable_partials:
            return await self._maybe_generate_partial()
        
        # Generate streaming results if enabled
        if self.cfg.enable_streaming:
            return await self._maybe_generate_streaming()
        
        return None
    
    async def _maybe_generate_partial(self) -> Optional[ASREvent]:
        """Generate partial result if conditions are met."""
        now = time.time()
        
        # Check if enough time has passed since last partial
        time_since_last = now - self._last_partial_time
        interval_seconds = self.cfg.partial_interval_ms / 1000.0
        
        if time_since_last < interval_seconds:
            return None
        
        # Check if we have enough audio for a partial
        if self._buf.shape[0] < self._partial_threshold_samples:
            return None
        
        # Generate partial result
        self._last_partial_time = now
        text = self._decode_text(self._buf.astype(np.float32) / 32768.0)
        
        if text and text.strip():
            return ASREvent(type="partial", text=text.strip(), t=now)
        
        return None

    async def _maybe_generate_streaming(self) -> Optional[ASREvent]:
        """Generate streaming result if conditions are met."""
        now = time.time()
        
        # Check if enough time has passed since last streaming result
        time_since_last = now - self._last_streaming_time
        interval_seconds = self.cfg.streaming_interval_ms / 1000.0
        
        if time_since_last < interval_seconds:
            return None
        
        # Check if we have enough audio for streaming
        if self._buf.shape[0] < self._streaming_threshold_samples:
            return None
        
        # Generate streaming result
        self._last_streaming_time = now
        text = self._decode_text(self._buf.astype(np.float32) / 32768.0)
        
        # Only return if we have enough characters
        if text and len(text.strip()) >= self.cfg.streaming_min_chars:
            return ASREvent(type="streaming", text=text.strip(), t=now)
        
        return None

    async def flush(self) -> Optional[ASREvent]:
        """
        Force a final decode and return a FINAL event (if any text).
        Resets state for the next utterance.
        """
        if self._closed:
            return None
        if self._buf.shape[0] == 0:
            return None

        f32 = (self._buf.astype(np.float32)) / 32768.0
        text = self._decode_text(f32)

        # reset utterance state
        self._buf = np.zeros(0, dtype=np.int16)
        self._last_partial_time = 0.0  # Reset partial timer
        self._last_streaming_time = 0.0  # Reset streaming timer

        if text:
            return ASREvent(type="final", text=text, t=self._now())
        return None

    async def close(self):
        self._closed = True
        # model has no explicit close; GC will handle

    # Optional explicit reset for hard boundaries (start/stop)
    def reset(self):
        self._buf = np.zeros(0, dtype=np.int16)




class StreamingASR:
    """
    Simplified ASR using faster-whisper:
      asr = StreamingASR(cfg)
      await asr.feed(frame_bytes)          # Just accumulates audio
      final = await asr.flush()            # -> ASREvent | None
      await asr.close()

    - No partials, only final results
    - Accumulates audio until flush
    - Single decode per utterance
    """
    def __init__(self, cfg: Optional[ASRConfig] = None):
        self.cfg = cfg or ASRConfig()
        self.impl = _FWBackend(self.cfg)

    async def feed(self, frame_bytes: bytes) -> Optional[ASREvent]:
        return await self.impl.feed(frame_bytes)

    async def flush(self) -> Optional[ASREvent]:
        return await self.impl.flush()

    async def close(self):
        await self.impl.close()

    def reset(self):
        reset_fn = getattr(self.impl, "reset", None)
        if callable(reset_fn):
            reset_fn()
