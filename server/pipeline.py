from __future__ import annotations

import asyncio
import base64
import contextlib
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import orjson

from .asr import ASRConfig, ASREvent, StreamingASR
from .logging import RichLogger, NDJSONLogger
from .planner import plan
from .settings import settings
from .tts import OptimizedStreamingTTS as StreamingTTS, TTSConfig
from .vad import StreamingVAD
from .wire import (
    MSG_AUDIO,
    MSG_CUT_PLAYBACK,
    MSG_DEBUG_TTS,
    MSG_FINAL,
    MSG_PARTIAL,
    MSG_SPEAK_START,
    MSG_STREAMING,
)

TurnState = Literal["LISTEN", "PLAN", "SPEAK"]


def _get_bool(key: str, default: bool = False) -> bool:
    """Helper function to parse boolean environment variables."""
    v = os.getenv(key)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")


# ----------------- pipeline config -----------------
@dataclass
class PipelineConfig:
    # VAD / endpointing
    min_vad_consecutive: int = 2  # How many consecutive frames must show speech to trigger VAD
    endpoint_silence_ms: int = 150  # How long to wait in silence before ending user speech
    min_utterance_ms: int = 200  # Minimum speech duration to process

    # ASR windowing (simplified)
    asr_max_window_ms: int = 4000  # Maximum audio context for speech recognition (prevents memory issues)

    # Barge-in controls
    barge_in_grace_ms: int = 100  # How long to ignore barge-in after TTS starts
    barge_in_min_rms: float = 0.1  # Minimum audio energy required to trigger barge-in
    min_audio_rms: float = 0.005  # Minimum audio energy for frame processing and logging

    # Duplex echo suppression (mute ASR while we stream TTS)
    duplex_mute_ms: int = 300  # How long to ignore microphone input after each TTS packet

    # Budgets (for metrics only - performance targets)
    budget_first_partial_ms: int = 250  # Target time for first ASR result
    budget_planner_ms: int = 150  # Target time for LLM planning
    budget_tts_first_ms: int = 200  # Target time for first TTS audio
    budget_p95_rt_ms: int = 600  # Target 95th percentile response time

    # Partial ASR Events (feature flag controlled)
    enable_partials: bool = False  # Feature flag for partial ASR results
    partial_interval_ms: int = 500  # Generate partials every 500ms
    partial_threshold_ms: int = 200  # Minimum speech before partials

    # Parallel LLM+TTS (feature flag controlled)
    enable_parallel_llm_tts: bool = False  # Feature flag for parallel execution
    parallel_tts_start_delay_ms: int = 100  # Delay before starting TTS synthesis
    parallel_max_wait_ms: int = 2000  # Maximum time to wait for parallel completion

    # Streaming ASR + Full Parallel (feature flag controlled)
    enable_streaming_asr: bool = False  # Feature flag for streaming ASR
    enable_full_parallel: bool = False  # Feature flag for full parallel pipeline
    streaming_asr_interval_ms: int = 200  # Generate streaming results every 200ms
    streaming_asr_min_chars: int = 3  # Minimum characters before streaming
    full_parallel_max_wait_ms: int = 3000  # Maximum time for full parallel execution

    def __post_init__(self):
        """Validate configuration values to ensure they're reasonable."""
        # Override with environment variables if set
        # Only override if environment variable is set (allows explicit parameter override)
        if os.getenv("VHYS_ENABLE_PARTIALS") is not None:
            self.enable_partials = _get_bool("VHYS_ENABLE_PARTIALS", False)
        if os.getenv("VHYS_PARTIAL_INTERVAL_MS") is not None:
            self.partial_interval_ms = int(os.getenv("VHYS_PARTIAL_INTERVAL_MS", "500"))
        if os.getenv("VHYS_PARTIAL_THRESHOLD_MS") is not None:
            self.partial_threshold_ms = int(os.getenv("VHYS_PARTIAL_THRESHOLD_MS", "200"))
        
        if os.getenv("VHYS_ENABLE_PARALLEL_LLM_TTS") is not None:
            self.enable_parallel_llm_tts = _get_bool("VHYS_ENABLE_PARALLEL_LLM_TTS", False)
        if os.getenv("VHYS_PARALLEL_TTS_START_DELAY_MS") is not None:
            self.parallel_tts_start_delay_ms = int(os.getenv("VHYS_PARALLEL_TTS_START_DELAY_MS", "100"))
        if os.getenv("VHYS_PARALLEL_MAX_WAIT_MS") is not None:
            self.parallel_max_wait_ms = int(os.getenv("VHYS_PARALLEL_MAX_WAIT_MS", "2000"))
        
        if os.getenv("VHYS_ENABLE_STREAMING_ASR") is not None:
            self.enable_streaming_asr = _get_bool("VHYS_ENABLE_STREAMING_ASR", False)
        if os.getenv("VHYS_ENABLE_FULL_PARALLEL") is not None:
            self.enable_full_parallel = _get_bool("VHYS_ENABLE_FULL_PARALLEL", False)
        if os.getenv("VHYS_STREAMING_ASR_INTERVAL_MS") is not None:
            self.streaming_asr_interval_ms = int(os.getenv("VHYS_STREAMING_ASR_INTERVAL_MS", "200"))
        if os.getenv("VHYS_STREAMING_ASR_MIN_CHARS") is not None:
            self.streaming_asr_min_chars = int(os.getenv("VHYS_STREAMING_ASR_MIN_CHARS", "3"))
        if os.getenv("VHYS_FULL_PARALLEL_MAX_WAIT_MS") is not None:
            self.full_parallel_max_wait_ms = int(os.getenv("VHYS_FULL_PARALLEL_MAX_WAIT_MS", "3000"))
        
        # Validate VAD/endpointing parameters
        if self.endpoint_silence_ms < 50:
            raise ValueError(f"endpoint_silence_ms must be >= 50ms, got {self.endpoint_silence_ms}")
        if self.min_utterance_ms < 100:
            raise ValueError(f"min_utterance_ms must be >= 100ms, got {self.min_utterance_ms}")
        # Note: min_utterance_ms can be > endpoint_silence_ms - they serve different purposes
        # min_utterance_ms: minimum speech duration to process
        # endpoint_silence_ms: silence duration before ending user speech
        
        # Validate barge-in parameters
        if self.barge_in_grace_ms < 50:
            raise ValueError(f"barge_in_grace_ms must be >= 50ms, got {self.barge_in_grace_ms}")
        if not 0.01 <= self.barge_in_min_rms <= 1.0:
            raise ValueError(f"barge_in_min_rms must be between 0.01 and 1.0, got {self.barge_in_min_rms}")
        
        # Validate duplex parameters
        if self.duplex_mute_ms < 100:
            raise ValueError(f"duplex_mute_ms must be >= 100ms, got {self.duplex_mute_ms}")
        
        # Validate partial ASR parameters
        if self.partial_interval_ms < 100:
            raise ValueError(f"partial_interval_ms must be >= 100ms, got {self.partial_interval_ms}")
        if self.partial_threshold_ms < 50:
            raise ValueError(f"partial_threshold_ms must be >= 50ms, got {self.partial_threshold_ms}")
        
        # Validate parallel processing parameters
        if self.parallel_tts_start_delay_ms < 0:
            raise ValueError(f"parallel_tts_start_delay_ms must be >= 0ms, got {self.parallel_tts_start_delay_ms}")
        if self.parallel_max_wait_ms < 500:
            raise ValueError(f"parallel_max_wait_ms must be >= 500ms, got {self.parallel_max_wait_ms}")
        
        # Validate streaming parameters
        if self.streaming_asr_interval_ms < 50:
            raise ValueError(f"streaming_asr_interval_ms must be >= 50ms, got {self.streaming_asr_interval_ms}")
        if self.streaming_asr_min_chars < 1:
            raise ValueError(f"streaming_asr_min_chars must be >= 1, got {self.streaming_asr_min_chars}")
        if self.full_parallel_max_wait_ms < 1000:
            raise ValueError(f"full_parallel_max_wait_ms must be >= 1000ms, got {self.full_parallel_max_wait_ms}")
        
        # Log pipeline configuration summary
        optimizations = []
        if self.enable_partials: optimizations.append("Partial ASR")
        if self.enable_parallel_llm_tts: optimizations.append("Parallel LLM+TTS")
        if self.enable_streaming_asr: optimizations.append("Streaming ASR")
        if self.enable_full_parallel: optimizations.append("Full Parallel")
        
        if optimizations:
            print(f"[PipelineConfig] ⚡ Optimizations: {', '.join(optimizations)}")
        else:
            print(f"[PipelineConfig] ⚡ Optimizations: None enabled")


# ----------------- helpers -----------------
def _rms16(frame_bytes: bytes) -> float:
    if not frame_bytes:
        return 0.0
    x = np.frombuffer(frame_bytes, dtype="<i2").astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(x * x)))


_JUNK_OK_RE = re.compile(r"^(ok(?:ay)?[.!]?\s*){3,}$", re.IGNORECASE)
_JUNK_FILLER_RE = re.compile(r"^(?:(um|uh|er|like|you know)[,.\s]*){3,}$", re.IGNORECASE)


def _looks_like_junk_final(text: str) -> bool:
    """Filter obvious echo/noise finals that derail the LLM."""
    t = (text or "").strip()
    if not t:
        return True
    if len(t) <= 2:
        return True
    if _JUNK_OK_RE.match(t):
        return True
    if _JUNK_FILLER_RE.match(t):
        return True
    toks = re.findall(r"\w+", t.lower())
    if toks and len(toks) >= 8 and len(set(toks)) <= 2:
        return True
    return False


def _should_buffer_frame(frame_bytes: bytes, rms: float, min_rms: float) -> bool:
    """Only buffer frames with sufficient energy and valid content (prevents empty/noisy frames)."""
    if len(frame_bytes) == 0:
        return False
    if rms < min_rms:  # Too quiet (likely silence)
        return False
    if rms > 0.8:   # Too loud (likely noise/echo)
        return False
    return True


# ----------------- main pipeline -----------------
class VoicePipeline:
    def __init__(self, session, cfg: Optional[PipelineConfig] = None):
        self.session = session
        self.cfg = cfg or PipelineConfig()

        # Log pipeline initialization with model info
        session_info = RichLogger.session_info(self.session.id, 0, "INIT")
        print(f"[{RichLogger._format_time()}] {session_info} 🚀 Initializing Voice Pipeline")
        print(f"[{RichLogger._format_time()}] {session_info} 🎯 ASR Model: {settings.faster_whisper_model}")
        print(f"[{RichLogger._format_time()}] {session_info} 🎤 TTS: {'Piper' if settings.use_piper else 'Disabled'}")
        print(f"[{RichLogger._format_time()}] {session_info} 🤖 LLM: {settings.llm_model}")

        # Core components
        self.vad = StreamingVAD()
        self.asr = StreamingASR(
            ASRConfig(
                sample_rate=settings.sample_rate,
                frame_ms=settings.frame_ms,
                max_window_ms=self.cfg.asr_max_window_ms,
                faster_model=settings.faster_whisper_model,
                # Pass partial configuration
                enable_partials=self.cfg.enable_partials,
                partial_interval_ms=self.cfg.partial_interval_ms,
                partial_threshold_ms=self.cfg.partial_threshold_ms,
                # Pass streaming configuration
                enable_streaming=self.cfg.enable_streaming_asr,
                streaming_interval_ms=self.cfg.streaming_asr_interval_ms,
                streaming_min_chars=self.cfg.streaming_asr_min_chars,
            )
        )
        self.tts = StreamingTTS(
            TTSConfig(
                out_sample_rate=16000,
                frame_ms=settings.frame_ms,
                use_piper=settings.use_piper,
                piper_model_path=settings.piper_model_path,
                piper_config_path=settings.piper_config_path,
                # TTS optimization settings
                pool_size=3,  # Pre-warm 3 processes
                warmup_text="Hello",  # Text to warm up processes
                enable_caching=True,  # Cache common responses
                cache_size=100  # Max cached responses
            )
        )
        
        # Initialize TTS pool at startup
        self._tts_pool_initialized = False

        # Turn / state
        self.state: TurnState = "LISTEN"
        self.turn_id = 0
        self._tts_task: Optional[asyncio.Task] = None
        self._interrupted = False

        # Endpointing counters
        self._seen_speech_frames = 0
        self._silence_frames = 0
        self._had_any_partial = False

        # Timing
        self._t_turn_start = 0.0
        self._t_asr_first_event = 0.0
        self._t_asr_final = 0.0
        self._t_plan_done = 0.0
        self._t_tts_first = 0.0
        self._t_speak_start = 0.0
        
        # Detailed gap analysis timing
        self._t_asr_final_to_plan_start = 0.0
        self._t_plan_start_to_plan_done = 0.0
        self._t_plan_done_to_tts_start = 0.0
        self._t_tts_start_to_tts_first = 0.0

        # Duplex echo suppression window deadline
        self._mute_asr_until = 0.0

        # Pre-barge buffer: keep ~200 ms (10×20 ms) of frames while speaking (captures quick responses like "yes", "no")
        self._pre_barge_frames: deque[bytes] = deque(maxlen=10)

        # Derived frame counts
        self._endpoint_silence_frames = max(1, int(self.cfg.endpoint_silence_ms / settings.frame_ms))
        self._min_utterance_frames = max(5, int(self.cfg.min_utterance_ms / settings.frame_ms))

        # Metrics - clear previous metrics on startup
        metrics_path = Path(settings.metrics_file)
        if metrics_path.exists():
            metrics_path.unlink()  # Remove previous metrics file
            print(f"[{RichLogger._format_time()}] {session_info} 🗑️  Cleared previous metrics: {settings.metrics_file}")
        self._logger = NDJSONLogger(settings.metrics_file)

    # ------------- controls -------------
    async def handle_control(self, obj: dict):
        typ = obj.get("type")

        if typ == "start":
            # Treat start as an absolute boundary
            session_info = RichLogger.session_info(self.session.id, self.turn_id, self.state)
            print(f"[{RichLogger._format_time()}] {session_info} {RichLogger.session_start()}")
            await self._hard_reset(reason="client_start")
            # Optionally bump logical turn counter for clarity in metrics
            self.turn_id += 1
            
            # Initialize TTS pool on first session start
            if not self._tts_pool_initialized:
                print(f"[{RichLogger._format_time()}] {session_info} 🔥 Initializing TTS pool...")
                await self.tts.initialize_pool()
                self._tts_pool_initialized = True
                print(f"[{RichLogger._format_time()}] {session_info} ✅ TTS pool ready!")
            
            return

        elif typ == "stop":
            # Finalize (emitting any last final), then hard reset everything
            session_info = RichLogger.session_info(self.session.id, self.turn_id, self.state)
            print(f"[{RichLogger._format_time()}] {session_info} {RichLogger.session_stop()}")
            with contextlib.suppress(Exception):
                await self._finalize_turn(flush_asr=True, reason="client_stop")
            await self._hard_reset(reason="post_client_stop")
            return

        elif typ == "barge_in":
            session_info = RichLogger.session_info(self.session.id, self.turn_id, self.state)
            print(f"[{RichLogger._format_time()}] {session_info} {RichLogger.barge_in()}")
            await self._on_barge_in()
            return

        elif typ == MSG_DEBUG_TTS:
            # manual TTS sanity
            text = (obj.get("text") or "Hello from Piper").strip()
            if self.state == "SPEAK":
                await self._on_barge_in()
            self.state = "SPEAK"
            self._t_speak_start = time.time()
            await self.session.send_json({"type": MSG_SPEAK_START, "t": self._t_speak_start})
            seq = 0
            total = 0
            async for chunk in self.tts.synth(text):
                b = getattr(chunk, "pcm_bytes", b"")
                total += len(b)
                # refresh duplex mute on every outbound packet
                self._mute_asr_until = time.time() + (self.cfg.duplex_mute_ms / 1000.0)
                await self.session.send_json(
                    {
                        "type": MSG_AUDIO,
                        "seq": getattr(chunk, "seq", seq),
                        "pcm_base64": base64.b64encode(b).decode("ascii"),
                    }
                )
                seq += 1
            print(f"[TTS debug] streamed chunks={seq} bytes={total}")
            self.state = "LISTEN"
            return

    # ------------- audio path -------------
    async def handle_audio(self, frame_bytes: bytes):
        now = time.time()

        # compute gates early (VAD runs regardless so it can detect barge-in)
        is_speech_gate = self.vad.update(frame_bytes)
        rms = _rms16(frame_bytes)

        # Rich logging for audio input (reduced frequency and smarter filtering)
        if not hasattr(self, '_audio_frame_count'):
            self._audio_frame_count = 0
        self._audio_frame_count += 1
        
        # Log audio input less frequently and only when there's actual activity
        should_log = (
            self._audio_frame_count % 50 == 0 or  # Every 1 second (50 * 20ms)
            (is_speech_gate and rms > 0.05) or    # Speech detected (increased threshold)
            (self.state == "LISTEN" and self._audio_frame_count % 10 == 0 and rms > self.cfg.min_audio_rms)  # More frequent in LISTEN, but only with sufficient energy
        )
        
        if should_log:
            session_info = RichLogger.session_info(self.session.id, self.turn_id, self.state)
            audio_info = RichLogger.audio_input(len(frame_bytes), rms, is_speech_gate)
            print(f"[{RichLogger._format_time()}] {session_info} {audio_info}")

        if self.state == "SPEAK":
            # keep a small rolling buffer while we speak, for barge-in priming (only buffer meaningful frames)
            if _should_buffer_frame(frame_bytes, rms, self.cfg.min_audio_rms):
                self._pre_barge_frames.append(frame_bytes)

            # ignore echo during tiny grace at start of TTS
            if (now - self._t_speak_start) * 1000.0 < self.cfg.barge_in_grace_ms:
                return

            # allow barge-in only when we see real speech (VAD + RMS)
            if is_speech_gate and rms >= self.cfg.barge_in_min_rms:
                print(f"[{RichLogger._format_time()}] {RichLogger.barge_in()}")
                prime = list(self._pre_barge_frames)  # include current
                await self._on_barge_in(prime_frames=prime)
            return

        # Not speaking: if within mute window (from TTS tail), drop frames to ASR/VAD
        if now < self._mute_asr_until:
            return

        if self.state == "LISTEN":
            if self.vad.raw_is_speech or is_speech_gate:
                # Only process frames with sufficient energy (consistent with buffering logic)
                if _should_buffer_frame(frame_bytes, rms, self.cfg.min_audio_rms):
                    if self._t_turn_start == 0.0:
                        await self._reset_for_next_utterance(new_turn=True)
                    self._seen_speech_frames += 1
                    self._silence_frames = 0
                    # Feed audio and check for partial results
                    partial_event = await self.asr.feed(frame_bytes)
                    if partial_event:
                        await self._emit_asr_event(partial_event)
            else:
                if self._seen_speech_frames > 0:
                    self._silence_frames += 1
                    if (
                        self._silence_frames >= self._endpoint_silence_frames
                        and self._seen_speech_frames >= self._min_utterance_frames
                    ):
                        # Process complete utterance
                        final = await self.asr.flush()
                        self._seen_speech_frames = 0
                        self._silence_frames = 0
                        if final and (final.text or "").strip():
                            if not _looks_like_junk_final(final.text):
                                await self._emit_asr_event(final)
                            else:
                                await self._reset_for_next_utterance(new_turn=False)
                        else:
                            await self._reset_for_next_utterance(new_turn=False)

    async def close(self):
        with contextlib.suppress(Exception):
            if self._tts_task and not self._tts_task.done():
                self._tts_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self._tts_task
        await self.asr.close()
        await self.tts.close()

    # ------------- internals -------------
    async def _hard_reset(self, *, reason: str):
        """
        Absolute boundary: stop playback, flush/clear ASR buffers, reset VAD and timers.
        Leaves the pipeline in a clean LISTEN state with no pending partials.
        """
        # Stop TTS if running
        if self.state == "SPEAK":
            with contextlib.suppress(Exception):
                await self.session.send_json({"type": MSG_CUT_PLAYBACK, "t": time.monotonic()})
        if self._tts_task and not self._tts_task.done():
            self._tts_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._tts_task

        # Flush ASR but do not emit to client
        with contextlib.suppress(Exception):
            _ = await self.asr.flush()

        # Try explicit ASR reset if provided
        reset_fn = getattr(self.asr, "reset", None)
        if callable(reset_fn):
            with contextlib.suppress(Exception):
                reset_fn()

        # Reset VAD & local counters/buffers
        self.vad.reset()
        self._seen_speech_frames = 0
        self._silence_frames = 0
        self._had_any_partial = False
        self._pre_barge_frames.clear()

        # Clear timing so next turn starts fresh
        self._t_turn_start = 0.0
        self._t_asr_first_event = 0.0
        self._t_asr_final = 0.0
        self._t_plan_done = 0.0
        self._t_tts_first = 0.0
        self._t_speak_start = 0.0
        
        # Clear gap analysis timing
        self._t_asr_final_to_plan_start = 0.0
        self._t_plan_start_to_plan_done = 0.0
        self._t_plan_done_to_tts_start = 0.0
        self._t_tts_start_to_tts_first = 0.0

        # End duplex mute immediately
        self._mute_asr_until = 0.0

        # Back to listen; do not increment turn id here
        self.state = "LISTEN"

        # Log boundary
        self._logger.write(
            {"t": time.time(), "evt": "hard_reset", "sid": self.session.id, "reason": reason}
        )

    async def _reset_for_next_utterance(self, *, new_turn: bool):
        self._seen_speech_frames = 0
        self._silence_frames = 0
        self._had_any_partial = False
        self.vad.reset()
        if new_turn:
            self.turn_id += 1
            self._interrupted = False
            now = time.time()
            self._t_turn_start = now
            self._t_asr_first_event = 0.0
            self._t_asr_final = 0.0
            self._t_plan_done = 0.0
            self._t_tts_first = 0.0
            self._t_speak_start = 0.0
            self._mute_asr_until = 0.0
            self._pre_barge_frames.clear()
            
            # Clear gap analysis timing
            self._t_asr_final_to_plan_start = 0.0
            self._t_plan_start_to_plan_done = 0.0
            self._t_plan_done_to_tts_start = 0.0
            self._t_tts_start_to_tts_first = 0.0
            
            self.state = "LISTEN"

    async def _emit_asr_event(self, evt: ASREvent):
        if self._t_asr_first_event == 0.0:
            self._t_asr_first_event = evt.t
        
        # Calculate latency for logging
        latency_ms = (evt.t - self._t_turn_start) * 1000 if self._t_turn_start > 0 else 0
        
        session_info = RichLogger.session_info(self.session.id, self.turn_id, self.state)
        
        if evt.type == "partial":
            # Handle partial results
            asr_info = RichLogger.asr_partial(evt.text, latency_ms)
            print(f"[{RichLogger._format_time()}] {session_info} {asr_info}")
            await self.session.send_json({"type": MSG_PARTIAL, "text": evt.text, "t": evt.t})
            
        elif evt.type == "streaming":
            # Handle streaming results
            asr_info = RichLogger.asr_streaming(evt.text, latency_ms)
            print(f"[{RichLogger._format_time()}] {session_info} {asr_info}")
            await self.session.send_json({"type": MSG_STREAMING, "text": evt.text, "t": evt.t})
            
        elif evt.type == "final":
            # Handle final results
            self._t_asr_final = evt.t
            asr_info = RichLogger.asr_final(evt.text, latency_ms)
            print(f"[{RichLogger._format_time()}] {session_info} {asr_info}")
            await self.session.send_json({"type": MSG_FINAL, "text": evt.text, "t": evt.t})
            await self._plan_and_speak(evt.text)

    async def _plan_and_speak(self, user_text: str):
        if self.state != "LISTEN":
            return
        
        # Track gap: ASR final to planning start
        self._t_asr_final_to_plan_start = time.time()
        gap_asr_to_plan = (self._t_asr_final_to_plan_start - self._t_asr_final) * 1000 if self._t_asr_final > 0 else 0
        
        # Log gap analysis
        session_info = RichLogger.session_info(self.session.id, self.turn_id, self.state)
        if gap_asr_to_plan > 50:  # Only log significant gaps
            gap_info = RichLogger.timing(f"Gap: ASR→Plan", gap_asr_to_plan)
            print(f"[{RichLogger._format_time()}] {session_info} {gap_info}")
        
        # Choose execution mode based on configuration
        if self.cfg.enable_full_parallel:
            await self._plan_and_speak_full_parallel(user_text)
        elif self.cfg.enable_parallel_llm_tts:
            await self._plan_and_speak_parallel(user_text)
        else:
            await self._plan_and_speak_sequential(user_text)

    async def _plan_and_speak_sequential(self, user_text: str):
        """Original sequential implementation for backward compatibility."""
        # Log state transition
        session_info = RichLogger.session_info(self.session.id, self.turn_id, self.state)
        transition = RichLogger.state_transition("LISTEN", "PLAN", "user_input")
        print(f"[{RichLogger._format_time()}] {session_info} {transition}")
        
        self.state = "PLAN"
        
        # Log planning start
        planning_info = RichLogger.planning_start(user_text)
        print(f"[{RichLogger._format_time()}] {session_info} {planning_info}")
        
        # Track planning start time
        t_plan_start = time.time()
        self._t_plan_start_to_plan_done = t_plan_start
        
        try:
            reply_text, channel, score = plan(user_text, ctx={"sid": self.session.id})
            
            # Log planning response
            response_info = RichLogger.planning_response(reply_text, channel, score)
            print(f"[{RichLogger._format_time()}] {session_info} {response_info}")
            
        except Exception as e:
            import traceback
            error_info = RichLogger.error(f"Planning failed: {repr(e)}")
            print(f"[{RichLogger._format_time()}] {session_info} {error_info}")
            traceback.print_exc()
            reply_text = "I hit a snag. Want to see available times?"
            channel = "voice"
            score = 0.5
        
        self._t_plan_done = time.time()
        
        # Log planning timing
        plan_duration = (self._t_plan_done - self._t_asr_final) * 1000 if self._t_asr_final > 0 else 0
        timing_info = RichLogger.timing("Planning", plan_duration)
        print(f"[{RichLogger._format_time()}] {session_info} {timing_info}")

        # Log state transition to SPEAK
        transition = RichLogger.state_transition("PLAN", "SPEAK", "ready_to_speak")
        print(f"[{RichLogger._format_time()}] {session_info} {transition}")
        
        self.state = "SPEAK"
        self._t_speak_start = time.time()
        
        # Track gap: planning done to TTS start
        self._t_plan_done_to_tts_start = self._t_speak_start
        gap_plan_to_tts = (self._t_plan_done_to_tts_start - self._t_plan_done) * 1000 if self._t_plan_done > 0 else 0
        
        # Log gap analysis
        if gap_plan_to_tts > 50:  # Only log significant gaps
            gap_info = RichLogger.timing(f"Gap: Plan→TTS", gap_plan_to_tts)
            print(f"[{RichLogger._format_time()}] {session_info} {gap_info}")
        
        await self.session.send_json({"type": MSG_SPEAK_START, "t": self._t_plan_done})
        self._tts_task = asyncio.create_task(self._tts_stream(reply_text))

    async def _plan_and_speak_parallel(self, user_text: str):
        """Parallel LLM planning and TTS synthesis for 20-30% latency improvement."""
        # Log state transition
        session_info = RichLogger.session_info(self.session.id, self.turn_id, self.state)
        transition = RichLogger.state_transition("LISTEN", "PLAN", "user_input")
        print(f"[{RichLogger._format_time()}] {session_info} {transition}")
        
        self.state = "PLAN"
        
        # Log planning start
        planning_info = RichLogger.planning_start(user_text)
        print(f"[{RichLogger._format_time()}] {session_info} {planning_info}")
        
        # Start LLM planning task
        planning_task = asyncio.create_task(self._plan_async(user_text))
        
        # Wait for a short delay before starting TTS (allows LLM to get started)
        if self.cfg.parallel_tts_start_delay_ms > 0:
            await asyncio.sleep(self.cfg.parallel_tts_start_delay_ms / 1000.0)
        
        # Start TTS synthesis task with a placeholder text (will be updated when planning completes)
        tts_task = asyncio.create_task(self._tts_stream_parallel(""))
        
        try:
            # Wait for both tasks to complete or timeout
            done, pending = await asyncio.wait(
                [planning_task, tts_task],
                timeout=self.cfg.parallel_max_wait_ms / 1000.0,
                return_when=asyncio.ALL_COMPLETED
            )
            
            # Cancel any pending tasks
            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            
            # Get planning result
            if planning_task in done:
                reply_text, channel, score = await planning_task
            else:
                # Fallback if planning didn't complete
                reply_text = "I hit a snag. Want to see available times?"
                channel = "voice"
                score = 0.5
            
            # Update TTS with actual text if it's still running
            if not tts_task.done():
                # We need to restart TTS with the actual text
                tts_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await tts_task
                
                # Start new TTS with actual text
                self._tts_task = asyncio.create_task(self._tts_stream(reply_text))
            else:
                # TTS task completed, start new one with actual text
                self._tts_task = asyncio.create_task(self._tts_stream(reply_text))
            
        except Exception as e:
            import traceback
            error_info = RichLogger.error(f"Parallel execution failed: {repr(e)}")
            print(f"[{RichLogger._format_time()}] {session_info} {error_info}")
            traceback.print_exc()
            
            # Fallback to sequential execution
            await self._plan_and_speak_sequential(user_text)
            return
        
        self._t_plan_done = time.time()
        
        # Log planning timing
        plan_duration = (self._t_plan_done - self._t_asr_final) * 1000 if self._t_asr_final > 0 else 0
        timing_info = RichLogger.timing("Planning (Parallel)", plan_duration)
        print(f"[{RichLogger._format_time()}] {session_info} {timing_info}")

        # Log state transition to SPEAK
        transition = RichLogger.state_transition("PLAN", "SPEAK", "ready_to_speak")
        print(f"[{RichLogger._format_time()}] {session_info} {transition}")
        
        self.state = "SPEAK"
        self._t_speak_start = time.time()
        await self.session.send_json({"type": MSG_SPEAK_START, "t": self._t_plan_done})
        
        # Await the TTS task to ensure audio is sent
        if self._tts_task and not self._tts_task.done():
            await self._tts_task

    async def _plan_and_speak_full_parallel(self, user_text: str):
        """Full parallel pipeline execution for maximum latency improvement."""
        # Log state transition
        session_info = RichLogger.session_info(self.session.id, self.turn_id, self.state)
        transition = RichLogger.state_transition("LISTEN", "PLAN", "user_input")
        print(f"[{RichLogger._format_time()}] {session_info} {transition}")
        
        self.state = "PLAN"
        
        # Log planning start
        planning_info = RichLogger.planning_start(user_text)
        print(f"[{RichLogger._format_time()}] {session_info} {planning_info}")
        
        # Start planning task
        planning_task = asyncio.create_task(self._plan_async(user_text))
        
        try:
            # Wait for planning to complete or timeout
            done, pending = await asyncio.wait(
                [planning_task],
                timeout=self.cfg.full_parallel_max_wait_ms / 1000.0,
                return_when=asyncio.ALL_COMPLETED
            )
            
            # Cancel any pending tasks
            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            
            # Get planning result
            if planning_task in done:
                reply_text, channel, score = await planning_task
            else:
                # Fallback if planning didn't complete
                reply_text = "I hit a snag. Want to see available times?"
                channel = "voice"
                score = 0.5
            
            # Start TTS with actual text
            self._tts_task = asyncio.create_task(self._tts_stream(reply_text))
            
        except Exception as e:
            import traceback
            error_info = RichLogger.error(f"Full parallel execution failed: {repr(e)}")
            print(f"[{RichLogger._format_time()}] {session_info} {error_info}")
            traceback.print_exc()
            
            # Fallback to sequential execution
            await self._plan_and_speak_sequential(user_text)
            return
        
        self._t_plan_done = time.time()
        
        # Log planning timing
        plan_duration = (self._t_plan_done - self._t_asr_final) * 1000 if self._t_asr_final > 0 else 0
        timing_info = RichLogger.timing("Planning (Full Parallel)", plan_duration)
        print(f"[{RichLogger._format_time()}] {session_info} {timing_info}")

        # Log state transition to SPEAK
        transition = RichLogger.state_transition("PLAN", "SPEAK", "ready_to_speak")
        print(f"[{RichLogger._format_time()}] {session_info} {transition}")
        
        self.state = "SPEAK"
        self._t_speak_start = time.time()
        await self.session.send_json({"type": MSG_SPEAK_START, "t": self._t_plan_done})
        
        # Await the TTS task to ensure audio is sent
        if self._tts_task and not self._tts_task.done():
            await self._tts_task

    async def _plan_async(self, user_text: str):
        """Async wrapper for LLM planning."""
        try:
            reply_text, channel, score = plan(user_text, ctx={"sid": self.session.id})
            
            # Log planning response
            session_info = RichLogger.session_info(self.session.id, self.turn_id, self.state)
            response_info = RichLogger.planning_response(reply_text, channel, score)
            print(f"[{RichLogger._format_time()}] {session_info} {response_info}")
            
            return reply_text, channel, score
            
        except Exception as e:
            import traceback
            error_info = RichLogger.error(f"Planning failed: {repr(e)}")
            session_info = RichLogger.session_info(self.session.id, self.turn_id, self.state)
            print(f"[{RichLogger._format_time()}] {session_info} {error_info}")
            traceback.print_exc()
            return "I hit a snag. Want to see available times?", "voice", 0.5

    async def _tts_stream_parallel(self, text: str):
        """TTS streaming for parallel execution (placeholder implementation)."""
        # This is a placeholder - in a real implementation, we might start TTS
        # with a generic response and then update it when planning completes
        session_info = RichLogger.session_info(self.session.id, self.turn_id, self.state)
        
        # Log TTS start
        tts_info = RichLogger.tts_start(text or "Processing...")
        print(f"[{RichLogger._format_time()}] {session_info} {tts_info}")
        
        # For now, just wait a bit to simulate TTS startup
        await asyncio.sleep(0.1)
        
        # This will be replaced by the actual TTS stream when planning completes
        return

    async def _tts_stream(self, text: str):
        session_info = RichLogger.session_info(self.session.id, self.turn_id, self.state)
        
        # Log TTS start
        tts_info = RichLogger.tts_start(text)
        print(f"[{RichLogger._format_time()}] {session_info} {tts_info}")
        
        seq = 0
        total_bytes = 0
        try:
            async for chunk in self.tts.synth(text):
                if self.state != "SPEAK":
                    break
                if self._t_tts_first == 0.0:
                    self._t_tts_first = time.time()
                    # Log TTS first chunk timing
                    tts_latency = (self._t_tts_first - self._t_plan_done) * 1000 if self._t_plan_done > 0 else 0
                    timing_info = RichLogger.timing("TTS First Chunk", tts_latency)
                    print(f"[{RichLogger._format_time()}] {session_info} {timing_info}")
                
                b = getattr(chunk, "pcm_bytes", b"")
                total_bytes += len(b)
                
                # Log every 25th chunk to avoid spam (every 500ms at 20ms chunks)
                if seq % 100 == 0:
                    chunk_info = RichLogger.tts_chunk(seq, len(b))
                    print(f"[{RichLogger._format_time()}] {session_info} {chunk_info}")
                
                # refresh duplex mute window on every outbound audio packet
                self._mute_asr_until = time.time() + (self.cfg.duplex_mute_ms / 1000.0)
                await self.session.send_json(
                    {
                        "type": MSG_AUDIO,
                        "seq": getattr(chunk, "seq", seq),
                        "pcm_base64": base64.b64encode(b).decode("ascii"),
                    }
                )
                seq += 1
        except asyncio.CancelledError:
            pass
        finally:
            # Log TTS completion
            complete_info = RichLogger.tts_complete(seq, total_bytes)
            print(f"[{RichLogger._format_time()}] {session_info} {complete_info}")
            
            if not self._interrupted:
                await self._finalize_turn(flush_asr=False, reason="tts_complete")

    async def _on_barge_in(self, prime_frames: Optional[list[bytes]] = None):
        # if we're not currently speaking, just ensure we're in LISTEN
        if self.state != "SPEAK":
            self.state = "LISTEN"
            return

        await self.session.send_json({"type": MSG_CUT_PLAYBACK, "t": time.monotonic()})
        self._interrupted = True

        if self._tts_task and not self._tts_task.done():
            self._tts_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._tts_task

        # immediately allow ASR again
        self._mute_asr_until = 0.0
        self.state = "LISTEN"

        # start a fresh turn and reset VAD/ASR counters
        await self._reset_for_next_utterance(new_turn=True)

        # prime ASR with frames buffered during SPEAK so early syllables aren't lost
        if prime_frames:
            for f in prime_frames:
                await self.asr.feed(f)

        self._pre_barge_frames.clear()

    async def _finalize_turn(self, flush_asr: bool, reason: str):
        if flush_asr:
            evt = await self.asr.flush()
            if evt and (evt.text or "").strip():
                if not _looks_like_junk_final(evt.text):
                    if self._t_asr_first_event == 0.0:
                        self._t_asr_first_event = evt.t
                    self._t_asr_final = evt.t
                    await self.session.send_json({"type": MSG_FINAL, "text": evt.text, "t": evt.t})

        t_end = time.time()
        
        # ASR timing: from turn start to first ASR event
        asr_ms = (
            int(max(0, (self._t_asr_first_event - self._t_turn_start) * 1000))
            if self._t_asr_first_event
            else 0
        )
        
        # Planning timing: from ASR final to planning completion
        plan_ms = int(max(0, (self._t_plan_done - self._t_asr_final) * 1000)) if self._t_plan_done and self._t_asr_final else 0
        
        # TTS timing: from planning completion to first TTS chunk
        tts_ms = (
            int(max(0, (self._t_tts_first - self._t_plan_done) * 1000))
            if self._t_tts_first and self._t_plan_done
            else 0
        )
        
        # Total response time: from turn start to end
        rt_ms = int(max(0, (t_end - self._t_turn_start) * 1000)) if self._t_turn_start else 0
        
        # Calculate detailed gap analysis
        gap_asr_to_plan = (self._t_asr_final_to_plan_start - self._t_asr_final) * 1000 if self._t_asr_final > 0 and self._t_asr_final_to_plan_start > 0 else 0
        gap_plan_to_tts = (self._t_plan_done_to_tts_start - self._t_plan_done) * 1000 if self._t_plan_done > 0 and self._t_plan_done_to_tts_start > 0 else 0
        # TTS start to first chunk is not a gap - it's the actual TTS processing time
        # The real gap would be between when we want to start TTS and when we actually start it
        gap_tts_start_to_first = 0  # This is not a gap, it's TTS processing time
        
        # Calculate pipeline gaps (unaccounted time)
        accounted_time = asr_ms + plan_ms + tts_ms
        gap_ms = max(0, rt_ms - accounted_time)

        # Log turn completion with rich timing breakdown
        session_info = RichLogger.session_info(self.session.id, self.turn_id, self.state)
        print(f"[{RichLogger._format_time()}] {session_info} {RichLogger.turn_complete(reason)}")
        print(f"[{RichLogger._format_time()}] {session_info} {RichLogger.turn_summary(rt_ms, asr_ms, plan_ms, tts_ms)}")
        
        # Log detailed gap analysis
        if gap_ms > 100:  # Only log significant gaps
            gap_info = RichLogger.timing(f"Pipeline Gap", gap_ms)
            print(f"[{RichLogger._format_time()}] {session_info} {gap_info}")
            
            # Log individual gap components
            if gap_asr_to_plan > 50:
                gap_info = RichLogger.timing(f"  └─ ASR→Plan", gap_asr_to_plan)
                print(f"[{RichLogger._format_time()}] {session_info} {gap_info}")
            if gap_plan_to_tts > 50:
                gap_info = RichLogger.timing(f"  └─ Plan→TTS", gap_plan_to_tts)
                print(f"[{RichLogger._format_time()}] {session_info} {gap_info}")
            # Note: TTS Start→First is not a gap, it's TTS processing time
        
        if self._interrupted:
            print(f"[{RichLogger._format_time()}] {session_info} {RichLogger.interrupted()}")

        self._logger.write(
            {
                "t": t_end,
                "evt": "turn_metrics",
                "sid": self.session.id,
                "turn": self.turn_id,
                "reason": reason,
                "rt_ms": rt_ms,
                "asr_ms": asr_ms,
                "plan_ms": plan_ms,
                "tts_ms": tts_ms,
                "gap_ms": gap_ms,
                "gap_asr_to_plan_ms": gap_asr_to_plan,
                "gap_plan_to_tts_ms": gap_plan_to_tts,
                "gap_tts_start_to_first_ms": gap_tts_start_to_first,
                "interrupted": self._interrupted,
            }
        )
        await self._reset_for_next_utterance(new_turn=False)
