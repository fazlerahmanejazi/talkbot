"""
Rich logging utilities for the voice pipeline.
Provides structured, emoji-enhanced logging for better debugging and monitoring.
"""

import time
from typing import Any, Dict, Optional


class RichLogger:
    """Enhanced logging with emojis, colors, and structured output for voice pipeline flow."""
    
    @staticmethod
    def _format_time() -> str:
        return time.strftime("%H:%M:%S", time.localtime())
    
    @staticmethod
    def _format_duration(ms: float) -> str:
        if ms < 1000:
            return f"{ms:.0f}ms"
        return f"{ms/1000:.1f}s"
    
    @staticmethod
    def session_info(session_id: str, turn_id: int, state: str) -> str:
        return f"🎯 [{session_id[:8]}] T{turn_id} {state}"
    
    @staticmethod
    def audio_input(bytes_len: int, rms: float, vad_speech: bool) -> str:
        status = "🔊" if vad_speech else "🔇"
        return f"{status} Audio: {bytes_len}B RMS:{rms:.3f}"
    
    @staticmethod
    def asr_partial(text: str, latency_ms: float) -> str:
        return f"📝 Partial: '{text}' ({RichLogger._format_duration(latency_ms)})"
    
    @staticmethod
    def asr_streaming(text: str, latency_ms: float) -> str:
        return f"🌊 Streaming: '{text}' ({RichLogger._format_duration(latency_ms)})"
    
    @staticmethod
    def asr_final(text: str, latency_ms: float) -> str:
        return f"✅ Final: '{text}' ({RichLogger._format_duration(latency_ms)})"
    
    @staticmethod
    def planning_start(text: str) -> str:
        return f"🧠 Planning: '{text}'"
    
    @staticmethod
    def planning_tool(tool_name: str, args: dict) -> str:
        return f"🔧 Tool: {tool_name}({', '.join(f'{k}={v}' for k, v in args.items())})"
    
    @staticmethod
    def planning_response(text: str, channel: str, confidence: float) -> str:
        return f"💬 Response: '{text}' [{channel}] ({confidence:.2f})"
    
    @staticmethod
    def tts_start(text: str) -> str:
        return f"🗣️  TTS: '{text}'"
    
    @staticmethod
    def tts_chunk(seq: int, bytes_len: int) -> str:
        return f"🎵 Chunk {seq}: {bytes_len}B"
    
    @staticmethod
    def tts_complete(chunks: int, total_bytes: int) -> str:
        return f"✅ TTS Complete: {chunks} chunks, {total_bytes}B"
    
    @staticmethod
    def state_transition(old_state: str, new_state: str, reason: str = "") -> str:
        return f"🔄 {old_state} → {new_state}" + (f" ({reason})" if reason else "")
    
    @staticmethod
    def barge_in() -> str:
        return "⚡ Barge-in detected!"
    
    @staticmethod
    def error(error_msg: str) -> str:
        return f"❌ Error: {error_msg}"
    
    @staticmethod
    def timing(component: str, duration_ms: float) -> str:
        return f"⏱️  {component}: {RichLogger._format_duration(duration_ms)}"
    
    @staticmethod
    def session_start() -> str:
        return "🚀 Session Start"
    
    @staticmethod
    def session_stop() -> str:
        return "🛑 Session Stop"
    
    @staticmethod
    def turn_complete(reason: str) -> str:
        return f"🏁 Turn Complete: {reason}"
    
    @staticmethod
    def turn_summary(total_ms: float, asr_ms: float, plan_ms: float, tts_ms: float) -> str:
        return f"⏱️  Total: {RichLogger._format_duration(total_ms)} | ASR: {RichLogger._format_duration(asr_ms)} | Plan: {RichLogger._format_duration(plan_ms)} | TTS: {RichLogger._format_duration(tts_ms)}"
    
    @staticmethod
    def interrupted() -> str:
        return "⚡ Interrupted: Yes"


class NDJSONLogger:
    """Metrics logger for structured data output."""
    
    def __init__(self, path: str):
        from pathlib import Path
        import orjson
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = Path(path)
        if not self.path.exists():
            self.path.touch()
        self._orjson = orjson

    def write(self, event: dict[str, Any]) -> None:
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(self._orjson.dumps(event).decode("utf-8") + "\n")
        except Exception:
            pass
