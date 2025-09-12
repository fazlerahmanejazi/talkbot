from __future__ import annotations

import asyncio
import orjson
import time
from typing import Any, Optional

import contextlib


from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from .settings import settings
from .wire import MSG_START, MSG_BARGE_IN, MSG_STOP
from .pipeline import VoicePipeline
from .metrics import summarize_file, read_turn_metrics
from .logging import RichLogger
import os

app = FastAPI(title="talkbot")

# Log server startup configuration
print("🚀 TalkBot Server Starting")
print(f"🎯 ASR: {settings.faster_whisper_model} | 🎤 TTS: {'Piper' if settings.use_piper else 'Disabled'} | 🤖 LLM: {settings.llm_model}")
print(f"📊 Audio: {settings.sample_rate}Hz @ {settings.frame_ms}ms | 🌐 {settings.host}:{settings.port}")

# Check optimization features
enable_partials = os.getenv("VHYS_ENABLE_PARTIALS", "false").lower() in ("true", "1", "yes", "on")
enable_parallel = os.getenv("VHYS_ENABLE_PARALLEL_LLM_TTS", "false").lower() in ("true", "1", "yes", "on")
enable_streaming = os.getenv("VHYS_ENABLE_STREAMING_ASR", "false").lower() in ("true", "1", "yes", "on")
enable_full_parallel = os.getenv("VHYS_ENABLE_FULL_PARALLEL", "false").lower() in ("true", "1", "yes", "on")

optimizations = []
if enable_partials: optimizations.append("Partial ASR")
if enable_parallel: optimizations.append("Parallel LLM+TTS")
if enable_streaming: optimizations.append("Streaming ASR")
if enable_full_parallel: optimizations.append("Full Parallel")

if optimizations:
    print(f"⚡ Optimizations: {', '.join(optimizations)}")
else:
    print("⚡ Optimizations: None enabled")

print("=" * 60)


# ----------------------------
# Simple per-connection session wrapper (same as before, trimmed)
# ----------------------------
class Session:
    def __init__(self, ws: WebSocket, session_id: Optional[str] = None):
        import uuid
        self.ws = ws
        self.id = session_id or str(uuid.uuid4())
        self._send_lock = asyncio.Lock()

    async def accept(self):
        await self.ws.accept()

    async def close(self, code: int = 1000):
        with contextlib.suppress(Exception):
            await self.ws.close(code=code)

    async def send_json(self, obj: dict[str, Any]):
        # serialize and send atomically
        payload = orjson.dumps(obj).decode("utf-8")
        async with self._send_lock:
            await self.ws.send_text(payload)


# ----------------------------
# WebSocket endpoint
# ----------------------------
@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    session = Session(ws)
    await session.accept()

    pipeline = VoicePipeline(session)

    try:
        while True:
            msg = await ws.receive()
            t = msg["type"]
            if t == "websocket.receive":
                if "text" in msg and msg["text"] is not None:
                    # Control JSON
                    try:
                        obj = orjson.loads(msg["text"])
                    except Exception:
                        await session.send_json({"type": "partial", "text": "bad json", "t": time.time()})
                        continue
                    await pipeline.handle_control(obj)
                elif "bytes" in msg and msg["bytes"] is not None:
                    # 20ms PCM16 frame(s)
                    data: bytes = msg["bytes"]
                    # Some clients may batch multiples of a 20ms frame; split if needed
                    frame_bytes = settings.bytes_per_frame or 640
                    # If not a multiple, just feed the whole thing as one frame (robustness)
                    if frame_bytes > 0 and len(data) % frame_bytes == 0:
                        for i in range(0, len(data), frame_bytes):
                            await pipeline.handle_audio(data[i:i+frame_bytes])
                    else:
                        await pipeline.handle_audio(data)
            elif t == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        with contextlib.suppress(Exception):
            await session.send_json({"type": "partial", "text": f"server_error:{type(e).__name__}", "t": time.time()})
    finally:
        with contextlib.suppress(Exception):
            await pipeline.close()
        with contextlib.suppress(Exception):
            await session.close()


# ----------------------------
# Health & minimal metrics view
# ----------------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/metrics")
def metrics_summary():
    return JSONResponse(summarize_file())

@app.get("/metrics/turns")
def get_turns():
    """Get all individual turn metrics."""
    turns = read_turn_metrics(settings.metrics_file)
    return JSONResponse(turns)

@app.delete("/metrics")
def reset_metrics():
    """Clear all metrics data by truncating the metrics file."""
    try:
        metrics_path = settings.metrics_file
        if os.path.exists(metrics_path):
            with open(metrics_path, 'w') as f:
                f.truncate(0)
        return {"message": "Metrics cleared successfully"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to clear metrics: {str(e)}"}
        )

