from __future__ import annotations
import os
from dataclasses import dataclass

# Load .env if present
try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass

def _get_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")

@dataclass(frozen=True)
class Settings:
    host: str = os.getenv("VHYS_HOST", "0.0.0.0")
    port: int = int(os.getenv("VHYS_PORT", "8080"))

    sample_rate: int = int(os.getenv("VHYS_SAMPLE_RATE", "16000"))
    channels: int = int(os.getenv("VHYS_CHANNELS", "1"))
    frame_ms: int = int(os.getenv("VHYS_FRAME_MS", "20"))
    pcm_width: int = int(os.getenv("VHYS_PCM_WIDTH", "2"))

    @property
    def samples_per_frame(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000)

    @property
    def bytes_per_frame(self) -> int:
        return self.samples_per_frame * self.pcm_width

    # Feature flags
    use_piper: bool = _get_bool("VHYS_USE_PIPER", True)
    enable_llm_fallback: bool = _get_bool("VHYS_ENABLE_LLM_FALLBACK", False)

    # Models / paths
    faster_whisper_model: str = os.getenv("VHYS_FASTER_WHISPER_MODEL", "base.en")
    piper_bin: str = os.getenv("VHYS_PIPER_BIN", "piper")
    piper_model_path: str = os.getenv("VHYS_PIPER_MODEL_PATH", "./models/en_US/amy/medium/en_US-amy-medium.onnx")
    piper_config_path: str = os.getenv("VHYS_PIPER_CONFIG_PATH", "./models/en_US/amy/medium/en_US-amy-medium.onnx.json")

    # Metrics
    metrics_dir: str = os.getenv("VHYS_METRICS_DIR", "./metrics")
    metrics_file: str = os.getenv("VHYS_METRICS_FILE", "./metrics/latency.ndjson")

    # LLM
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    llm_model: str = os.getenv("VHYS_LLM_MODEL", "gpt-4o-mini")
    llm_log: bool = _get_bool("VHYS_LLM_LOG", True)   # <— turn on/off verbose LLM logs
    

settings = Settings()
