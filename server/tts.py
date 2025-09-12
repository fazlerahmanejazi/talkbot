from __future__ import annotations
import asyncio, math, struct, contextlib, os, json, urllib.request
from dataclasses import dataclass
from typing import AsyncGenerator, Optional, Literal
from pathlib import Path
import numpy as np
from .settings import settings

TTSEventType = Literal["speak_start", "audio"]

@dataclass
class TTSConfig:
    out_sample_rate: int = 16000
    frame_ms: int = 20
    use_piper: bool = settings.use_piper
    piper_bin: str = settings.piper_bin
    piper_model_path: str = settings.piper_model_path
    piper_config_path: str = settings.piper_config_path
    
    # Process pooling optimization settings
    pool_size: int = 3  # Number of pre-warmed processes
    warmup_text: str = "Hello"  # Text to warm up processes
    enable_caching: bool = True  # Cache common responses
    cache_size: int = 100  # Max cached responses
    
    # Streaming optimization settings
    chunk_size: int = 1024  # Smaller chunks for faster streaming
    buffer_size: int = 4096  # Buffer size for reading
    enable_streaming: bool = True  # Enable streaming optimizations

@dataclass
class TTSAudioChunk:
    seq: int
    pcm_bytes: bytes

def _int16_clip(x: np.ndarray) -> np.ndarray:
    y = np.clip(x, -1.0, 1.0)
    return (y * 32767.0).astype("<i2")

def _linear_resample_f32(x: np.ndarray, src_hz: int, dst_hz: int) -> np.ndarray:
    if src_hz == dst_hz or x.size == 0:
        return x
    ratio = dst_hz / float(src_hz)
    n_out = int(math.floor(x.size * ratio))
    xp = np.arange(x.size, dtype=np.float64)
    fp = x.astype(np.float32)
    new_pos = np.linspace(0, x.size - 1, num=n_out, dtype=np.float64)
    out = np.interp(new_pos, xp, fp).astype(np.float32)
    return out

def _download_piper_model(model_path: str, config_path: str) -> bool:
    """Download Piper model and config files if they don't exist."""
    model_file = Path(model_path)
    config_file = Path(config_path)
    
    if model_file.exists() and config_file.exists():
        return True
    
    print(f"[TTS] 📥 Piper model files not found, downloading...")
    model_file.parent.mkdir(parents=True, exist_ok=True)
    
    model_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx"
    config_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json"
    
    try:
        if not model_file.exists():
            print(f"[TTS] 📥 Downloading model from {model_url}")
            urllib.request.urlretrieve(model_url, model_file)
        
        if not config_file.exists():
            print(f"[TTS] 📥 Downloading config from {config_url}")
            urllib.request.urlretrieve(config_url, config_file)
        
        return True
    except Exception as e:
        print(f"[TTS] ❌ Failed to download Piper model: {e}")
        return False

class _PiperProcess:
    """A single Piper process that can be reused."""
    
    def __init__(self, cfg: TTSConfig):
        self.cfg = cfg
        self.proc: Optional[asyncio.subprocess.Process] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._is_warmed_up = False
        self._last_used = 0.0
        
    async def _drain_stderr(self):
        try:
            if not self.proc or not self.proc.stderr:
                return
            err = await self.proc.stderr.read()
            if err:
                print("[Piper stderr]", err.decode(errors="ignore")[:4000])
        except Exception:
            pass

    async def start(self):
        """Start the Piper process."""
        try:
            args = [
                self.cfg.piper_bin,
                "--model", self.cfg.piper_model_path,
                "--config", self.cfg.piper_config_path,
                "--output_file", "-",
            ]
            self.proc = await asyncio.create_subprocess_exec(
                *args, stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            self._stderr_task = asyncio.create_task(self._drain_stderr())
        except Exception as e:
            raise RuntimeError(f"Piper launch failed: {e!r}")

    async def warmup(self):
        """Warm up the process with a test synthesis."""
        if self._is_warmed_up:
            return
        
        print(f"[TTS] 🔥 Warming up Piper process...")
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Synthesize warmup text
            async for _ in self.synth(self.cfg.warmup_text):
                pass  # Consume all chunks to complete warmup
            
            warmup_time = (asyncio.get_event_loop().time() - start_time) * 1000
            print(f"[TTS] ✅ Process warmed up in {warmup_time:.1f}ms")
            self._is_warmed_up = True
        except Exception as e:
            print(f"[TTS] ❌ Warmup failed: {e}")

    async def synth(self, text: str) -> AsyncGenerator[TTSAudioChunk, None]:
        """Synthesize text using this process."""
        if not self.proc or self.proc.returncode is not None:
            await self.start()
            await self.warmup()
        
        self._last_used = asyncio.get_event_loop().time()
        
        assert self.proc and self.proc.stdin and self.proc.stdout
        
        try:
            self.proc.stdin.write(text.encode("utf-8") + b"\n")
            await self.proc.stdin.drain()
        finally:
            with contextlib.suppress(Exception):
                self.proc.stdin.close()

        reader = _WavStreamReader(self.proc.stdout)
        fmt = await reader.read_header()
        src_rate = fmt.sample_rate
        if fmt.channels != 1 or fmt.bits_per_sample != 16:
            raise RuntimeError(f"Piper voice must be mono/16-bit (got {fmt.channels}ch/{fmt.bits_per_sample}bit)")

        samples_per_frame = int(self.cfg.out_sample_rate * self.cfg.frame_ms / 1000)
        buf_f32 = np.zeros(0, dtype=np.float32)
        seq = 0

        try:
            # Use streaming-optimized chunk size if enabled
            chunk_size = self.cfg.chunk_size if self.cfg.enable_streaming else 4096
            async for pcm_bytes in reader.iter_data_bytes(chunk_size):
                i16 = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32)
                f32 = i16 / 32768.0
                if src_rate != self.cfg.out_sample_rate:
                    f32 = _linear_resample_f32(f32, src_rate, self.cfg.out_sample_rate)
                if buf_f32.size == 0:
                    buf_f32 = f32
                else:
                    buf_f32 = np.concatenate([buf_f32, f32])

                while buf_f32.size >= samples_per_frame:
                    frame = buf_f32[:samples_per_frame]
                    buf_f32 = buf_f32[samples_per_frame:]
                    out = _int16_clip(frame).tobytes()
                    yield TTSAudioChunk(seq=seq, pcm_bytes=out)
                    seq += 1

            if buf_f32.size > 0:
                yield TTSAudioChunk(seq=seq, pcm_bytes=_int16_clip(buf_f32).tobytes())
        finally:
            # Don't close the process - keep it alive for reuse
            pass

    async def close(self):
        """Close the process."""
        try:
            if self.proc and self.proc.returncode is None:
                self.proc.kill()
        except Exception:
            pass
        if self._stderr_task and not self._stderr_task.done():
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._stderr_task, timeout=0.2)

class _PiperProcessPool:
    """Pool of pre-warmed Piper processes for fast TTS."""
    
    def __init__(self, cfg: TTSConfig):
        self.cfg = cfg
        self.processes: list[_PiperProcess] = []
        self._available: asyncio.Queue = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._initialized = False
        
    async def initialize(self):
        """Initialize the process pool."""
        if self._initialized:
            return
            
        print(f"[TTS] 🏊 Initializing process pool (size: {self.cfg.pool_size})")
        start_time = asyncio.get_event_loop().time()
        
        # Create and warm up processes
        for i in range(self.cfg.pool_size):
            process = _PiperProcess(self.cfg)
            await process.start()
            await process.warmup()
            self.processes.append(process)
            await self._available.put(process)
        
        init_time = (asyncio.get_event_loop().time() - start_time) * 1000
        print(f"[TTS] ✅ Process pool initialized in {init_time:.1f}ms")
        self._initialized = True

    async def get_process(self) -> _PiperProcess:
        """Get an available process from the pool."""
        # Try to get an available process
        try:
            process = await asyncio.wait_for(self._available.get(), timeout=1.0)
            return process
        except asyncio.TimeoutError:
            # If no process available, create a temporary one
            print(f"[TTS] ⚠️  No available processes, creating temporary one")
            process = _PiperProcess(self.cfg)
            await process.start()
            await process.warmup()
            return process

    async def return_process(self, process: _PiperProcess):
        """Return a process to the pool."""
        if process in self.processes:
            await self._available.put(process)
        else:
            # Temporary process - close it
            await process.close()

    async def close_all(self):
        """Close all processes in the pool."""
        for process in self.processes:
            await process.close()
        self.processes.clear()

class _TTSCache:
    """Simple LRU cache for TTS responses."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache: dict[str, list[TTSAudioChunk]] = {}
        self.access_order: list[str] = []
    
    def get(self, text: str) -> Optional[list[TTSAudioChunk]]:
        """Get cached audio chunks for text."""
        if text in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(text)
            self.access_order.append(text)
            return self.cache[text]
        return None
    
    def put(self, text: str, chunks: list[TTSAudioChunk]):
        """Cache audio chunks for text."""
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_text = self.access_order.pop(0)
            del self.cache[lru_text]
        
        self.cache[text] = chunks
        self.access_order.append(text)

@dataclass
class _WavFmt:
    audio_format: int
    channels: int
    sample_rate: int
    byte_rate: int
    block_align: int
    bits_per_sample: int

class _WavStreamReader:
    def __init__(self, stdout: asyncio.StreamReader):
        self.r = stdout
        self._data_remaining: Optional[int] = None

    async def _read_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = await self.r.read(n - len(buf))
            if not chunk:
                raise EOFError("Unexpected EOF while reading WAV header")
            buf += chunk
        return buf

    async def read_header(self) -> _WavFmt:
        hdr = await self._read_exact(12)
        if hdr[:4] != b"RIFF" or hdr[8:12] != b"WAVE":
            raise RuntimeError("Not a RIFF/WAVE stream")
        fmt: Optional[_WavFmt] = None
        while True:
            chunk_hdr = await self._read_exact(8)
            cid = chunk_hdr[:4]
            csize = struct.unpack("<I", chunk_hdr[4:8])[0]
            if cid == b"fmt ":
                data = await self._read_exact(csize)
                if csize < 16:
                    raise RuntimeError("Invalid fmt chunk")
                (audio_format, channels, sample_rate, byte_rate, block_align, bits_per_sample) = struct.unpack("<HHIIHH", data[:16])
                fmt = _WavFmt(audio_format, channels, sample_rate, byte_rate, block_align, bits_per_sample)
                extra = csize - 16
                if extra > 0: _ = await self._read_exact(extra)
            elif cid == b"data":
                if fmt is None: raise RuntimeError("WAV 'data' before 'fmt'")
                self._data_remaining = csize
                return fmt
            else:
                _ = await self._read_exact(csize)

    async def iter_data_bytes(self, chunk_size: int = 4096):
        if self._data_remaining is None:
            raise RuntimeError("Must call read_header() first")
        remaining = self._data_remaining
        while remaining > 0:
            to_read = min(chunk_size, remaining)
            data = await self.r.read(to_read)
            if not data: break
            remaining -= len(data)
            yield data

class OptimizedStreamingTTS:
    """Optimized TTS with process pooling and caching."""
    
    def __init__(self, cfg: Optional[TTSConfig] = None):
        self.cfg = cfg or TTSConfig()
        
        # Ensure model files exist
        if not _download_piper_model(self.cfg.piper_model_path, self.cfg.piper_config_path):
            raise RuntimeError("Failed to download or locate Piper model files")
        
        # Initialize process pool and cache
        self.pool = _PiperProcessPool(self.cfg)
        self.cache = _TTSCache(self.cfg.cache_size) if self.cfg.enable_caching else None
        
        print(f"[TTS] 🚀 Optimized TTS initialized")
        print(f"[TTS] ⚙️  Pool size: {self.cfg.pool_size}, Caching: {self.cfg.enable_caching}")
        print(f"[TTS] 🌊 Streaming: {self.cfg.enable_streaming}, Chunk size: {self.cfg.chunk_size}")
        
        # Initialize the pool immediately at startup
        self._pool_initialized = False
        # Note: Pool will be initialized on first synth() call to avoid blocking __init__
    
    async def initialize_pool(self):
        """Initialize the process pool. Call this after creating the TTS instance."""
        if not self._pool_initialized:
            await self.pool.initialize()
            self._pool_initialized = True

    async def synth(self, text: str) -> AsyncGenerator[TTSAudioChunk, None]:
        """Synthesize text with optimizations."""
        # Initialize pool on first use
        if not self._pool_initialized:
            await self.pool.initialize()
            self._pool_initialized = True
        
        # Check cache first
        if self.cache:
            cached_chunks = self.cache.get(text)
            if cached_chunks:
                print(f"[TTS] 💾 Cache hit for: '{text[:50]}...'")
                for chunk in cached_chunks:
                    yield chunk
                return
        
        # Get process from pool
        process = await self.pool.get_process()
        chunks = []
        
        try:
            async for chunk in process.synth(text):
                chunks.append(chunk)
                yield chunk
        finally:
            # Return process to pool
            await self.pool.return_process(process)
        
        # Cache the result
        if self.cache and chunks:
            self.cache.put(text, chunks)

    async def close(self):
        """Close the TTS system."""
        await self.pool.close_all()
