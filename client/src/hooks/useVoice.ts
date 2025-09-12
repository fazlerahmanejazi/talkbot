import { useEffect, useRef, useState } from "react";

/** DSP helpers */
function floatToPCM16(input: Float32Array): Int16Array {
  const out = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}
function resampleLinear(input: Float32Array, srcRate: number, dstRate: number): Float32Array {
  if (srcRate === dstRate) return input;
  const ratio = srcRate / dstRate;
  const outLen = Math.floor(input.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const idx = i * ratio;
    const i0 = Math.floor(idx);
    const frac = idx - i0;
    const s0 = input[i0] ?? 0;
    const s1 = input[i0 + 1] ?? s0;
    out[i] = s0 + (s1 - s0) * frac;
  }
  return out;
}
function* frame20ms(samples16k: Float32Array): Generator<Int16Array> {
  const SAMPLES_PER_FRAME = 320; // 20ms @ 16k
  for (let i = 0; i + SAMPLES_PER_FRAME <= samples16k.length; i += SAMPLES_PER_FRAME) {
    yield floatToPCM16(samples16k.subarray(i, i + SAMPLES_PER_FRAME));
  }
}

/** Streaming audio player */
class StreamingPlayer {
  private ctx: AudioContext;
  private node: ScriptProcessorNode;
  private q: Float32Array[] = [];
  private gain = 1.8; // small boost

  constructor() {
    this.ctx = new (window.AudioContext || (window as any).webkitAudioContext)({ latencyHint: "interactive" });
    this.node = this.ctx.createScriptProcessor(2048, 1, 1);
    this.node.onaudioprocess = (e) => {
      const out = e.outputBuffer.getChannelData(0);
      let o = 0;
      while (o < out.length) {
        if (!this.q.length) { out.fill(0, o); break; }
        const chunk = this.q[0];
        const n = Math.min(chunk.length, out.length - o);
        out.set(chunk.subarray(0, n), o);
        o += n;
        if (n < chunk.length) this.q[0] = chunk.subarray(n);
        else this.q.shift();
      }
    };
    this.node.connect(this.ctx.destination);
  }
  async resume() { if (this.ctx.state !== "running") { try { await this.ctx.resume(); } catch {} } }
  enqueuePCM16(bytes: Uint8Array, srcRate = 16000) {
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const n = dv.byteLength / 2;
    const pcm = new Int16Array(n);
    for (let i = 0; i < n; i++) pcm[i] = dv.getInt16(i * 2, true);
    const f32 = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let v = (pcm[i] / 0x8000) * this.gain;
      if (v > 1) v = 1; if (v < -1) v = -1;
      f32[i] = v;
    }
    const up = resampleLinear(f32, srcRate, this.ctx.sampleRate);
    this.q.push(up);
  }
  cut() { this.q = []; }
  secondsQueued() { return this.q.reduce((a, x) => a + x.length, 0) / this.ctx.sampleRate; }
  async close() { try { this.node.disconnect(); await this.ctx.close(); } catch {} }
}

export interface TranscriptLine { type: "partial" | "final"; text: string; t: number; }

export function useVoice() {
  const wsRef = useRef<WebSocket | null>(null);
  const wsOpenRef = useRef(false);
  const playerRef = useRef<StreamingPlayer | null>(null);

  const micCtxRef = useRef<AudioContext | null>(null);
  const micProcRef = useRef<ScriptProcessorNode | null>(null);
  const micSrcRef = useRef<MediaStreamAudioSourceNode | null>(null);

  const [connected, setConnected] = useState(false);
  const [transcript, setTranscript] = useState<TranscriptLine[]>([]);
  const [speaking, setSpeaking] = useState(false);
  const [framesSent, setFramesSent] = useState(0);
  const framesSentRef = useRef(0);
  const [queuedSec, setQueuedSec] = useState(0);

  useEffect(() => {
    playerRef.current = new StreamingPlayer();
    const id = setInterval(() => setQueuedSec(playerRef.current?.secondsQueued() || 0), 100);
    return () => { clearInterval(id); playerRef.current?.close(); };
  }, []);

  function ensureWs() {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return;
    const ws = new WebSocket("/ws/audio"); // proxied by Vite to FastAPI
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      wsOpenRef.current = true; setConnected(true);
      ws.send(JSON.stringify({ type: "start", sample_rate: 16000 }));
    };
    ws.onclose = () => { wsOpenRef.current = false; setConnected(false); };
    ws.onmessage = async (ev) => {
      if (typeof ev.data !== "string") return;
      const msg = JSON.parse(ev.data);
      if (msg.type === "partial") {
        // Handle partial results
        setTranscript(t => [...t, { type: msg.type, text: msg.text || "", t: msg.t }]);
      } else if (msg.type === "final") {
        setTranscript(t => [...t, { type: msg.type, text: msg.text || "", t: msg.t }]);
      } else if (msg.type === "speak_start") {
        await playerRef.current?.resume();
        setSpeaking(true);
      } else if (msg.type === "audio") {
        const b = atob(msg.pcm_base64 as string);
        console.log("audio chunk", msg.seq, "bytes", b.length);
        const bytes = new Uint8Array(b.length);
        for (let i = 0; i < b.length; i++) bytes[i] = b.charCodeAt(i);
        playerRef.current?.enqueuePCM16(bytes, 16000);
      } else if (msg.type === "cut_playback") {
        playerRef.current?.cut();
        setSpeaking(false);
      }
    };
    wsRef.current = ws;
  }

  async function startMic() {
    ensureWs();
    await playerRef.current?.resume();

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      video: false
    });
    const ctx = new (window.AudioContext || (window as any).webkitAudioContext)({ latencyHint: "interactive" });
    const src = ctx.createMediaStreamSource(stream);
    const BUFFER_SIZE = 2048;
    const proc = ctx.createScriptProcessor(BUFFER_SIZE, 1, 1);

    proc.onaudioprocess = (e) => {
      const inRate = ctx.sampleRate;
      const f32 = e.inputBuffer.getChannelData(0);
      const d16k = resampleLinear(f32, inRate, 16000);
      for (const pcm16 of frame20ms(d16k)) {
        if (wsOpenRef.current) wsRef.current?.send(pcm16.buffer);
        framesSentRef.current += 1;
      }
      setFramesSent(framesSentRef.current);
    };

    src.connect(proc); proc.connect(ctx.destination);
    micCtxRef.current = ctx; micSrcRef.current = src; micProcRef.current = proc;
  }

  async function bargeIn() {
    if (wsOpenRef.current) wsRef.current?.send(JSON.stringify({ type: "barge_in", ts: performance.now() / 1000 }));
    playerRef.current?.cut();
  }

  async function stopMic() {
    try { if (wsOpenRef.current) wsRef.current?.send(JSON.stringify({ type: "stop" })); } catch {}
    try { micProcRef.current?.disconnect(); } catch {}
    try { micSrcRef.current?.disconnect(); } catch {}
    try { await micCtxRef.current?.close(); } catch {}
    micProcRef.current = null; micSrcRef.current = null; micCtxRef.current = null;
    try { wsRef.current?.close(); } catch {}
    wsOpenRef.current = false; setConnected(false);
  }

  function debugTts(text = "Hello from Piper") {
    if (!wsOpenRef.current) ensureWs();
    const id = setInterval(() => {
      if (wsOpenRef.current) {
        wsRef.current?.send(JSON.stringify({ type: "debug_tts", text }));
        clearInterval(id);
      }
    }, 50);
  }

  useEffect(() => { (window as any).__vhys_debug_tts = debugTts; }, []);

  return { connected, transcript, speaking, framesSent, queuedSec, startMic, bargeIn, stopMic, debugTts };
}
