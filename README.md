# talkbot

Low-latency **voice loop**: `LISTEN → PLAN → SPEAK` with **hard barge-in**.  
Built to feel natural and conversational — no long pauses, no “robotic” delays.  

---

## 🚀 Super Quick Start

One command to get everything running:

```bash
./deploy.sh
```

---

## 🛠️ Manual Setup (if needed)

### Backend (Python)
```bash
# Configure
cp .env.sample .env

# Install dependencies & set up env
make setup

# Run backend server
make run  # starts on http://localhost:8080
```

### Frontend (Client)
```bash
cd client
npm install
npm run dev  # opens http://localhost:5173
```

---

## 💻 Environment Notes

This prototype was developed and benchmarked primarily on a **MacBook Air M2 (8GB)**.  

- **ASR** → [Whisper tiny.en](https://github.com/openai/whisper) (int8, via Faster-Whisper). Runs locally with real-time streaming.  
- **TTS** → [Piper](https://github.com/rhasspy/piper) with **Amy (medium)** voice. Optimized with process pooling + pre-warming.  
- **LLM** → OpenAI **GPT-4** and **GPT-4o-mini** via API for planning.  
  - Codebase also supports **local LLMs** 
- Latency results (~3–4s RT) were achieved **without GPU acceleration**.  

**Takeaway:** This validates feasibility on lightweight consumer hardware, while leaving clear headroom for further gains on CPU-rich (ASR) or GPU-rich (LLM) cloud environments.


## ⚡ Why This Is Hard

Building **low-latency voice AI** isn’t just “ASR + LLM + TTS.”  
The real challenge is hitting **sub-second responses** that *feel instant* to a human ear.  

Started at **10–12s round trips** — frustrating and unusable.  
Step by step, brought it down to **3–4s**, while keeping rollback safety and production stability.  

---

## 🔑 Key Challenges & Breakthroughs

### 1. **ASR Latency**
- **Problem:** Endpointing added ~500ms of silence before recognition.  
- **Solution:** Tuned VAD thresholds; added **partial ASR events** every 200–500ms.  
- **Impact:** ~40% faster ASR; user sees words appear while speaking.

### 2. **Planning Delay (LLM)**
- **Problem:** LLM planning was 5–6s sequentially.  
- **Solution:** Ran **LLM planning + TTS warmup in parallel**; added timeout + fallback.  
- **Impact:** Down to ~1–2s planning time.

### 3. **TTS Bottleneck**
- **Problem:** TTS (Piper) startup overhead ~900–1000ms per utterance.  
- **Solution:** Added **process pooling + pre-warming**; cached common phrases.  
- **Impact:** First audio chunk in **100–200ms**.

### 4. **Barge-in (Full Duplex)**
- **Problem:** User interruptions weren’t handled.  
- **Solution:** Added **hard barge-in**: detect speech, cut playback instantly, feed ASR.  
- **Impact:** Conversations feel natural, never blocked.


---

## 📊 Performance Journey

| Metric                | Before (Baseline) | After (Optimized) |
|-----------------------|------------------|------------------|
| Total Response Time   | 9–10s            | 3–4s ✅           |
| Planning Time (LLM)   | 5–6s             | 1–2s ✅           |
| TTS First Chunk       | ~1s              | 100–200ms ✅      |
| ASR Response          | 84–171ms         | 50–100ms ✅       |

---

## 🔭 Next Goals

### 1. From Partials → **True Streaming ASR**
- Switch Whisper chunked mode to continuous streaming (200ms cadence).  
- Emit word-level timestamps for smarter barge-in.

### 2. **Better ASR Models on CPU-rich Cloud**
- Run Faster-Whisper int8/int4 on AWS/GCP CPU VMs.  
- Goal: **ASR-first-partial < 150ms** at p95.  

### 3. **Local LLM on GPU-rich Cloud**
- Run Llama 3 / Mistral 7B+ with vLLM or TGI on A10/A100.  
- Keep OpenAI fallback; enforce strict 12–20 word cap.  
- Goal: **Planner < 600ms**.  

### 4. **TTS Enhancements**
- Expand process pool; profile faster voices.  
- Add more caching for common acks.  

### 5. **Barge-in & Duplex Robustness**
- Adaptive RMS thresholds, echo-aware gating.  
- Maintain **cut-to-silence < 80ms**.  

---

## 🏆 Final Architecture

**Core Components**
1. Pipeline State Machine — `LISTEN → PLAN → SPEAK` with hard barge-in.  
2. Streaming/Chunked ASR — partials + promotion guard.  
3. Parallel Processing — planner + TTS prep overlap.  
4. Optimized TTS — process pool + caching.

