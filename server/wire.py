# Client → Server control
MSG_START = "start"
MSG_BARGE_IN = "barge_in"
MSG_STOP = "stop"
MSG_DEBUG_TTS = "debug_tts"

# Server → Client ASR
MSG_PARTIAL = "partial"  # Partial ASR results
MSG_STREAMING = "streaming"  # Streaming ASR results
MSG_FINAL = "final"

# Server → Client TTS/Playback
MSG_SPEAK_START = "speak_start"
MSG_AUDIO = "audio"
MSG_CUT_PLAYBACK = "cut_playback"

# Server → Client metrics
MSG_METRICS = "metrics"
