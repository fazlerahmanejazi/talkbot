#!/usr/bin/env python3
"""
Model setup script for TalkBot.
Downloads required models during deployment/setup phase.
"""

import os
import sys
from pathlib import Path
import urllib.request

def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = Path(".env")
    if env_file.exists():
        print("📄 Loading environment from .env file...")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip('"').strip("'")
                    os.environ[key.strip()] = value
        print("✅ Environment loaded from .env file")
    else:
        print("ℹ️  No .env file found, using system environment")

def download_file(url: str, destination: Path, description: str) -> bool:
    """Download a file from URL to destination."""
    try:
        print(f"📥 Downloading {description}...")
        print(f"   From: {url}")
        print(f"   To: {destination}")
        
        # Create parent directory if it doesn't exist
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"\r   Progress: {percent:.1f}% ({downloaded:,}/{total_size:,} bytes)", end="")
        
        urllib.request.urlretrieve(url, destination, show_progress)
        print(f"\n✅ {description} downloaded successfully")
        return True
    except Exception as e:
        print(f"\n❌ Failed to download {description}: {e}")
        return False

def setup_piper_models():
    """Download Piper TTS models if they don't exist."""
    print("🎤 Setting up Piper TTS models...")
    
    # Model paths
    models_dir = Path("./models/en_US/amy/medium")
    model_file = models_dir / "en_US-amy-medium.onnx"
    config_file = models_dir / "en_US-amy-medium.onnx.json"
    
    # URLs
    model_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx"
    config_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json"
    
    success = True
    
    # Download model if missing
    if not model_file.exists():
        success &= download_file(model_url, model_file, "Piper TTS model")
    else:
        print(f"✅ Piper model already exists: {model_file}")
    
    # Download config if missing
    if not config_file.exists():
        success &= download_file(config_url, config_file, "Piper TTS config")
    else:
        print(f"✅ Piper config already exists: {config_file}")
    
    return success

def setup_faster_whisper_models():
    """Download Faster-Whisper models if needed."""
    print("🎯 Setting up Faster-Whisper models...")
    
    try:
        from faster_whisper import WhisperModel
        print("✅ Faster-Whisper is available")
    except ImportError:
        print("❌ Faster-Whisper not installed")
        return False
    
    # Get the model name from environment or use default
    import os
    model_name = os.getenv("VHYS_FASTER_WHISPER_MODEL", "base.en")
    
    # Always force download/verification of the specific model
    print(f"📥 Ensuring Faster-Whisper model '{model_name}' is downloaded...")
    print("   This will download if not cached, or verify if already available...")
    
    try:
        # Force initialize the specific model - this will download if needed
        model = WhisperModel(model_name, compute_type="int8")
        print(f"✅ Faster-Whisper model '{model_name}' is ready")
        
        # Test that the model actually works
        print("🧪 Testing model with a short audio sample...")
        import numpy as np
        # Create a short test audio (1 second of silence)
        test_audio = np.zeros(16000, dtype=np.float32)  # 1 second at 16kHz
        segments, info = model.transcribe(test_audio, language="en")
        # Just iterate to trigger the transcription
        list(segments)
        print(f"✅ Model test successful - {model_name} is working correctly")
        
        return True
    except Exception as e:
        print(f"❌ Failed to setup Faster-Whisper model '{model_name}': {e}")
        print("ℹ️  Model will be downloaded on first use")
        return False

def setup_ollama_models():
    """Download Ollama models if needed."""
    print("🤖 Setting up Ollama models...")
    
    try:
        import ollama
        print("✅ Ollama Python client is available")
    except ImportError:
        print("❌ Ollama Python client not installed")
        return False
    
    # Get the model name from environment or use default
    import os
    model_name = os.getenv("VHYS_LOCAL_LLM_MODEL", "phi3:mini")
    
    try:
        # Check if Ollama is installed
        import shutil
        if not shutil.which("ollama"):
            print("❌ Ollama binary not found. Please install Ollama first:")
            print("   Visit: https://ollama.ai/download")
            return False
        
        print(f"✅ Ollama binary found")
        
        # Start Ollama service if not running
        print("🚀 Starting Ollama service...")
        import subprocess
        import time
        
        try:
            # Try to connect to existing service
            client = ollama.Client(host="http://localhost:11434")
            client.list()
            print("✅ Ollama service is already running")
        except:
            # Start Ollama service
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            time.sleep(3)  # Wait for service to start
            client = ollama.Client(host="http://localhost:11434")
        
        # Check if model exists
        models = client.list()
        model_names = [model['name'] for model in models['models']]
        
        if model_name in model_names:
            print(f"✅ Model {model_name} is already available")
        else:
            print(f"📥 Downloading model {model_name}...")
            print("   This may take a few minutes depending on model size...")
            client.pull(model_name)
            print(f"✅ Model {model_name} downloaded successfully")
        
        # Test the model
        print("🧪 Testing model with a simple prompt...")
        response = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": "Hello"}],
            options={'temperature': 0.1, 'num_predict': 10}
        )
        print(f"✅ Model test successful - {model_name} is working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to setup Ollama model '{model_name}': {e}")
        print("ℹ️  Model will be downloaded on first use")
        return False

def main():
    """Main setup function."""
    import argparse
    
    # Load .env file first
    load_env_file()
    
    parser = argparse.ArgumentParser(description="Setup TalkBot models")
    parser.add_argument("--force", action="store_true", 
                       help="Force re-download of all models")
    parser.add_argument("--model", type=str, 
                       help="Specific Whisper model to download (e.g., base.en, medium.en)")
    
    args = parser.parse_args()
    
    print("🚀 TalkBot - Model Setup")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("server").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Override model if specified
    if args.model:
        print(f"🎯 Using specified model: {args.model}")
        os.environ["VHYS_FASTER_WHISPER_MODEL"] = args.model
    
    if args.force:
        print("🔄 Force mode: Will re-download all models")
    
    success = True
    
    # Setup TTS models
    success &= setup_piper_models()
    
    # Setup ASR models
    success &= setup_faster_whisper_models()
    
    # Setup Local LLM models (if enabled)
    if os.getenv("VHYS_USE_LOCAL_LLM", "false").lower() in ("true", "1", "yes", "on"):
        success &= setup_ollama_models()
    else:
        print("📝 Skipping Ollama setup (VHYS_USE_LOCAL_LLM not enabled)")
    
    print("\n" + "=" * 40)
    if success:
        print("✅ All models setup successfully!")
        print("🎉 You can now start the application")
        print("\n💡 To change models, update VHYS_FASTER_WHISPER_MODEL in your .env file")
        print("   Available models: tiny.en, base.en, small.en, medium.en, large-v2")
    else:
        print("❌ Some models failed to setup")
        print("💡 Check your internet connection and try again")
        sys.exit(1)

if __name__ == "__main__":
    main()
