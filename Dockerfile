# Multi-stage build for talkbot
FROM python:3.11-slim as python-base

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Piper TTS binary
RUN curl -L https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz | tar -xz -C /usr/local/bin/ && \
    chmod +x /usr/local/bin/piper

# Set working directory
WORKDIR /app

# Copy Python requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY server/ ./server/
COPY models/ ./models/
COPY setup_models.py ./

# Create directory for metrics
RUN mkdir -p /app/metrics

# Setup models during build (download if not present)
RUN python setup_models.py

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8080"]
