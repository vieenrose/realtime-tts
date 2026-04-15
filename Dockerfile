FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

LABEL maintainer="MOSS-TTS-Realtime Service"
LABEL description="ONNX-based streaming TTS for LiveKit voice agents"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3-pip \
    python3.12-venv \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Create app directory
WORKDIR /app

# Copy inference code (model files mounted at runtime)
COPY tts_service.py /app/tts_service.py

# Environment variables
ENV MODEL_DIR=/app/models/MOSS-TTS-Realtime-ONNX
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CUDA_VISIBLE_DEVICES=0

# Expose service port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the TTS service
CMD ["uvicorn", "tts_service:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "300"]