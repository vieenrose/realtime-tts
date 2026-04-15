FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

LABEL maintainer="MOSS-TTS-Realtime Service"
LABEL description="ONNX-based streaming TTS for LiveKit voice agents (DGX Spark ARM64 CUDA)"

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
# Use Microsoft nightly builds for ARM64 CUDA (Grace Hopper / DGX Spark)
RUN pip install --no-cache-dir numpy soundfile librosa tokenizers fastapi uvicorn[standard] websockets \
    nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12

# Install onnxruntime-gpu from Microsoft nightly for ARM64 CUDA 13 (use PyPI for dependencies)
RUN pip install --pre --no-cache-dir --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ --extra-index-url https://pypi.org/simple/ onnxruntime-gpu

# Create app directory
WORKDIR /app

# Copy inference code (model files mounted at runtime)
COPY tts_service.py /app/tts_service.py

# Environment variables
ENV MODEL_DIR=/app/models/MOSS-TTS-Realtime-ONNX
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CUDA_VISIBLE_DEVICES=0
ENV LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH"

# Expose service port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the TTS service
CMD ["uvicorn", "tts_service:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "300"]