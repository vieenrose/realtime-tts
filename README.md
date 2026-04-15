# MOSS-TTS-Realtime ONNX Streaming Service

A FastAPI/WebSocket service for streaming text-to-speech synthesis using the MOSS-TTS-Realtime ONNX model, designed for integration with LiveKit voice agents.

## Overview

This project provides a pure ONNX Runtime inference pipeline for MOSS-TTS-Realtime, enabling **real-time streaming TTS** without PyTorch dependency at runtime. The service supports:

- **Streaming WebSocket API** - Receive text deltas and emit audio chunks in real-time
- **REST Batch API** - Synthesize complete text to WAV file
- **Voice Cloning** - Use reference audio to clone speaker voice
- **Multi-language** - Supports 20+ languages including zh-TW, en, ja, ko, de, es, fr, ru

## Architecture

```
Reference Audio ──► Codec Encoder ──► RVQ Audio Codes (voice clone context)
                                           │
                                           ▼
Text Deltas ──► Backbone LLM (Qwen3-1.7B) ──► Hidden States
                                                    │
                                                    ▼
                                            Local Transformer ──► 16-codebook Audio Tokens
                                                                        │
                                                                        ▼
                                                                Codec Decoder ──► 24 kHz Waveform
```

| Component | ONNX Model | Description |
|-----------|------------|-------------|
| Backbone LLM | `backbone_f32.onnx` (6.5GB) | Qwen3-1.7B causal LM, maintains growing KV-cache |
| Local Transformer | `local_transformer_f32.onnx` (1GB) | Depth-wise decoder, fresh KV-cache per frame |
| Codec Encoder | `codec_encoder.onnx` (3.4GB) | Encodes reference speaker waveform |
| Codec Decoder | `codec_decoder.onnx` (3.4GB) / `codec_decoder_int8.onnx` (850MB) | Decodes RVQ to 24kHz waveform |

## Requirements

- Python 3.12+
- ONNX Runtime (CPU or GPU with CUDA 13 + cuDNN 9)
- 16GB RAM minimum (for FP32 models)
- 8GB RAM with INT8 codec decoder

## Quick Start

### 1. Download Model

```bash
# Install huggingface-hub
pip install huggingface-hub

# Authenticate (model is gated)
hf auth login

# Download ONNX models (exclude large audio_synth folder)
hf download pltobing/MOSS-TTS-Realtime-ONNX \
    --local-dir models/MOSS-TTS-Realtime-ONNX \
    --exclude "audio_synth/*"
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Test Benchmark

```bash
python3 test_tts.py
```

Expected output:
- Audio file saved to `test_output.wav`
- RTF (Real-Time Factor) reported
- Target RTF < 1.0 for real-time streaming (requires GPU)

### 4. Run Streaming Service

```bash
# CPU mode
python3 tts_service.py

# Or with uvicorn
uvicorn tts_service:app --host 0.0.0.0 --port 8000
```

### 5. Docker Deployment (GPU)

```bash
# Build image
docker-compose build

# Run with GPU access
docker-compose up -d
```

## API Endpoints

### Health Check

```
GET /health
```

Returns service status and model info.

### WebSocket Streaming TTS

```
WebSocket /tts/stream
```

Protocol:
1. Client: `{"type": "init"}` or `{"type": "init", "reference_audio": "/path/to/ref.wav"}`
2. Server: `{"type": "ready", "ttfb_ms": <init_time>}`
3. Client: `{"type": "text", "delta": "text chunk"}` (streaming LLM output)
4. Server: `{"type": "audio", "data": base64_pcm16, "sample_rate": 24000}`
5. Client: `{"type": "end"}`
6. Server: `{"type": "done"}`

Audio format: mono 16-bit PCM at 24kHz, base64-encoded.

### REST Batch Synthesis

```
POST /tts/synthesize?text=Hello%20World&reference_audio=/path/to/ref.wav
```

Returns WAV file (mono 24kHz 16-bit PCM).

### List Available Voices

```
GET /voices
```

Returns reference audio files available for voice cloning.

## Performance

### CPU Benchmark Results

| Metric | Value |
|--------|-------|
| Audio Duration | 28.48s |
| Synthesis Time | 120.83s |
| RTF | 4.24 |

**Note**: RTF > 1.0 means synthesis takes longer than audio duration. For real-time streaming (RTF < 1.0), GPU acceleration with CUDA + cuDNN is required.

### Recommended Decoding Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| temperature | 0.7-0.8 | Lower = more stable |
| top_p | 0.6 | Nucleus sampling |
| top_k | 30-34 | Top-k sampling |
| repetition_penalty | 1.1-1.5 | Prevents loops |

## Project Structure

```
.
├── README.md                 # This file
├── CLAUDE.md                  # Claude Code development guide
├── Dockerfile                 # CUDA 12.8 runtime container
├── docker-compose.yml         # GPU-enabled deployment
├── requirements.txt           # Python dependencies
├── tts_service.py             # FastAPI streaming service
├── test_tts.py                # Benchmark test script
└── models/
    └── MOSS-TTS-Realtime-ONNX/ # Downloaded model files
        ├── onnx_models/
        │   ├── backbone_f32/
        │   ├── local_transformer_f32/
        │   ├── codec_encoder/
        │   └── codec_decoder/
        ├── onnx_models_quantized/
        │   └── codec_decoder_int8/  # Memory-efficient variant
        ├── configs/
        ├── tokenizers/
        └── audio_ref/  # Reference speakers for voice cloning
```

## CUDA Setup

For GPU acceleration:

```bash
# Install cuDNN 9 for CUDA 13
pip install nvidia-cudnn-cu13==9.20.0.48

# Verify CUDA provider
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Should show: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

## License

Apache 2.0 - See model repository for details.

## References

- [MOSS-TTS-Realtime ONNX](https://huggingface.co/pltobing/MOSS-TTS-Realtime-ONNX)
- [ONNX Runtime CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [LiveKit Voice Agents](https://docs.livekit.io/agents/)