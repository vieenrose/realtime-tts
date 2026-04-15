# Realtime TTS ONNX Streaming Service

A FastAPI/WebSocket service for streaming text-to-speech synthesis with CUDA acceleration, designed for LiveKit voice agents supporting zh-TW and English mixed text.

## TTS Model Selection

After benchmarking multiple ONNX TTS models on CUDA 13 aarch64:

| Model | RTF | Latency | Streaming | Recommendation |
|-------|-----|---------|-----------|----------------|
| **Fun-CosyVoice3** | **0.007** | 14.8ms | ✅ Yes | **Best for streaming** |
| Chatterbox-q4 | 0.08 | 191ms | ✅ Yes | Alternative (smaller) |
| VoxCPM2-ONNX | 36.5 | 97ms | ✅ Yes | ❌ Too slow on this GPU |
| MOSS-TTS-Realtime | 2.5 | - | ✅ Yes | ❌ Not realtime |
| Kokoro-v1.1-zh | 8.37 | 5.9s | ❌ No | ❌ Batch only |

**Recommended: Fun-CosyVoice3** (RTF 0.007 = 143x faster than realtime)

## Architecture (Fun-CosyVoice3)

```
Text Input ──► LLM (Qwen-based) ──► Speech Tokens (6561 vocab)
                                         │
                                         ▼
                               Flow Decoder (DiT) ──► Mel Spectrogram
                                                              │
                                                              ▼
                                                      HiFi-GAN Vocoder ──► 24kHz Audio
```

**Streaming Config:**
- `chunk_size: 25` tokens (50 mel frames)
- `token_frame_rate: 25` Hz
- Each chunk produces 2s of audio in 14.8ms

## Requirements

- Python 3.12+
- ONNX Runtime GPU (CUDA 13 + cuDNN 9)
- CUDA-capable GPU (tested on aarch64)

## Quick Start

### 1. Download Models

```bash
# Install huggingface-hub
pip install huggingface-hub

# Download Fun-CosyVoice3 (recommended)
hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
    --local-dir models/Fun-CosyVoice3

# Alternative: Chatterbox Multilingual Q4 (smaller, ~800MB)
hf download ipsilondev/chatterbox-multilingual-ONNX-q4 \
    --local-dir models/chatterbox-q4

# Alternative: VoxCPM2 (best on RTX 4090, slow on aarch64)
hf download ai4all8/VoxCPM2-ONNX \
    --local-dir models/VoxCPM2-ONNX
```

### 2. Install Dependencies

```bash
pip install numpy onnxruntime-gpu soundfile

# For CUDA 13 on aarch64 (Microsoft nightly)
pip install --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ onnxruntime-gpu

# cuDNN 9 for CUDA 13
pip install nvidia-cudnn-cu13 nvidia-cublas-cu13 nvidia-cuda-nvrtc-cu13
```

### 3. Configure CUDA Environment

```bash
# Set library paths for pip-installed cuDNN
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cudnn/lib:$VENV/lib/python3.12/site-packages/nvidia/cublas/lib:$VENV/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH"

# Verify CUDA provider
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Should show: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

### 4. Run Benchmark

```bash
source setup_cuda_env.sh
python3 benchmark_tts_real.py
```

Expected output:
```
| Model | RTF | Latency | Audio Generated | Quality | Realtime |
| cosyvoice | 0.007 | 14.8ms | 20.00s | ✓ | ✓ |
| chatterbox | 0.080 | 191ms | 2.40s | ✓ | ✓ |
```

### 5. Run Streaming Service

```bash
# With Fun-CosyVoice3
python3 tts_service_cosyvoice.py

# Or with uvicorn
uvicorn tts_service_cosyvoice:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### WebSocket Streaming TTS

```
WebSocket /tts/stream
```

Protocol:
1. Client: `{"type": "init", "text": "你好，Welcome to realtime TTS."}`
2. Server: `{"type": "ready", "ttfb_ms": <first_chunk_latency>}`
3. Client: `{"type": "text", "delta": " 這是一個測試。"}`
4. Server: `{"type": "audio", "data": base64_pcm16, "sample_rate": 24000}`
5. Client: `{"type": "end"}`
6. Server: `{"type": "done"}`

Audio format: mono 16-bit PCM at 24kHz, base64-encoded.

### REST Batch Synthesis

```
POST /tts/synthesize
Body: {"text": "你好世界", "voice": "default"}
```

Returns WAV file (mono 24kHz 16-bit PCM).

## Performance Details

### Memory Requirements (Fun-CosyVoice3)

| Metric | Value |
|--------|-------|
| Model RAM footprint | 3.1GB |
| Peak RAM usage | 3.2GB |
| Inference memory delta | ~2MB |
| Recommended min RAM | 5GB (with buffer) |
| Disk usage (full) | 9.1GB |

**Model memory breakdown:**
- `flow.decoder.estimator.fp32.onnx`: 1.97GB RAM (file: 1.27GB)
- `speech_tokenizer_v3.onnx`: 1.14GB RAM (file: 925MB)
- `campplus.onnx`: 0.4MB RAM (file: 27MB)

RAM/file ratio: 1.4x (ONNX Runtime overhead)

### Fun-CosyVoice3 Benchmark (CUDA)

```
Chunk latency: 14.80ms
Audio per chunk: 2.00s
Total audio: 20.00s (10 chunks)
Total synthesis time: 0.148s
RTF: 0.0074
```

**RTF 0.007** means:
- 1 second of synthesis produces 143 seconds of audio
- First audio chunk available in ~15ms
- Ideal for streaming voice agents

### Chatterbox Multilingual Q4 Benchmark

```
Synthesis latency: 191ms
Audio duration: 2.40s
RTF: 0.08
```

**RTF 0.08** means:
- 12x faster than realtime
- Suitable for non-demanding streaming
- Smaller model (800MB vs 3GB+)

### Quality Validation

- Audio RMS: 0.17 (normal levels)
- Peak: 0.9 (no clipping)
- STT transcription: Valid speech patterns detected

## Model Files Structure

```
models/
├── Fun-CosyVoice3/                  # Recommended (9.1GB)
│   ├── flow.decoder.estimator.fp32.onnx    # 1.3GB - streaming decoder
│   ├── speech_tokenizer_v3.onnx            # 925MB - text→tokens
│   ├── speech_tokenizer_v3.batch.onnx      # 925MB - batch variant
│   ├── campplus.onnx                       # 27MB - speaker embedding
│   ├── cosyvoice3.yaml                     # streaming config
│   └── flow.pt / llm.pt / hift.pt          # PyTorch weights (fallback)
│
├── chatterbox-q4/                   # Alternative (793MB)
│   └── onnx/
│       ├── embed_tokens.onnx               # 68MB
│       ├── language_model.onnx             # 353MB - Q4 quantized
│       ├── speech_encoder.onnx             # 180MB
│       └── conditional_decoder.onnx        # 225MB
│
├── VoxCPM2-ONNX/                    # Not recommended on aarch64 (17GB)
│   ├── voxcpm2_prefill.onnx
│   ├── voxcpm2_decode_step.onnx
│   ├── audio_vae_encoder.onnx
│   └── audio_vae_decoder.onnx
│
└── MOSS-TTS-Realtime-ONNX/          # Legacy (16GB, RTF 2.5)
    └── onnx_models/
        ├── backbone_f32/
        ├── local_transformer_f32/
        ├── codec_encoder/
        └── codec_decoder/
```

## Decoding Parameters (CosyVoice3)

| Parameter | Value | Notes |
|-----------|-------|-------|
| chunk_size | 25 | Streaming chunk size (tokens) |
| num_decoding_left_chunks | -1 | Use all context |
| cfg_rate | 0.7 | Classifier-free guidance |
| temperature | 0.8 | Flow matching temperature |
| top_p | 0.8 | Nucleus sampling |
| top_k | 25 | Top-k sampling |

## CUDA Setup Troubleshooting

If CUDA provider fails:

```bash
# Check cuDNN installation
ls .venv/lib/python3.12/site-packages/nvidia/cudnn/lib/
# Should show libcudnn.so.9

# Verify LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
# Must include cudnn, cublas, cuda_nvrtc paths

# Test ONNX Runtime
python3 -c "import onnxruntime as ort; sess = ort.InferenceSession('model.onnx', providers=['CUDAExecutionProvider']); print(sess.get_providers())"
```

## Multi-Language Support

Fun-CosyVoice3 supports:
- Chinese (zh, zh-TW, zh-CN)
- English (en)
- Japanese (ja)
- Korean (ko)
- German (de), Spanish (es), French (fr), Russian (ru)
- And 20+ other languages

## License

Apache 2.0 - See individual model repositories for details.

## References

- [Fun-CosyVoice3](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- [Chatterbox Multilingual ONNX](https://huggingface.co/ipsilondev/chatterbox-multilingual-ONNX-q4)
- [VoxCPM2-ONNX](https://huggingface.co/ai4all8/VoxCPM2-ONNX)
- [ONNX Runtime CUDA EP](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [LiveKit Agents](https://docs.livekit.io/agents/)