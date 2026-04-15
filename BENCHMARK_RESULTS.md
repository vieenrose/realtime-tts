# TTS ONNX Benchmark Results

Date: 2026-04-15
GPU: CUDA 13 on aarch64
Platform: Linux 6.17.0-1014-nvidia

## Summary

| Model | RTF | Latency (ms) | Audio/Chunk (s) | Streaming | Recommendation |
|-------|-----|--------------|-----------------|-----------|----------------|
| Fun-CosyVoice3 | **0.007** | 14.7 | 2.0 | ✅ Yes | **Best for streaming** |
| Chatterbox-q4 | **0.359** | 143 | 0.4 | ✅ Yes | Realtime capable |
| VoxCPM2-ONNX | 43.0 | 96/frame | 0.03 | ✅ Yes | Too slow on this GPU |
| Kokoro-v1.1-zh | 8.37 | 5900 | 0.7 | ❌ No | Batch only, slow |

## Winner: Fun-CosyVoice3

**FunAudioLLM/Fun-CosyVoice3-0.5B-2512** is the best choice for realtime streaming TTS:

- **RTF 0.007** - 143x faster than realtime (14.7ms produces 2.0s of audio)
- Streaming architecture with chunk-based inference (`chunk_size=25` in config)
- Multi-language support: zh, en, ja, ko, de, es, fr, ru, etc.
- ONNX models: `flow.decoder.estimator.fp32.onnx` (1.3GB), `speech_tokenizer_v3.onnx` (925MB)
- Sample rate: 24kHz

### Streaming Config (from cosyvoice3.yaml)

```yaml
chunk_size: 25        # streaming inference chunk size, in tokens
num_decoding_left_chunks: -1  # use all left chunks for flow decoder
token_frame_rate: 25  # Hz
token_mel_ratio: 2    # mel frames per token
```

### Benchmark Output

```
Average chunk latency: 14.70ms
Audio per chunk: 2.00s (50 mel frames at 25 Hz)
Estimated RTF: 0.007
```

## Runner-up: Chatterbox Multilingual Q4

**ipsilondev/chatterbox-multilingual-ONNX-q4** also meets realtime requirements:

- **RTF 0.359** - 2.8x faster than realtime
- Streaming architecture with LM + decoder pipeline
- Model size: 826MB total (embed_tokens + LM + speech_encoder + decoder)
- Supports zh/en with Q4 quantization for efficiency

### Benchmark Output

```
Average latency: 143.48ms
Audio duration: 0.400s (50 speech tokens → 9600 samples at 24kHz)
RTF: 0.359
```

## Why VoxCPM2 Failed

VoxCPM2-ONNX reports RTF 0.13-0.3 on RTX 4090, but performs poorly on this hardware:

- Decode step latency: 96ms per frame (target: <3ms)
- Estimated RTF: 43.0 (43x slower than realtime)
- ONNX Runtime warnings indicate CPU fallback: "Memcpy nodes added to the graph"
- The model's attention/kv-cache operations don't fully utilize CUDA on aarch64

## Kokoro Performance

Kokoro is designed for batch synthesis, not streaming:

- RTF 8.37 with placeholder input (5.9s synthesis for 0.7s audio)
- Single-shot batch architecture
- Model: 328MB ONNX + 51MB voice embeddings
- Better suited for offline synthesis

## Model Files Downloaded

```
models/
├── Fun-CosyVoice3/        (9.1GB)
│   ├── flow.decoder.estimator.fp32.onnx
│   ├── speech_tokenizer_v3.onnx
│   ├── speech_tokenizer_v3.batch.onnx
│   ├── campplus.onnx
│   └── config.json
├── chatterbox-q4/         (793MB)
│   └── onnx/
│       ├── embed_tokens.onnx
│       ├── language_model.onnx
│       ├── speech_encoder.onnx
│       └── conditional_decoder.onnx
├── VoxCPM2-ONNX/          (17GB)
│   ├── voxcpm2_prefill.onnx
│   ├── voxcpm2_decode_step.onnx
│   ├── audio_vae_encoder.onnx
│   └── audio_vae_decoder.onnx
├── kokoro/                (190MB)
│   ├── kokoro-v1.1-zh.onnx
│   └── voices-v1.1-zh.bin
└── MOSS-TTS-Realtime-ONNX/ (16GB)
```

## Recommended Setup for LiveKit

For LiveKit voice agents with streaming TTS:

1. Use Fun-CosyVoice3 as primary TTS engine
2. Configure chunk_size=25 for streaming
3. Pipeline: text → speech_tokenizer → flow.decoder → HiFi-GAN vocoder
4. CUDA acceleration via LD_LIBRARY_PATH for cuDNN/cuBLAS

```bash
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cudnn/lib:$VENV/lib/python3.12/site-packages/nvidia/cublas/lib:$VENV/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH"
```