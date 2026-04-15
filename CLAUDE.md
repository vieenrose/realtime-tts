# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Build and deploy MOSS-TTS-Realtime ONNX for LiveKit with CUDA acceleration, supporting streaming input/output for zh-TW and English mixed text.

## MOSS-TTS-Realtime ONNX Architecture

Pure ONNX Runtime inference pipeline - no PyTorch/Transformers dependency at runtime.

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

**Components:**
| Component | ONNX Model | Notes |
|-----------|------------|-------|
| Backbone LLM | `backbone_f32.onnx` | Qwen3-1.7B, maintains growing KV-cache |
| Local Transformer | `local_transformer_f32.onnx` | Depth-wise decoder, fresh KV-cache per frame |
| Codec Encoder | `codec_encoder.onnx` | Encodes reference speaker, run once per session |
| Codec Decoder | `codec_decoder.onnx` / `codec_decoder_int8.onnx` | Decodes RVQ to 24kHz, streaming KV-cache |

**Key specs:**
- Sampling rate: 24,000 Hz
- 16 RVQ codebooks per frame
- Memory: ~13GB with FP32, INT8 codec decoder recommended for 16GB RAM
- Supports 20+ languages (zh, en, ja, ko, de, es, fr, ru, etc.)

## Model Files

ONNX model: `pltobing/MOSS-TTS-Realtime-ONNX` (gated, requires HF auth)

Download structure:
```
onnx_models/
├── backbone_f32/backbone_f32.onnx + .data
├── local_transformer_f32/local_transformer_f32.onnx + .data
├── codec_encoder/codec_encoder.onnx + .data
├── codec_decoder/codec_decoder.onnx + .data
onnx_models_quantized/
├── codec_decoder_int8/codec_decoder_int8.onnx + .data
configs/
├── config_backbone.json
├── config_codec.json
tokenizers/
├── tokenizer.json
├── tokenizer_config.json
audio_ref/ (reference speakers for voice cloning)
```

## Download Command

```bash
hf download pltobing/MOSS-TTS-Realtime-ONNX --local-dir models/MOSS-TTS-Realtime-ONNX
```

## Dependencies

```bash
pip install numpy onnxruntime-gpu soundfile librosa tokenizers
```

## Recommended Decoding Parameters (optimized for zh-TW/en)

| Parameter | Value |
|-----------|-------|
| temperature | 0.7-0.8 |
| top_p | 0.6 |
| top_k | 30-34 |
| repetition_penalty | 1.1-1.5 |
| repetition_window | 50 |

## LiveKit Integration

For real-time voice agents:
- Streaming text input (incremental chunks from LLM via `push_text()`)
- Streaming audio output (RVQ tokens decoded incrementally)
- Voice cloning via reference audio conditioning
- KV-cache management across multi-turn dialogue

## CUDA/ONNX Runtime

- Use `CUDAExecutionProvider` for GPU acceleration
- INT8 codec decoder reduces memory without quality loss
- Backbone/local transformer must remain FP32 (INT8 causes hallucination)
- FP16 conversion possible for GPU (not tested on CPU)