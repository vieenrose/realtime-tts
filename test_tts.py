#!/usr/bin/env python3
"""Quick test script for MOSS-TTS-Realtime ONNX inference."""

import sys
from pathlib import Path
from typing import Tuple

# Add model directory to path
MODEL_DIR = Path(__file__).parent / "models" / "MOSS-TTS-Realtime-ONNX"

# Import model inference code directly
sys.path.insert(0, str(MODEL_DIR))

import json
import time
import wave
import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from inferencer_onnx import MossTTSRealtimeInferenceONNX, MossTTSRealtimeProcessor
from moss_text_tokenizer import MOSSTextTokenizer

SAMPLE_RATE = 24000
NDArrayInt = npt.NDArray[np.int64]


def sanitize_tokens(
    tokens: NDArrayInt,
    codebook_size: int,
    eos_audio_id: int,
) -> Tuple[NDArrayInt, bool]:
    """Validate and truncate audio tokens at EOS or invalid code boundaries.

    The codec decoder only accepts tokens in range [0, codebook_size).
    EOS token (1026) and other invalid tokens must be filtered out.
    """
    if tokens.ndim == 1:
        tokens = np.expand_dims(tokens, axis=0)

    if tokens.size == 0:
        return tokens, False

    # Find rows with EOS token
    eos_rows = np.nonzero(tokens[:, 0] == eos_audio_id)[0]

    # Find rows with invalid codes (outside [0, codebook_size))
    invalid_rows = ((tokens < 0) | (tokens >= codebook_size)).any(axis=1)
    invalid_rows_idx = np.nonzero(invalid_rows)[0]

    stop_idx = None
    if eos_rows.size > 0:
        stop_idx = int(eos_rows[0])

    if invalid_rows_idx.size > 0:
        invalid_idx = int(invalid_rows_idx[0])
        stop_idx = invalid_idx if stop_idx is None else min(stop_idx, invalid_idx)

    if stop_idx is not None:
        tokens = tokens[:stop_idx]
        return tokens, True

    return tokens, False


def main():
    print(f"Testing MOSS-TTS-Realtime ONNX from {MODEL_DIR}")

    # Check providers
    print(f"Available providers: {ort.get_available_providers()}")

    # Load tokenizer
    tokenizer = MOSSTextTokenizer(
        str(MODEL_DIR / "tokenizers" / "tokenizer.json"),
        str(MODEL_DIR / "tokenizers" / "tokenizer_config.json")
    )

    # Load configs
    with open(MODEL_DIR / "configs" / "config_backbone.json") as f:
        backbone_config = json.load(f)
    with open(MODEL_DIR / "configs" / "config_codec.json") as f:
        codec_config = json.load(f)

    print(f"Backbone: rvq={backbone_config['rvq']}, audio_pad={backbone_config['audio_pad_token']}")
    print(f"Codec: codebook_size={codec_config['quantizer_kwargs']['codebook_size']}")

    # Provider selection - prefer CUDA
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # Use INT8 codec decoder if available
    codec_decoder_path = MODEL_DIR / "onnx_models_quantized" / "codec_decoder_int8" / "codec_decoder_int8.onnx"
    if not codec_decoder_path.exists():
        codec_decoder_path = MODEL_DIR / "onnx_models" / "codec_decoder" / "codec_decoder.onnx"
        print("Using FP32 codec decoder")
    else:
        print("Using INT8 codec decoder")

    # Load sessions (matching actual file names from huggingface)
    print("Loading backbone LLM...")
    backbone_llm = ort.InferenceSession(
        str(MODEL_DIR / "onnx_models" / "backbone_f32" / "backbone_f32.onnx"),
        providers=providers
    )
    print(f"Backbone provider: {backbone_llm.get_providers()}")

    print("Loading local transformer...")
    backbone_local = ort.InferenceSession(
        str(MODEL_DIR / "onnx_models" / "local_transformer_f32" / "local_transformer_f32.onnx"),
        providers=providers
    )

    print("Loading codec decoder...")
    codec_decoder = ort.InferenceSession(
        str(codec_decoder_path),
        providers=providers
    )

    print("Loading codec encoder...")
    codec_encoder = ort.InferenceSession(
        str(MODEL_DIR / "onnx_models" / "codec_encoder" / "codec_encoder.onnx"),
        providers=providers
    )

    # Initialize inferencer with zh-TW/en optimized params
    inferencer = MossTTSRealtimeInferenceONNX(
        tokenizer=tokenizer,
        backbone_llm=backbone_llm,
        backbone_local=backbone_local,
        codec_decoder=codec_decoder,
        codec_encoder=codec_encoder,
        backbone_config=backbone_config,
        codec_config=codec_config,
        temperature=0.75,
        top_p=0.6,
        top_k=30,
        repetition_penalty=1.2,
        repetition_window=50,
    )

    processor = MossTTSRealtimeProcessor(
        tokenizer=tokenizer,
        channels=backbone_config["rvq"],  # 16
        audio_pad_tokens=backbone_config["audio_pad_token"]  # 1024
    )

    # Test with reference audio for voice cloning
    ref_audio_candidates = [
        MODEL_DIR / "audio_ref" / "female_shadowheart.flac",
        MODEL_DIR / "audio_ref" / "male_stewie.mp3",
    ]
    ref_audio = None
    for ref in ref_audio_candidates:
        if ref.exists():
            ref_audio = ref
            break

    if ref_audio:
        print(f"\nUsing reference audio: {ref_audio}")
        prompt_tokens = inferencer._encode_reference_audio(str(ref_audio))
        input_ids = processor.make_ensemble(prompt_tokens.squeeze(1))
    else:
        print("\nNo reference audio - using default voice")
        input_ids = processor.make_ensemble()

    # Test text (zh-TW and English mixed)
    test_text = "你好，Welcome to MOSS TTS Realtime. 這是一個測試，testing streaming text-to-speech synthesis."

    print(f"\nSynthesizing: {test_text}")

    inferencer.reset_turn(input_ids=input_ids, include_system_prompt=False, reset_cache=True)

    # Get codebook_size and eos_audio_id for sanitization
    codebook_size = inferencer.codebook_size  # 1024
    eos_audio_id = inferencer.eos_audio_id  # 1026

    start_time = time.time()
    audio_chunks = []

    # Push text and collect audio (with sanitization)
    audio_frames = inferencer.push_text(test_text)
    for frame in audio_frames:
        tokens = frame if frame.ndim == 2 else frame[0] if frame.ndim == 3 else frame
        tokens, truncated = sanitize_tokens(tokens, codebook_size, eos_audio_id)
        if tokens.size > 0:
            inferencer.push_tokens(tokens)
        if truncated:
            break  # Stop if we hit EOS

    for chunk in inferencer.audio_chunks():
        if chunk.size > 0:
            audio_chunks.append(chunk.astype(np.float32).reshape(-1))

    # End and drain (with sanitization)
    inferencer.end_text()
    while True:
        frames = inferencer.drain(max_steps=1)
        if not frames:
            break
        for frame in frames:
            tokens = frame if frame.ndim == 2 else frame[0] if frame.ndim == 3 else frame
            tokens, truncated = sanitize_tokens(tokens, codebook_size, eos_audio_id)
            if tokens.size > 0:
                inferencer.push_tokens(tokens)
            if truncated or inferencer.is_finished:
                break
        for chunk in inferencer.audio_chunks():
            if chunk.size > 0:
                audio_chunks.append(chunk.astype(np.float32).reshape(-1))
        if inferencer.is_finished:
            break

    # Flush final buffer (with sanitization for remaining tokens)
    # Note: flush() internally handles the buffer, but we need to ensure
    # the buffer doesn't have invalid tokens
    final = inferencer.flush()
    if final is not None and final.size > 0:
        audio_chunks.append(final.astype(np.float32).reshape(-1))

    elapsed = time.time() - start_time

    if audio_chunks:
        audio = np.concatenate(audio_chunks)
        duration = len(audio) / SAMPLE_RATE
        rtf = elapsed / duration if duration > 0 else 0

        # Save WAV
        out_path = Path(__file__).parent / "test_output.wav"
        pcm16 = np.clip(audio, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16)
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm16.tobytes())

        print(f"\n=== Results ===")
        print(f"Audio duration: {duration:.2f}s")
        print(f"Synthesis time: {elapsed:.2f}s")
        print(f"RTF (Real-Time Factor): {rtf:.3f}")
        print(f"Output saved to: {out_path}")
        print(f"Target RTF < 1.0: {'PASS' if rtf < 1.0 else 'FAIL'}")
    else:
        print("ERROR: No audio generated")


if __name__ == "__main__":
    main()