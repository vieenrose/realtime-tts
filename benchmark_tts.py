#!/usr/bin/env python3
"""Benchmark TTS ONNX models for RTF comparison."""

import os
import sys
import time
import wave
import numpy as np
from pathlib import Path

# Set CUDA library paths BEFORE importing onnxruntime
VENV_DIR = Path(__file__).parent / ".venv"
os.environ["LD_LIBRARY_PATH"] = (
    f"{VENV_DIR}/lib/python3.12/site-packages/nvidia/cudnn/lib:"
    f"{VENV_DIR}/lib/python3.12/site-packages/nvidia/cublas/lib:"
    f"{VENV_DIR}/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:"
    f"{os.environ.get('LD_LIBRARY_PATH', '')}"
)

import onnxruntime as ort

TEST_TEXT = "你好，Welcome to realtime TTS. 這是一個測試。"
SAMPLE_RATE = 24000

def benchmark_kokoro():
    """Benchmark Kokoro ONNX (82M model, optimized for speed)."""
    from kokoro_onnx import Kokoro

    model_dir = Path("models/kokoro")

    if not model_dir.exists():
        print("Kokoro model not found")
        return None

    print("\n=== Kokoro ONNX (82M) ===")

    # Kokoro is designed for fast inference
    kokoro = Kokoro(model_dir / "kokoro-v1.1-zh.onnx", model_dir / "voices-v1.1-zh.bin")

    # Benchmark synthesis
    text = TEST_TEXT
    lang_code = "z"  # Chinese voice

    print(f"Synthesizing: {text}")
    start_time = time.time()

    audio, sr = kokoro.create(text, voice=lang_code, speed=1.0)

    elapsed = time.time() - start_time
    audio_duration = len(audio) / sr
    rtf = elapsed / audio_duration

    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"Synthesis time: {elapsed:.3f}s")
    print(f"RTF: {rtf:.3f}")

    # Save audio
    output_path = f"output_kokoro.wav"
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())
    print(f"Saved to: {output_path}")

    return {"rtf": rtf, "audio_duration": audio_duration, "synthesis_time": elapsed}

def benchmark_voxcpm2():
    """Benchmark VoxCPM2 ONNX (streaming architecture)."""
    model_dir = Path("models/VoxCPM2-ONNX")

    if not model_dir.exists():
        print("VoxCPM2-ONNX not found")
        return None

    print("\n=== VoxCPM2 ONNX ===")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # Load all sessions
    print("Loading models...")
    start_load = time.time()

    prefill = ort.InferenceSession(str(model_dir / "voxcpm2_prefill.onnx"), providers=providers)
    decode_step = ort.InferenceSession(str(model_dir / "voxcpm2_decode_step.onnx"), providers=providers)
    vae_encoder = ort.InferenceSession(str(model_dir / "audio_vae_encoder.onnx"), providers=providers)
    vae_decoder = ort.InferenceSession(str(model_dir / "audio_vae_decoder.onnx"), providers=providers)

    load_time = time.time() - start_load
    print(f"Model load time: {load_time:.2f}s")
    print(f"Providers: {prefill.get_providers()}")

    # Report model sizes
    sizes = {
        "prefill": 6.6,
        "decode_step": 20.5,
        "vae_encoder": 0.7,
        "vae_decoder": 1.0
    }
    print(f"Model sizes: {sum(sizes.values()):.1f}MB total")

    # Test decode step latency (this is the streaming component)
    print("\nTesting decode step latency (streaming)...")

    # Simulate decode step with realistic inputs
    seq_len = 10  # typical sequence length after prefill
    dit_hidden = np.random.randn(1, 2048).astype(np.float32)
    base_keys = np.random.randn(1, 28, 2, seq_len, 128).astype(np.float32)
    base_values = np.random.randn(1, 28, 2, seq_len, 128).astype(np.float32)
    residual_keys = np.random.randn(1, 8, 2, seq_len, 128).astype(np.float32)
    residual_values = np.random.randn(1, 8, 2, seq_len, 128).astype(np.float32)
    prefix_feat_cond = np.random.randn(1, 4, 64).astype(np.float32)
    noise = np.random.randn(1, 4, 64).astype(np.float32)
    cfg_value = np.array(1.0, dtype=np.float32)

    # Warmup
    for _ in range(3):
        decode_step.run(None, {
            "dit_hidden": dit_hidden,
            "base_next_keys": base_keys,
            "base_next_values": base_values,
            "residual_next_keys": residual_keys,
            "residual_next_values": residual_values,
            "prefix_feat_cond": prefix_feat_cond,
            "noise": noise,
            "cfg_value": cfg_value
        })

    # Benchmark decode step
    num_iterations = 50
    start_time = time.time()
    for _ in range(num_iterations):
        decode_step.run(None, {
            "dit_hidden": dit_hidden,
            "base_next_keys": base_keys,
            "base_next_values": base_values,
            "residual_next_keys": residual_keys,
            "residual_next_values": residual_values,
            "prefix_feat_cond": prefix_feat_cond,
            "noise": noise,
            "cfg_value": cfg_value
        })
    elapsed = time.time() - start_time

    step_latency_ms = (elapsed / num_iterations) * 1000
    frames_per_second = 1000 / step_latency_ms  # each step produces one frame
    audio_per_second = frames_per_second * (64 / SAMPLE_RATE)  # 64 samples per frame

    print(f"Decode step latency: {step_latency_ms:.2f}ms")
    print(f"Estimated max audio/sec: {audio_per_second:.2f}s audio per second")
    print(f"Estimated RTF: ~{1/audio_per_second:.3f} (streaming decode)")

    return {
        "load_time": load_time,
        "decode_latency_ms": step_latency_ms,
        "providers": prefill.get_providers()
    }

def benchmark_chatterbox():
    """Benchmark Chatterbox Multilingual ONNX Q4."""
    model_dir = Path("models/chatterbox-q4")
    onnx_dir = model_dir / "onnx"

    if not onnx_dir.exists():
        print("Chatterbox ONNX not found")
        return None

    print("\n=== Chatterbox Multilingual ONNX Q4 ===")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    print("Loading models...")
    start_load = time.time()

    embed_tokens = ort.InferenceSession(str(onnx_dir / "embed_tokens.onnx"), providers=providers)
    language_model = ort.InferenceSession(str(onnx_dir / "language_model.onnx"), providers=providers)
    speech_encoder = ort.InferenceSession(str(onnx_dir / "speech_encoder.onnx"), providers=providers)
    decoder = ort.InferenceSession(str(onnx_dir / "conditional_decoder.onnx"), providers=providers)

    load_time = time.time() - start_load
    print(f"Model load time: {load_time:.2f}s")
    print(f"Providers: {embed_tokens.get_providers()}")

    # Model sizes
    sizes = {
        "embed_tokens": 68,
        "language_model": 353,
        "speech_encoder": 180,
        "decoder": 225
    }
    print(f"Model sizes: {sum(sizes.values())}MB total")

    # Test language model inference (main bottleneck)
    print("\nTesting language model latency...")

    # Simulate realistic input
    batch_size = 1
    seq_len = 32
    hidden_size = 2560

    # Get actual input specs
    lm_inputs = language_model.get_inputs()
    print(f"LM inputs: {[i.name for i in lm_inputs]}")

    # Create dummy inputs based on model specs
    dummy_inputs = {}
    for inp in lm_inputs:
        shape = [dim if isinstance(dim, int) else batch_size for dim in inp.shape]
        if inp.type == 'tensor(int64)':
            dummy_inputs[inp.name] = np.random.randint(0, 1000, shape).astype(np.int64)
        else:
            dummy_inputs[inp.name] = np.random.randn(*shape).astype(np.float32)

    # Warmup
    try:
        for _ in range(3):
            language_model.run(None, dummy_inputs)
    except Exception as e:
        print(f"LM warmup failed (needs proper tokenizer): {e}")
        print("Testing decoder instead...")

        # Test decoder as fallback
        dec_inputs = decoder.get_inputs()
        dummy_dec_inputs = {}
        for inp in dec_inputs:
            shape = [dim if isinstance(dim, int) else batch_size for dim in inp.shape]
            if inp.type == 'tensor(int64)':
                dummy_dec_inputs[inp.name] = np.random.randint(0, 1000, shape).astype(np.int64)
            else:
                dummy_dec_inputs[inp.name] = np.random.randn(*shape).astype(np.float32)

        # Benchmark decoder
        num_iterations = 20
        start_time = time.time()
        for _ in range(num_iterations):
            decoder.run(None, dummy_dec_inputs)
        elapsed = time.time() - start_time

        dec_latency_ms = (elapsed / num_iterations) * 1000
        print(f"Decoder latency: {dec_latency_ms:.2f}ms")

    return {
        "load_time": load_time,
        "providers": embed_tokens.get_providers()
    }

def benchmark_cosyvoice():
    """Benchmark Fun-CosyVoice3 ONNX."""
    model_dir = Path("models/Fun-CosyVoice3")

    if not model_dir.exists():
        print("Fun-CosyVoice3 not found")
        return None

    print("\n=== Fun-CosyVoice3 ===")

    onnx_files = list(model_dir.glob("*.onnx"))
    if not onnx_files:
        print("No ONNX files found")
        return None

    print(f"ONNX files: {[f.name for f in onnx_files]}")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # Load main models
    print("Loading models...")
    start_load = time.time()

    models_loaded = {}
    for onnx_file in onnx_files:
        try:
            sess = ort.InferenceSession(str(onnx_file), providers=providers)
            models_loaded[onnx_file.name] = sess
            print(f"  Loaded: {onnx_file.name} ({onnx_file.stat().st_size / 1024**2:.1f}MB)")
        except Exception as e:
            print(f"  Failed: {onnx_file.name}: {e}")

    load_time = time.time() - start_load
    print(f"Model load time: {load_time:.2f}s")

    # Check speech_tokenizer (key model for streaming)
    if "speech_tokenizer_v3.onnx" in models_loaded:
        tokenizer = models_loaded["speech_tokenizer_v3.onnx"]
        print(f"\nSpeech tokenizer inputs:")
        for inp in tokenizer.get_inputs():
            print(f"  {inp.name}: shape={inp.shape} type={inp.type}")

    return {
        "load_time": load_time,
        "models": list(models_loaded.keys()),
        "providers": list(models_loaded.values())[0].get_providers() if models_loaded else []
    }

def main():
    print("TTS ONNX Benchmark")
    print(f"Test text: {TEST_TEXT}")
    print(f"CUDA available: {'CUDAExecutionProvider' in ort.get_available_providers()}")

    results = {}

    # Benchmark Kokoro (fastest baseline)
    try:
        results["kokoro"] = benchmark_kokoro()
    except Exception as e:
        print(f"Kokoro benchmark failed: {e}")
        results["kokoro"] = None

    # Benchmark VoxCPM2 (streaming)
    results["voxcpm2"] = benchmark_voxcpm2()

    # Benchmark Chatterbox
    results["chatterbox"] = benchmark_chatterbox()

    # Benchmark CosyVoice
    results["cosyvoice"] = benchmark_cosyvoice()

    print("\n" + "="*50)
    print("=== Summary ===")
    print("="*50)

    for model, result in results.items():
        if result:
            if "rtf" in result:
                print(f"{model}: RTF={result['rtf']:.3f} (real synthesis)")
            elif "decode_latency_ms" in result:
                print(f"{model}: decode={result['decode_latency_ms']:.2f}ms (streaming)")
            else:
                print(f"{model}: load_time={result['load_time']:.2f}s")
        else:
            print(f"{model}: Not available")

    # Recommendation
    print("\n=== Recommendation ===")
    best_rtf = None
    best_model = None
    for model, result in results.items():
        if result and "rtf" in result:
            if best_rtf is None or result["rtf"] < best_rtf:
                best_rtf = result["rtf"]
                best_model = model

    if best_model:
        print(f"Best RTF: {best_model} ({best_rtf:.3f})")

    # Streaming candidates
    print("\nStreaming candidates:")
    if results["voxcpm2"] and results["voxcpm2"].get("decode_latency_ms"):
        print(f"  VoxCPM2: {results['voxcpm2']['decode_latency_ms']:.2f}ms per frame")

if __name__ == "__main__":
    main()