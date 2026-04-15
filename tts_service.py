"""MOSS-TTS-Realtime ONNX Streaming TTS Service for LiveKit.

A FastAPI/WebSocket service that provides streaming text-to-speech synthesis
using the MOSS-TTS-Realtime ONNX model, suitable for integration with LiveKit
voice agents.

Endpoints:
- /health - Health check
- /tts/stream - WebSocket streaming TTS
- /tts/synthesize - REST endpoint for batch synthesis
"""

import base64
import json
import logging
import os
import sys
import time
import wave
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response

# Add model directory to path
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models/MOSS-TTS-Realtime-ONNX"))
sys.path.insert(0, str(MODEL_DIR))

from inferencer_onnx import MossTTSRealtimeInferenceONNX, MossTTSRealtimeProcessor
from moss_text_tokenizer import MOSSTextTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model state
inferencer: Optional[MossTTSRealtimeInferenceONNX] = None
tokenizer: Optional[MOSSTextTokenizer] = None
processor: Optional[MossTTSRealtimeProcessor] = None
reference_audio_path: Optional[str] = None
backbone_config: Optional[dict] = None
codec_config: Optional[dict] = None

app = FastAPI(title="MOSS-TTS-Realtime ONNX Service")

CODEC_SAMPLE_RATE = 24000
NDArrayInt = npt.NDArray[np.int64]
NDArrayFloat = npt.NDArray[np.floating]


def pcm16_to_b64(audio: np.ndarray) -> str:
    """Convert float32 audio [-1,1] to base64-encoded 16-bit PCM."""
    pcm16 = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    return base64.b64encode(pcm16.tobytes()).decode()


def sanitize_audio_tokens(
    tokens: NDArrayInt,
    codebook_size: int,
    eos_audio_id: int,
) -> NDArrayInt:
    """Filter out EOS and invalid tokens before codec decoding.

    The codec decoder only accepts tokens in range [0, codebook_size).
    EOS token (1026) must be removed to avoid out-of-bounds errors.
    """
    if tokens.ndim == 1:
        tokens = np.expand_dims(tokens, axis=0)

    if tokens.size == 0:
        return tokens

    # Find first row with EOS or invalid token
    eos_rows = np.nonzero(tokens[:, 0] == eos_audio_id)[0]
    invalid_rows = ((tokens < 0) | (tokens >= codebook_size)).any(axis=1)
    invalid_rows_idx = np.nonzero(invalid_rows)[0]

    stop_idx = None
    if eos_rows.size > 0:
        stop_idx = int(eos_rows[0])
    if invalid_rows_idx.size > 0:
        invalid_idx = int(invalid_rows_idx[0])
        stop_idx = invalid_idx if stop_idx is None else min(stop_idx, invalid_idx)

    if stop_idx is not None:
        return tokens[:stop_idx]
    return tokens


def load_model():
    """Load ONNX model sessions and initialize inferencer."""
    global inferencer, tokenizer, processor, reference_audio_path, backbone_config, codec_config

    logger.info(f"Loading models from {MODEL_DIR}")

    # Load tokenizer
    tokenizer = MOSSTextTokenizer(
        str(MODEL_DIR / "tokenizers" / "tokenizer.json"),
        str(MODEL_DIR / "tokenizers" / "tokenizer_config.json")
    )

    # Provider selection - prefer CUDA if available
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    available = ort.get_available_providers()
    logger.info(f"Available ONNX Runtime providers: {available}")

    if "CUDAExecutionProvider" not in available:
        logger.warning("CUDA not available, falling back to CPU")
        providers = ["CPUExecutionProvider"]

    # Load configs
    with open(MODEL_DIR / "configs" / "config_backbone.json") as f:
        backbone_config = json.load(f)
    with open(MODEL_DIR / "configs" / "config_codec.json") as f:
        codec_config = json.load(f)

    # Use INT8 codec decoder for memory efficiency (if available)
    codec_decoder_path = MODEL_DIR / "onnx_models_quantized" / "codec_decoder_int8" / "codec_decoder_int8.onnx"
    if not codec_decoder_path.exists():
        codec_decoder_path = MODEL_DIR / "onnx_models" / "codec_decoder" / "codec_decoder.onnx"
        logger.info("Using FP32 codec decoder")
    else:
        logger.info("Using INT8 codec decoder")

    # Load ONNX sessions (paths match huggingface repo structure)
    backbone_llm = ort.InferenceSession(
        str(MODEL_DIR / "onnx_models" / "backbone_f32" / "backbone_f32.onnx"),
        providers=providers
    )
    backbone_local = ort.InferenceSession(
        str(MODEL_DIR / "onnx_models" / "local_transformer_f32" / "local_transformer_f32.onnx"),
        providers=providers
    )
    codec_decoder = ort.InferenceSession(
        str(codec_decoder_path),
        providers=providers
    )
    codec_encoder = ort.InferenceSession(
        str(MODEL_DIR / "onnx_models" / "codec_encoder" / "codec_encoder.onnx"),
        providers=providers
    )

    # Initialize inferencer with params optimized for zh-TW/en mixed text
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
        channels=backbone_config["rvq"],
        audio_pad_tokens=backbone_config["audio_pad_token"]
    )

    # Set default reference audio
    ref_audio_candidates = [
        MODEL_DIR / "audio_ref" / "female_shadowheart.flac",
        MODEL_DIR / "audio_ref" / "male_stewie.mp3",
        MODEL_DIR / "audio_ref" / "david-attenborough.mp3",
    ]
    for ref in ref_audio_candidates:
        if ref.exists():
            reference_audio_path = str(ref)
            logger.info(f"Using reference audio: {reference_audio_path}")
            break

    logger.info("Model loading complete")


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "model_loaded": inferencer is not None,
        "reference_audio": reference_audio_path,
        "codec_config": codec_config.get("quantizer_kwargs", {}).get("codebook_size") if codec_config else None
    })


@app.websocket("/tts/stream")
async def tts_stream(ws: WebSocket):
    """WebSocket streaming TTS endpoint for LiveKit voice agents.

    Protocol:
    1. Client sends: {"type": "init"} or {"type": "init", "reference_audio": "/path/to/ref.wav"}
    2. Server sends: {"type": "ready", "ttfb_ms": <init_time>}
    3. Client sends: {"type": "text", "delta": "text chunk"} (streaming LLM output)
    4. Server sends: {"type": "audio", "data": base64_pcm16, "sample_rate": 24000}
    5. Client sends: {"type": "end"} when LLM stream ends
    6. Server sends: {"type": "audio", ...} for remaining audio
    7. Server sends: {"type": "done"} when synthesis complete

    Audio format: mono 16-bit PCM at 24kHz, base64-encoded.
    """
    await ws.accept()

    start_time = time.time()
    audio_chunks_sent = 0

    try:
        # Initialize session
        init_msg = await ws.receive_json()
        if init_msg.get("type") != "init":
            await ws.send_json({"type": "error", "message": "Expected init message first"})
            return

        # Setup voice cloning with reference audio
        ref_audio = init_msg.get("reference_audio", reference_audio_path)
        if ref_audio and Path(ref_audio).exists():
            prompt_tokens = inferencer._encode_reference_audio(ref_audio)
            input_ids = processor.make_ensemble(prompt_tokens.squeeze(1))
            logger.info(f"Voice cloning enabled with: {ref_audio}")
        else:
            input_ids = processor.make_ensemble()
            logger.info("No reference audio - using default voice")

        # Reset turn state (clear cache for new session)
        inferencer.reset_turn(input_ids=input_ids, include_system_prompt=False, reset_cache=True)

        init_time_ms = (time.time() - start_time) * 1000
        await ws.send_json({"type": "ready", "ttfb_ms": init_time_ms})

        # Streaming text processing loop
        while True:
            msg = await ws.receive_json()
            msg_type = msg.get("type")

            if msg_type == "text":
                delta = msg.get("delta", "")
                if not delta:
                    continue

                # Push text delta, get audio token frames
                audio_frames = inferencer.push_text(delta)

                # Decode audio frames to waveform (with sanitization)
                codebook_size = inferencer.codebook_size
                eos_audio_id = inferencer.eos_audio_id
                for frame in audio_frames:
                    tokens = sanitize_audio_tokens(frame, codebook_size, eos_audio_id)
                    if tokens.size > 0:
                        inferencer.push_tokens(tokens)

                # Stream decoded audio chunks
                for audio_chunk in inferencer.audio_chunks():
                    audio_b64 = pcm16_to_b64(audio_chunk)
                    await ws.send_json({
                        "type": "audio",
                        "data": audio_b64,
                        "sample_rate": CODEC_SAMPLE_RATE,
                        "format": "pcm16"
                    })
                    audio_chunks_sent += 1

            elif msg_type == "end":
                # Signal end of text input
                inferencer.end_text()

                # Drain remaining generation (text-pad tokens until EOS)
                codebook_size = inferencer.codebook_size
                eos_audio_id = inferencer.eos_audio_id
                while True:
                    frames = inferencer.drain(max_steps=1)
                    if not frames:
                        break
                    for frame in frames:
                        tokens = sanitize_audio_tokens(frame, codebook_size, eos_audio_id)
                        if tokens.size > 0:
                            inferencer.push_tokens(tokens)
                    if inferencer.is_finished:
                        break

                    for audio_chunk in inferencer.audio_chunks():
                        audio_b64 = pcm16_to_b64(audio_chunk)
                        await ws.send_json({
                            "type": "audio",
                            "data": audio_b64,
                            "sample_rate": CODEC_SAMPLE_RATE,
                            "format": "pcm16"
                        })
                        audio_chunks_sent += 1

                # Flush final audio buffer
                final_audio = inferencer.flush()
                if final_audio is not None and len(final_audio) > 0:
                    audio_b64 = pcm16_to_b64(final_audio)
                    await ws.send_json({
                        "type": "audio",
                        "data": audio_b64,
                        "sample_rate": CODEC_SAMPLE_RATE,
                        "format": "pcm16"
                    })
                    audio_chunks_sent += 1

                await ws.send_json({"type": "done"})
                break

            elif msg_type == "abort":
                logger.info("Session aborted by client")
                break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")

    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
        await ws.send_json({"type": "error", "message": str(e)})

    finally:
        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"Session completed: {audio_chunks_sent} chunks in {duration_ms:.1f}ms")


@app.post("/tts/synthesize")
async def synthesize(text: str, reference_audio: Optional[str] = None):
    """REST endpoint for batch synthesis.

    Args:
        text: Text to synthesize
        reference_audio: Optional path to reference audio for voice cloning

    Returns: WAV file (mono 24kHz 16-bit PCM)
    """
    ref_audio = reference_audio or reference_audio_path

    # Setup voice cloning
    if ref_audio and Path(ref_audio).exists():
        prompt_tokens = inferencer._encode_reference_audio(ref_audio)
        input_ids = processor.make_ensemble(prompt_tokens.squeeze(1))
    else:
        input_ids = processor.make_ensemble()

    inferencer.reset_turn(input_ids=input_ids, include_system_prompt=False, reset_cache=True)

    # Collect audio chunks
    audio_chunks: list[np.ndarray] = []

    # Get sanitization params
    codebook_size = inferencer.codebook_size
    eos_audio_id = inferencer.eos_audio_id

    # Process all text at once (with sanitization)
    audio_frames = inferencer.push_text(text)
    for frame in audio_frames:
        tokens = sanitize_audio_tokens(frame, codebook_size, eos_audio_id)
        if tokens.size > 0:
            inferencer.push_tokens(tokens)

    for audio_chunk in inferencer.audio_chunks():
        if audio_chunk.size > 0:
            audio_chunks.append(audio_chunk.astype(np.float32).reshape(-1))

    # Signal end and drain (with sanitization)
    inferencer.end_text()
    while True:
        frames = inferencer.drain(max_steps=1)
        if not frames:
            break
        for frame in frames:
            tokens = sanitize_audio_tokens(frame, codebook_size, eos_audio_id)
            if tokens.size > 0:
                inferencer.push_tokens(tokens)
        if inferencer.is_finished:
            break
        for audio_chunk in inferencer.audio_chunks():
            if audio_chunk.size > 0:
                audio_chunks.append(audio_chunk.astype(np.float32).reshape(-1))

    # Flush final buffer
    final_audio = inferencer.flush()
    if final_audio is not None:
        audio_chunks.append(final_audio.astype(np.float32).reshape(-1))

    if not audio_chunks:
        return JSONResponse({"error": "No audio generated"}, status_code=500)

    # Concatenate and encode as WAV
    audio = np.concatenate(audio_chunks)
    pcm16 = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)

    # Build WAV in memory
    wav_buffer = []
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(CODEC_SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())

    duration_seconds = len(audio) / CODEC_SAMPLE_RATE

    return Response(
        content=bytes(wav_buffer),
        media_type="audio/wav",
        headers={
            "X-Sample-Rate": str(CODEC_SAMPLE_RATE),
            "X-Audio-Duration": f"{duration_seconds:.2f}",
            "X-Audio-Chunks": str(len(audio_chunks))
        }
    )


@app.get("/voices")
async def list_voices():
    """List available reference voices for voice cloning."""
    voices = []
    audio_ref_dir = MODEL_DIR / "audio_ref"
    if audio_ref_dir.exists():
        for f in audio_ref_dir.iterdir():
            if f.suffix in (".wav", ".flac", ".mp3"):
                voices.append({
                    "name": f.stem,
                    "path": str(f),
                    "size_kb": f.stat().st_size // 1024
                })
    return JSONResponse({"voices": voices})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)