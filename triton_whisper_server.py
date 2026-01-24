"""
Whisper Speech-to-Text Server using TensorRT-LLM
OpenAI-compatible API endpoints
Optimized for Blackwell GPU
"""

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
import uvicorn

# Add TensorRT-LLM whisper example to path
sys.path.insert(0, "/app/tensorrt_llm/examples/models/core/whisper")

app = FastAPI(title="Whisper TensorRT-LLM Server", version="1.0.0")

# Global model instance
whisper_runner = None
ENGINE_DIR = os.environ.get("WHISPER_ENGINE_DIR", "/workspace/whisper_engine")
ASSETS_DIR = "/app/tensorrt_llm/examples/models/core/whisper/assets"


def get_runner():
    """Lazy-load the Whisper TensorRT-LLM runner."""
    global whisper_runner
    if whisper_runner is None:
        print(f"Loading Whisper TensorRT-LLM engine from {ENGINE_DIR}...")

        from run import WhisperTRTLLM

        whisper_runner = WhisperTRTLLM(
            engine_dir=ENGINE_DIR,
            assets_dir=ASSETS_DIR,
            batch_size=8,  # Must match engine max_batch_size
        )
        print("Whisper TensorRT-LLM engine loaded successfully")
    return whisper_runner


@app.on_event("startup")
async def startup_event():
    """Pre-load the model on startup."""
    try:
        get_runner()
    except Exception as e:
        print(f"Warning: Failed to pre-load model: {e}")
        print("Model will be loaded on first request")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return JSONResponse(content={"status": "healthy"})


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "whisper-large-v3-turbo",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "tensorrt-llm",
            }
        ]
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(default="whisper-large-v3-turbo"),
    language: Optional[str] = Form(default=None),
    prompt: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
):
    """OpenAI-compatible audio transcription endpoint."""
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        runner = get_runner()

        from run import decode_wav_file

        # Build the text prefix based on language
        lang = language if language else "en"
        text_prefix = f"<|startoftranscript|><|{lang}|><|transcribe|><|notimestamps|>"

        # Transcribe using TensorRT-LLM Whisper
        results, duration = decode_wav_file(
            input_file_path=tmp_path,
            model=runner,
            text_prefix=text_prefix,
            mel_filters_dir=ASSETS_DIR,
        )

        # Extract transcribed text
        # results format: [(0, [""], words_list)]
        text = ""
        if results and len(results) > 0:
            if isinstance(results[0], tuple) and len(results[0]) >= 3:
                words = results[0][2]
                text = " ".join(words) if isinstance(words, list) else str(words)
            else:
                text = str(results[0])

        text = text.strip()

        if response_format == "json":
            return {"text": text}
        elif response_format == "verbose_json":
            return {
                "task": "transcribe",
                "language": lang,
                "duration": duration,
                "text": text,
                "segments": [],
            }
        elif response_format == "text":
            return PlainTextResponse(content=text)
        else:
            return {"text": text}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.post("/v1/audio/translations")
async def translate(
    file: UploadFile = File(...),
    model: str = Form(default="whisper-large-v3-turbo"),
    prompt: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
):
    """OpenAI-compatible audio translation endpoint (translate to English)."""
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        runner = get_runner()

        from run import decode_wav_file

        text_prefix = "<|startoftranscript|><|en|><|translate|><|notimestamps|>"

        results, duration = decode_wav_file(
            input_file_path=tmp_path,
            model=runner,
            text_prefix=text_prefix,
            mel_filters_dir=ASSETS_DIR,
        )

        text = ""
        if results and len(results) > 0:
            if isinstance(results[0], tuple) and len(results[0]) >= 3:
                words = results[0][2]
                text = " ".join(words) if isinstance(words, list) else str(words)
            else:
                text = str(results[0])

        text = text.strip()

        if response_format == "json":
            return {"text": text}
        elif response_format == "text":
            return PlainTextResponse(content=text)
        else:
            return {"text": text}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
