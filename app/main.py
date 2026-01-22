import os
import re
import logging
import sys
import json
import time
import uuid
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import StreamingResponse
import asyncio
from pydantic import BaseModel, Field
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import pypdf
import io
from bs4 import BeautifulSoup


# --- OpenAI Chat Completion Models ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "openai/gpt-oss-20b"
    messages: list[ChatMessage]
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Optional[ChatCompletionUsage] = None

# --- LOGGING SETUP ---
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Create logger
logger = logging.getLogger("rag-api")
logger.setLevel(logging.INFO)

# Console handler (for docker logs)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
logger.addHandler(console_handler)

# File handler with rotation (10MB max, keep 5 backups)
LOG_DIR = os.getenv("LOG_DIR", "/app/logs")
os.makedirs(LOG_DIR, exist_ok=True)
file_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "rag-api.log"),
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
logger.addHandler(file_handler)

app = FastAPI()

# 1. Configuration & Clients
COLLECTION_NAME = "factory_sops"
DIMENSION = 384  # Matches all-MiniLM-L6-v2

# Milvus URL: Use localhost:19530 when running outside Docker (Anaconda Prompt)
# Use milvus:19530 when running inside Docker container
MILVUS_URL = os.getenv("MILVUS_URL", "http://localhost:19530")
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000/v1")

client = MilvusClient(MILVUS_URL)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
llm_client = OpenAI(base_url=VLLM_URL, api_key="token-not-needed")

# 2. Initialize Milvus Schema
if not client.has_collection(COLLECTION_NAME):
    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=DIMENSION)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="machine_id", datatype=DataType.VARCHAR, max_length=100)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="IVF_FLAT", metric_type="COSINE", params={"nlist": 128})
    client.create_collection(collection_name=COLLECTION_NAME, schema=schema, index_params=index_params)

# --- FILE PARSING HELPERS ---
def extract_text_from_pdf(content: bytes) -> str:
    pdf_reader = pypdf.PdfReader(io.BytesIO(content))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_html(content: bytes) -> str:
    soup = BeautifulSoup(content, "lxml")
    # Remove script and style elements
    for element in soup(["script", "style"]):
        element.decompose()
    return soup.get_text(separator="\n", strip=True)

def extract_text_from_txt(content: bytes) -> str:
    # Try UTF-8 first, fall back to latin-1
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("latin-1")

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".html", ".htm"}

# --- CHUNKING HELPER ---
def chunk_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 200, min_chunk_size: int = 100) -> list[str]:
    """
    Structure-aware chunking that preserves JSON blocks and uses separator hierarchy.

    Args:
        text: Input text to chunk
        chunk_size: Target chunk size in characters (~400 tokens)
        chunk_overlap: Overlap between chunks in characters (~50 tokens)
        min_chunk_size: Minimum chunk size to avoid tiny fragments

    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []

    # Preserve JSON blocks by replacing them with placeholders
    json_pattern = r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\])'
    json_blocks = []

    def replace_json(match):
        json_blocks.append(match.group(0))
        return f"__JSON_BLOCK_{len(json_blocks) - 1}__"

    text_with_placeholders = re.sub(json_pattern, replace_json, text)

    # Split using separator hierarchy
    separators = ["\n\n", "\n", ". ", " "]

    def split_by_separator(text_to_split: str, sep_index: int = 0) -> list[str]:
        if sep_index >= len(separators):
            return [text_to_split] if text_to_split.strip() else []

        separator = separators[sep_index]
        parts = text_to_split.split(separator)

        result = []
        for part in parts:
            if len(part) <= chunk_size:
                if part.strip():
                    result.append(part + (separator if separator != " " else ""))
            else:
                result.extend(split_by_separator(part, sep_index + 1))
        return result

    segments = split_by_separator(text_with_placeholders)

    # Merge segments into chunks with overlap
    chunks = []
    current_chunk = ""

    for segment in segments:
        if len(current_chunk) + len(segment) <= chunk_size:
            current_chunk += segment
        else:
            if current_chunk.strip() and len(current_chunk.strip()) >= min_chunk_size:
                chunks.append(current_chunk.strip())
            # Start new chunk with overlap from previous
            if chunk_overlap > 0 and current_chunk:
                overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
                current_chunk = overlap_text + segment
            else:
                current_chunk = segment

    # Add final chunk
    if current_chunk.strip() and len(current_chunk.strip()) >= min_chunk_size:
        chunks.append(current_chunk.strip())

    # Restore JSON blocks in all chunks
    def restore_json(chunk_text: str) -> str:
        for i, json_block in enumerate(json_blocks):
            chunk_text = chunk_text.replace(f"__JSON_BLOCK_{i}__", json_block)
        return chunk_text

    chunks = [restore_json(chunk) for chunk in chunks]

    return chunks if chunks else [text.strip()[:chunk_size]]

# --- INGESTION ENDPOINT ---
@app.post("/ingest")
async def ingest_doc(machine_id: str = Form(...), file: UploadFile = File(...)):
    filename = file.filename.lower() if file.filename else ""
    logger.info(f"INGEST | machine_id={machine_id} | file={filename}")

    try:
        ext = os.path.splitext(filename)[1]

        if ext not in SUPPORTED_EXTENSIONS:
            error_msg = f"Unsupported file type: {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
            logger.warning(f"INGEST | REJECTED | {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)

        content = await file.read()

        # Extract text based on file type
        if ext == ".pdf":
            full_text = extract_text_from_pdf(content)
        elif ext in (".html", ".htm"):
            full_text = extract_text_from_html(content)
        else:  # .txt
            full_text = extract_text_from_txt(content)

        # Chunking (Structure-aware with JSON preservation, ~400 tokens per chunk)
        chunks = chunk_text(full_text, chunk_size=1500, chunk_overlap=200, min_chunk_size=100)

        # Embed and Insert
        data = []
        for chunk in chunks:
            embedding = embed_model.encode(chunk).tolist()
            data.append({
                "vector": embedding,
                "text": chunk,
                "machine_id": machine_id
            })

        client.insert(collection_name=COLLECTION_NAME, data=data)
        logger.info(f"INGEST | SUCCESS | machine_id={machine_id} | chunks={len(chunks)}")
        return {"status": "success", "chunks_ingested": len(chunks), "machine_id": machine_id, "file_type": ext}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"INGEST | ERROR | machine_id={machine_id} | {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- QUERY ENDPOINT ---
@app.post("/query")

async def query_rag(
    request: Request,
    user_query: str = Form(...),
    machine_id: str = Form(None),
    stream: bool = Form(False)
):
    query_preview = user_query[:100] + "..." if len(user_query) > 100 else user_query
    logger.info(f"QUERY | machine_id={machine_id} | stream={stream} | query={query_preview}")

    try:
        # 1. Embed user query
        query_vector = embed_model.encode(user_query).tolist()

        # 2. Search Milvus (with optional Machine ID filter to be "Smart")
        filter_expr = f"machine_id == '{machine_id}'" if machine_id else ""

        search_res = client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vector],
            filter=filter_expr,
            limit=3,
            output_fields=["text"]
        )

        # 3. Build Context
        context_chunks = [res['entity']['text'] for res in search_res[0]]
        context_text = "\n---\n".join(context_chunks)
        logger.info(f"QUERY | MILVUS | found {len(context_chunks)} chunks")

        if not context_chunks:
            context_text = "No relevant SOP found for this machine."

        # 4. Generate Answer using the LLM
        system_prompt = "You are a manufacturing assistant. Use the provided SOP context to answer the user's technical question precisely."
        prompt = f"CONTEXT FROM SOPS:\n{context_text}\n\nUSER QUESTION: {user_query}\n\nANSWER:"

        if stream:
            # Async streaming response with client disconnection detection
            async def generate():
                response = None
                tokens_generated = 0
                try:
                    response = llm_client.chat.completions.create(
                        model="openai/gpt-oss-20b",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.2,
                        stream=True
                    )
                    for chunk in response:
                        # Check if client disconnected
                        if await request.is_disconnected():
                            logger.warning(f"QUERY | CANCELLED | client disconnected after {tokens_generated} tokens")
                            if response:
                                response.close()
                            return
                        if chunk.choices[0].delta.content:
                            tokens_generated += 1
                            yield chunk.choices[0].delta.content
                    logger.info(f"QUERY | SUCCESS | stream complete, {tokens_generated} tokens")
                except GeneratorExit:
                    # Client disconnected mid-stream
                    logger.warning(f"QUERY | CANCELLED | client disconnected (GeneratorExit) after {tokens_generated} tokens")
                    if response:
                        response.close()
                except Exception as e:
                    error_msg = f"\n\n[ERROR] LLM generation failed: {type(e).__name__}: {e}"
                    logger.error(f"QUERY | ERROR | {type(e).__name__}: {e}")
                    yield error_msg

            return StreamingResponse(generate(), media_type="text/plain")
        else:
            # Non-streaming response
            try:
                response = llm_client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                )
                logger.info(f"QUERY | SUCCESS | non-stream complete")
                return {
                    "answer": response.choices[0].message.content,
                    "sources_found": len(context_chunks)
                }
            except Exception as e:
                error_msg = f"LLM generation failed: {type(e).__name__}: {e}"
                logger.error(f"QUERY | ERROR | {error_msg}")
                return {
                    "answer": f"[ERROR] {error_msg}",
                    "sources_found": len(context_chunks),
                    "error": True
                }

    except Exception as e:
        error_msg = f"Query processing failed: {type(e).__name__}: {e}"
        logger.error(f"QUERY | ERROR | {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


# --- OpenAI-Compatible RAG Endpoint ---
@app.post("/rag/{stream_param}/{machine_id}/v1/chat/completions")
async def rag_chat_completions(
    stream_param: str,
    machine_id: str,
    request: Request,
    body: ChatCompletionRequest
):
    """
    OpenAI-compatible chat completions endpoint that routes through the RAG pipeline.

    URL format: /rag/{stream_param}/{machine_id}/v1/chat/completions
    - stream_param: "stream" for streaming response, "no-stream" for non-streaming
    - machine_id: The collection/machine ID for RAG retrieval

    The last user message is used as the query for RAG retrieval.
    """
    # Validate stream_param
    if stream_param not in ("stream", "no-stream"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid stream parameter: '{stream_param}'. Must be 'stream' or 'no-stream'"
        )

    # Determine streaming from URL path (overrides body.stream)
    use_stream = (stream_param == "stream")

    # Extract the last user message as the query
    user_messages = [msg for msg in body.messages if msg.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found in request")

    user_query = user_messages[-1].content
    query_preview = user_query[:100] + "..." if len(user_query) > 100 else user_query
    logger.info(f"RAG-CHAT | machine_id={machine_id} | stream={use_stream} | query={query_preview}")

    try:
        # 1. Embed user query
        query_vector = embed_model.encode(user_query).tolist()

        # 2. Search Milvus with machine_id filter
        filter_expr = f"machine_id == '{machine_id}'" if machine_id else ""

        search_res = client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vector],
            filter=filter_expr,
            limit=3,
            output_fields=["text"]
        )

        # 3. Build Context
        context_chunks = [res['entity']['text'] for res in search_res[0]]
        context_text = "\n---\n".join(context_chunks)
        logger.info(f"RAG-CHAT | MILVUS | found {len(context_chunks)} chunks")

        if not context_chunks:
            context_text = "No relevant SOP found for this machine."

        # 4. Build messages with RAG context
        system_prompt = "You are a manufacturing assistant. Use the provided SOP context to answer the user's technical question precisely."
        rag_prompt = f"CONTEXT FROM SOPS:\n{context_text}\n\nUSER QUESTION: {user_query}\n\nANSWER:"

        messages_for_llm = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": rag_prompt}
        ]

        # Generate unique ID for this completion
        completion_id = f"chatcmpl-rag-{uuid.uuid4().hex[:12]}"
        created_timestamp = int(time.time())

        if use_stream:
            # Streaming response in SSE format (OpenAI-compatible)
            async def generate_sse():
                tokens_generated = 0
                response = None
                try:
                    response = llm_client.chat.completions.create(
                        model=body.model,
                        messages=messages_for_llm,
                        temperature=body.temperature,
                        max_tokens=body.max_tokens,
                        stream=True
                    )

                    for chunk in response:
                        # Check if client disconnected
                        if await request.is_disconnected():
                            logger.warning(f"RAG-CHAT | CANCELLED | client disconnected after {tokens_generated} tokens")
                            if response:
                                response.close()
                            return

                        delta_content = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
                        finish_reason = chunk.choices[0].finish_reason

                        # Build SSE chunk in OpenAI format
                        sse_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_timestamp,
                            "model": body.model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": delta_content} if delta_content else {},
                                "finish_reason": finish_reason
                            }]
                        }

                        if delta_content or finish_reason:
                            tokens_generated += 1
                            yield f"data: {json.dumps(sse_chunk)}\n\n"

                    # Send [DONE] marker
                    yield "data: [DONE]\n\n"
                    logger.info(f"RAG-CHAT | SUCCESS | stream complete, {tokens_generated} chunks")

                except GeneratorExit:
                    logger.warning(f"RAG-CHAT | CANCELLED | client disconnected (GeneratorExit) after {tokens_generated} tokens")
                    if response:
                        response.close()
                except Exception as e:
                    error_chunk = {
                        "error": {
                            "message": f"LLM generation failed: {type(e).__name__}: {e}",
                            "type": "server_error"
                        }
                    }
                    logger.error(f"RAG-CHAT | ERROR | {type(e).__name__}: {e}")
                    yield f"data: {json.dumps(error_chunk)}\n\n"

            return StreamingResponse(
                generate_sse(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            # Non-streaming response
            try:
                response = llm_client.chat.completions.create(
                    model=body.model,
                    messages=messages_for_llm,
                    temperature=body.temperature,
                    max_tokens=body.max_tokens
                )

                answer = response.choices[0].message.content
                logger.info(f"RAG-CHAT | SUCCESS | non-stream complete")

                return ChatCompletionResponse(
                    id=completion_id,
                    created=created_timestamp,
                    model=body.model,
                    choices=[
                        ChatCompletionChoice(
                            index=0,
                            message=ChatMessage(role="assistant", content=answer),
                            finish_reason="stop"
                        )
                    ],
                    usage=ChatCompletionUsage(
                        prompt_tokens=len(rag_prompt.split()),
                        completion_tokens=len(answer.split()),
                        total_tokens=len(rag_prompt.split()) + len(answer.split())
                    )
                )
            except Exception as e:
                error_msg = f"LLM generation failed: {type(e).__name__}: {e}"
                logger.error(f"RAG-CHAT | ERROR | {error_msg}")
                raise HTTPException(status_code=500, detail=error_msg)

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"RAG query processing failed: {type(e).__name__}: {e}"
        logger.error(f"RAG-CHAT | ERROR | {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)
