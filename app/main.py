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
    stream: Optional[bool] = None  # None = use URL path, True/False = explicit
    # Additional OpenAI fields that n8n/LangChain may send
    tools: Optional[list] = None
    tool_choice: Optional[str | dict] = None
    functions: Optional[list] = None
    function_call: Optional[str | dict] = None
    n: Optional[int] = None
    stop: Optional[str | list[str]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    top_p: Optional[float] = None

    class Config:
        extra = "allow"  # Allow additional fields we don't explicitly define

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


# --- OpenAI Responses API Models ---
class ResponsesInputMessage(BaseModel):
    role: str
    content: str

class ResponsesTool(BaseModel):
    type: str  # "file_search", "web_search", "code_interpreter", "function", etc.
    # Additional fields depending on tool type (optional)

class ResponsesRequest(BaseModel):
    model: str = "openai/gpt-oss-20b"
    input: str | list[ResponsesInputMessage]  # Can be string or message list
    instructions: Optional[str] = None
    temperature: Optional[float] = 0.2
    max_output_tokens: Optional[int] = None
    stream: Optional[bool] = False
    tools: Optional[list[ResponsesTool]] = None

class ResponsesOutputContent(BaseModel):
    type: str = "output_text"
    text: str

class ResponsesOutputMessage(BaseModel):
    type: str = "message"
    id: str
    role: str = "assistant"
    content: list[ResponsesOutputContent]

class ResponsesResponse(BaseModel):
    id: str
    object: str = "response"
    created_at: int
    model: str
    output: list[ResponsesOutputMessage]
    usage: Optional[ChatCompletionUsage] = None


# Built-in OpenAI tools that are not implemented
UNSUPPORTED_OPENAI_TOOLS = {"file_search", "web_search", "code_interpreter"}

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

app = FastAPI(
    title="Manufacturing RAG API",
    description="""
RAG (Retrieval-Augmented Generation) API for manufacturing SOPs.

## Features
- **Document Ingestion**: Upload PDF, TXT, HTML files for a machine
- **Document Management**: List, delete documents by machine_id or document_id
- **RAG Query**: Query documents with LLM-powered responses
- **OpenAI-Compatible**: Chat completions and Responses API endpoints
""",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

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

# --- LIST DOCUMENTS ENDPOINT ---
@app.get("/documents")
async def list_documents(machine_id: str = None):
    """
    List all documents in the RAG system, grouped by original uploaded file.

    Query params:
        machine_id: Optional filter by machine_id

    Returns:
        List of documents with document_id, filename, machine_id, and chunk count
    """
    logger.info(f"LIST-DOCS | machine_id={machine_id}")

    try:
        # Query all chunks (or filtered by machine_id)
        filter_expr = f"machine_id == '{machine_id}'" if machine_id else ""

        # Use query to get all chunks
        results = client.query(
            collection_name=COLLECTION_NAME,
            filter=filter_expr if filter_expr else "",
            output_fields=["id", "machine_id", "text", "document_id", "filename"],
            limit=10000  # Reasonable limit
        )

        # Group chunks by document_id
        documents_map = {}

        for chunk in results:
            doc_id = chunk.get("document_id")
            if doc_id:
                if doc_id not in documents_map:
                    documents_map[doc_id] = {
                        "document_id": doc_id,
                        "filename": chunk.get("filename", "unknown"),
                        "machine_id": chunk["machine_id"],
                        "chunk_count": 0,
                        "chunk_ids": []
                    }
                documents_map[doc_id]["chunk_count"] += 1
                documents_map[doc_id]["chunk_ids"].append(chunk["id"])

        documents = list(documents_map.values())

        # Get unique machine_ids for summary
        unique_machines = list(set(chunk["machine_id"] for chunk in results)) if results else []

        logger.info(f"LIST-DOCS | SUCCESS | {len(documents)} documents, {len(results)} total chunks, {len(unique_machines)} machines")
        return {
            "total_documents": len(documents),
            "total_chunks": len(results),
            "unique_machines": unique_machines,
            "documents": documents
        }
    except Exception as e:
        logger.error(f"LIST-DOCS | ERROR | {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- DELETE DOCUMENTS ENDPOINTS ---
@app.delete("/documents/{machine_id}")
async def delete_by_machine(machine_id: str):
    """
    Delete all documents for a specific machine_id.

    Path params:
        machine_id: The machine ID to delete all documents for

    Returns:
        Number of documents deleted
    """
    logger.info(f"DELETE-DOCS | machine_id={machine_id}")

    try:
        # First count how many documents will be deleted
        results = client.query(
            collection_name=COLLECTION_NAME,
            filter=f"machine_id == '{machine_id}'",
            output_fields=["id"],
            limit=10000
        )

        count = len(results)

        if count == 0:
            logger.warning(f"DELETE-DOCS | NOT FOUND | machine_id={machine_id}")
            raise HTTPException(status_code=404, detail=f"No documents found for machine_id: {machine_id}")

        # Delete all documents with this machine_id
        client.delete(
            collection_name=COLLECTION_NAME,
            filter=f"machine_id == '{machine_id}'"
        )

        logger.info(f"DELETE-DOCS | SUCCESS | machine_id={machine_id} | deleted={count}")
        return {
            "status": "success",
            "machine_id": machine_id,
            "documents_deleted": count
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"DELETE-DOCS | ERROR | {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{machine_id}/{document_id}")
async def delete_document(machine_id: str, document_id: str):
    """
    Delete a specific document (all its chunks) by document_id for a given machine_id.

    Path params:
        machine_id: The machine ID (for verification)
        document_id: The document UUID (all chunks from the same uploaded file share this)

    Returns:
        Confirmation of deletion with chunk count
    """
    logger.info(f"DELETE-DOC | machine_id={machine_id} | document_id={document_id}")

    try:
        # Find all chunks belonging to this document_id
        results = client.query(
            collection_name=COLLECTION_NAME,
            filter=f"document_id == '{document_id}'",
            output_fields=["id", "machine_id", "filename"],
            limit=10000
        )

        if not results:
            logger.warning(f"DELETE-DOC | NOT FOUND | document_id={document_id}")
            raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

        # Verify the document belongs to this machine_id
        doc_machine_id = results[0]["machine_id"]
        if doc_machine_id != machine_id:
            logger.warning(f"DELETE-DOC | MISMATCH | document_id={document_id} belongs to {doc_machine_id}, not {machine_id}")
            raise HTTPException(
                status_code=400,
                detail=f"Document {document_id} belongs to machine_id '{doc_machine_id}', not '{machine_id}'"
            )

        filename = results[0].get("filename", "unknown")
        chunk_count = len(results)

        # Delete all chunks with this document_id
        client.delete(
            collection_name=COLLECTION_NAME,
            filter=f"document_id == '{document_id}'"
        )

        logger.info(f"DELETE-DOC | SUCCESS | machine_id={machine_id} | document_id={document_id} | filename={filename} | chunks_deleted={chunk_count}")
        return {
            "status": "success",
            "machine_id": machine_id,
            "document_id": document_id,
            "filename": filename,
            "chunks_deleted": chunk_count
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"DELETE-DOC | ERROR | {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- INGESTION ENDPOINT ---
@app.post("/ingest")
async def ingest_doc(machine_id: str = Form(...), file: UploadFile = File(...)):
    original_filename = file.filename if file.filename else "unknown"
    filename_lower = original_filename.lower()
    logger.info(f"INGEST | machine_id={machine_id} | file={original_filename}")

    try:
        ext = os.path.splitext(filename_lower)[1]

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

        # Generate unique document_id for this upload (all chunks share this ID)
        document_id = str(uuid.uuid4())

        # Embed and Insert
        data = []
        for chunk in chunks:
            embedding = embed_model.encode(chunk).tolist()
            data.append({
                "vector": embedding,
                "text": chunk,
                "machine_id": machine_id,
                "document_id": document_id,
                "filename": original_filename
            })

        client.insert(collection_name=COLLECTION_NAME, data=data)
        logger.info(f"INGEST | SUCCESS | machine_id={machine_id} | document_id={document_id} | chunks={len(chunks)}")
        return {
            "status": "success",
            "document_id": document_id,
            "filename": original_filename,
            "chunks_ingested": len(chunks),
            "machine_id": machine_id,
            "file_type": ext
        }
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

    # Debug: Log the full request body
    logger.info(f"RAG-CHAT | DEBUG REQUEST | {body.model_dump_json()}")

    # Determine streaming: body.stream takes precedence if explicitly set, otherwise use URL
    # This allows n8n to control streaming via the API request
    if body.stream is not None:
        use_stream = body.stream
        logger.info(f"RAG-CHAT | streaming from body.stream={body.stream}")
    else:
        use_stream = (stream_param == "stream")
        logger.info(f"RAG-CHAT | streaming from URL param={stream_param}")

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
            # Format per OpenAI spec:
            # 1. First chunk: delta: {"role": "assistant"} (no content field)
            # 2. Content chunks: delta: {"content": "..."}
            # 3. Final chunk: delta: {} with finish_reason: "stop"
            # 4. End with: data: [DONE]
            async def generate_sse():
                tokens_generated = 0
                response = None
                first_chunk = True
                sent_finish = False
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

                        # First chunk: send role only (no content field)
                        if first_chunk:
                            first_chunk = False
                            role_chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created_timestamp,
                                "model": body.model,
                                "system_fingerprint": "fp_rag",
                                "choices": [{
                                    "index": 0,
                                    "delta": {"role": "assistant"},
                                    "logprobs": None,
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(role_chunk)}\n\n"
                            tokens_generated += 1

                        # Content chunks: send content only
                        if delta_content:
                            content_chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created_timestamp,
                                "model": body.model,
                                "system_fingerprint": "fp_rag",
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": delta_content},
                                    "logprobs": None,
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(content_chunk)}\n\n"
                            tokens_generated += 1

                        # Track if we've seen finish_reason
                        if finish_reason:
                            sent_finish = True

                    # Final chunk: empty delta with finish_reason
                    finish_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_timestamp,
                        "model": body.model,
                        "system_fingerprint": "fp_rag",
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "logprobs": None,
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(finish_chunk)}\n\n"
                    tokens_generated += 1

                    # Send [DONE] marker - REQUIRED
                    yield "data: [DONE]\n\n"
                    logger.info(f"RAG-CHAT | SUCCESS | stream complete, {tokens_generated} chunks")

                except GeneratorExit:
                    logger.warning(f"RAG-CHAT | CANCELLED | client disconnected (GeneratorExit) after {tokens_generated} tokens")
                    if response:
                        response.close()
                    # Still send [DONE] on client disconnect
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    logger.error(f"RAG-CHAT | ERROR | {type(e).__name__}: {e}")
                    # Send error and [DONE]
                    error_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_timestamp,
                        "model": body.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": f"\n\n[Error: {type(e).__name__}]"},
                            "logprobs": None,
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    # Send finish and [DONE] even on error
                    finish_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_timestamp,
                        "model": body.model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "logprobs": None,
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(finish_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
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


# --- OpenAI Responses API RAG Endpoint ---
@app.post("/rag/{stream_param}/{machine_id}/v1/responses")
async def rag_responses(
    stream_param: str,
    machine_id: str,
    request: Request,
    body: ResponsesRequest
):
    """
    OpenAI Responses API-compatible endpoint that routes through the RAG pipeline.

    URL format: /rag/{stream_param}/{machine_id}/v1/responses
    - stream_param: "stream" for streaming response, "no-stream" for non-streaming
    - machine_id: The collection/machine ID for RAG retrieval

    Note: Built-in OpenAI tools (file_search, web_search, code_interpreter) are not
    implemented. Use n8n's native tool integrations for these capabilities.
    """
    # Validate stream_param
    if stream_param not in ("stream", "no-stream"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid stream parameter: '{stream_param}'. Must be 'stream' or 'no-stream'"
        )

    # Check for unsupported OpenAI built-in tools
    if body.tools:
        unsupported_requested = [
            tool.type for tool in body.tools if tool.type in UNSUPPORTED_OPENAI_TOOLS
        ]
        if unsupported_requested:
            error_msg = f"Tool not implemented: {', '.join(unsupported_requested)}. " \
                        f"These are OpenAI-hosted tools not available on this endpoint. " \
                        f"Use n8n's native tool integrations (SerpAPI, Google Search, etc.) instead."
            logger.warning(f"RAG-RESPONSES | REJECTED | unsupported tools: {unsupported_requested}")
            raise HTTPException(status_code=501, detail=error_msg)

    # Determine streaming from URL path (overrides body.stream)
    use_stream = (stream_param == "stream")

    # Extract user query from input (string or last user message)
    if isinstance(body.input, str):
        user_query = body.input
    else:
        user_messages = [m for m in body.input if m.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found in input")
        user_query = user_messages[-1].content

    query_preview = user_query[:100] + "..." if len(user_query) > 100 else user_query
    logger.info(f"RAG-RESPONSES | machine_id={machine_id} | stream={use_stream} | query={query_preview}")

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
        logger.info(f"RAG-RESPONSES | MILVUS | found {len(context_chunks)} chunks")

        if not context_chunks:
            context_text = "No relevant SOP found for this machine."

        # 4. Build messages with RAG context
        system_prompt = body.instructions or "You are a manufacturing assistant. Use the provided SOP context to answer the user's technical question precisely."
        rag_prompt = f"CONTEXT FROM SOPS:\n{context_text}\n\nUSER QUESTION: {user_query}\n\nANSWER:"

        messages_for_llm = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": rag_prompt}
        ]

        # Generate unique IDs for this response
        response_id = f"resp-rag-{uuid.uuid4().hex[:12]}"
        message_id = f"msg-{uuid.uuid4().hex[:12]}"
        created_timestamp = int(time.time())

        if use_stream:
            # Streaming response in SSE format (Responses API format)
            # OpenAI Responses API uses "event:" prefix before "data:" lines
            async def generate_sse():
                tokens_generated = 0
                full_text = ""
                response = None
                try:
                    # Send initial response.created event
                    created_event = {
                        "type": "response.created",
                        "response": {
                            "id": response_id,
                            "object": "response",
                            "created_at": created_timestamp,
                            "model": body.model,
                            "status": "in_progress"
                        }
                    }
                    yield f"event: response.created\ndata: {json.dumps(created_event)}\n\n"

                    # Send output_item.added event
                    item_added_event = {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {
                            "type": "message",
                            "id": message_id,
                            "role": "assistant",
                            "content": []
                        }
                    }
                    yield f"event: response.output_item.added\ndata: {json.dumps(item_added_event)}\n\n"

                    # Send content_part.added event
                    content_part_event = {
                        "type": "response.content_part.added",
                        "output_index": 0,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": ""
                        }
                    }
                    yield f"event: response.content_part.added\ndata: {json.dumps(content_part_event)}\n\n"

                    response = llm_client.chat.completions.create(
                        model=body.model,
                        messages=messages_for_llm,
                        temperature=body.temperature,
                        max_tokens=body.max_output_tokens,
                        stream=True
                    )

                    for chunk in response:
                        # Check if client disconnected
                        if await request.is_disconnected():
                            logger.warning(f"RAG-RESPONSES | CANCELLED | client disconnected after {tokens_generated} tokens")
                            if response:
                                response.close()
                            return

                        delta_content = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
                        finish_reason = chunk.choices[0].finish_reason

                        if delta_content:
                            tokens_generated += 1
                            full_text += delta_content
                            # Send content delta event
                            delta_event = {
                                "type": "response.output_text.delta",
                                "output_index": 0,
                                "content_index": 0,
                                "delta": delta_content
                            }
                            yield f"event: response.output_text.delta\ndata: {json.dumps(delta_event)}\n\n"

                    # Send output_text.done event
                    text_done_event = {
                        "type": "response.output_text.done",
                        "output_index": 0,
                        "content_index": 0,
                        "text": full_text
                    }
                    yield f"event: response.output_text.done\ndata: {json.dumps(text_done_event)}\n\n"

                    # Send output_item.done event
                    item_done_event = {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {
                            "type": "message",
                            "id": message_id,
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": full_text}]
                        }
                    }
                    yield f"event: response.output_item.done\ndata: {json.dumps(item_done_event)}\n\n"

                    # Send response.completed event
                    completed_event = {
                        "type": "response.completed",
                        "response": {
                            "id": response_id,
                            "object": "response",
                            "created_at": created_timestamp,
                            "model": body.model,
                            "status": "completed",
                            "output": [{
                                "type": "message",
                                "id": message_id,
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": full_text}]
                            }],
                            "usage": {
                                "input_tokens": len(rag_prompt.split()),
                                "output_tokens": tokens_generated,
                                "total_tokens": len(rag_prompt.split()) + tokens_generated
                            }
                        }
                    }
                    yield f"event: response.completed\ndata: {json.dumps(completed_event)}\n\n"

                    # Send [DONE] marker (no event prefix for this one)
                    yield "data: [DONE]\n\n"
                    logger.info(f"RAG-RESPONSES | SUCCESS | stream complete, {tokens_generated} chunks")

                except GeneratorExit:
                    logger.warning(f"RAG-RESPONSES | CANCELLED | client disconnected (GeneratorExit) after {tokens_generated} tokens")
                    if response:
                        response.close()
                except Exception as e:
                    error_event = {
                        "type": "error",
                        "error": {
                            "message": f"LLM generation failed: {type(e).__name__}: {e}",
                            "type": "server_error"
                        }
                    }
                    logger.error(f"RAG-RESPONSES | ERROR | {type(e).__name__}: {e}")
                    yield f"event: error\ndata: {json.dumps(error_event)}\n\n"

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
                    max_tokens=body.max_output_tokens
                )

                answer = response.choices[0].message.content
                logger.info(f"RAG-RESPONSES | SUCCESS | non-stream complete")

                return ResponsesResponse(
                    id=response_id,
                    created_at=created_timestamp,
                    model=body.model,
                    output=[
                        ResponsesOutputMessage(
                            id=message_id,
                            content=[ResponsesOutputContent(text=answer)]
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
                logger.error(f"RAG-RESPONSES | ERROR | {error_msg}")
                raise HTTPException(status_code=500, detail=error_msg)

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"RAG query processing failed: {type(e).__name__}: {e}"
        logger.error(f"RAG-RESPONSES | ERROR | {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)
