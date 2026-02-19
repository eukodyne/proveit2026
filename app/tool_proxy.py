"""
OpenAI-compatible chat completions proxy with full tool-calling support.

Runs on port 8081 alongside the main RAG API (port 8080).

Endpoints:
  POST /v1/chat/completions                         - Pure LLM proxy (no RAG)
  POST /rag/{machine_id}/v1/chat/completions         - RAG-augmented with tool support
  GET  /v1/models                                    - List available models
  GET  /health                                       - Health check
"""

import os
import json
import time
import uuid
import logging
import sys
from typing import Optional
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# --- Logging ---
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger("tool-proxy")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
logger.addHandler(console_handler)

LOG_DIR = os.getenv("LOG_DIR", "/app/logs")
os.makedirs(LOG_DIR, exist_ok=True)
file_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "tool-proxy.log"),
    maxBytes=10 * 1024 * 1024, backupCount=5,
)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
logger.addHandler(file_handler)


# --- Pydantic models (full OpenAI tool-calling schema) ---

class FunctionParameters(BaseModel):
    """JSON Schema for function parameters."""
    type: Optional[str] = "object"
    properties: Optional[dict] = None
    required: Optional[list[str]] = None

    class Config:
        extra = "allow"


class FunctionDef(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict] = None


class ToolDef(BaseModel):
    type: str = "function"
    function: FunctionDef


class ToolCallFunction(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: ToolCallFunction


class ChatMessage(BaseModel):
    role: str                                       # "system"|"user"|"assistant"|"tool"
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None     # assistant requesting tool use
    tool_call_id: Optional[str] = None              # tool result referencing a call
    name: Optional[str] = None

    class Config:
        extra = "allow"


class ChatCompletionRequest(BaseModel):
    model: str = "openai/gpt-oss-20b"
    messages: list[ChatMessage]
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    tools: Optional[list[ToolDef]] = None
    tool_choice: Optional[str | dict] = None
    n: Optional[int] = None
    stop: Optional[str | list[str]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    top_p: Optional[float] = None

    class Config:
        extra = "allow"


class ResponseToolCall(BaseModel):
    id: str
    type: str = "function"
    function: ToolCallFunction


class ResponseMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[list[ResponseToolCall]] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ResponseMessage
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


# --- App & clients ---

app = FastAPI(
    title="Tool-Calling Proxy",
    description=(
        "OpenAI-compatible /v1/chat/completions proxy with full tool-calling "
        "support and optional RAG augmentation.  Port 8081."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

COLLECTION_NAME = "factory_sops"
DIMENSION = 384

MILVUS_URL = os.getenv("MILVUS_URL", "http://localhost:19530")
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000/v1")

milvus_client = MilvusClient(MILVUS_URL)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
llm_client = OpenAI(base_url=VLLM_URL, api_key="token-not-needed")


# --- Helpers ---

def messages_to_dicts(messages: list[ChatMessage]) -> list[dict]:
    """Convert Pydantic ChatMessage list to plain dicts for the OpenAI SDK."""
    result = []
    for msg in messages:
        d: dict = {"role": msg.role}
        if msg.content is not None:
            d["content"] = msg.content
        elif msg.role in ("user", "system", "tool"):
            # OpenAI SDK requires content for these roles
            d["content"] = ""
        if msg.tool_calls:
            d["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        if msg.tool_call_id:
            d["tool_call_id"] = msg.tool_call_id
        if msg.name:
            d["name"] = msg.name
        result.append(d)
    return result


def build_llm_kwargs(body: ChatCompletionRequest, messages: list[dict]) -> dict:
    """Build kwargs for llm_client.chat.completions.create(), forwarding tools."""
    kwargs: dict = {
        "model": body.model,
        "messages": messages,
        "temperature": body.temperature,
        "stream": bool(body.stream),
    }
    if body.max_tokens is not None:
        kwargs["max_tokens"] = body.max_tokens
    if body.tools:
        kwargs["tools"] = [t.model_dump() for t in body.tools]
    if body.tool_choice is not None:
        kwargs["tool_choice"] = body.tool_choice
    if body.stop is not None:
        kwargs["stop"] = body.stop
    if body.top_p is not None:
        kwargs["top_p"] = body.top_p
    if body.presence_penalty is not None:
        kwargs["presence_penalty"] = body.presence_penalty
    if body.frequency_penalty is not None:
        kwargs["frequency_penalty"] = body.frequency_penalty
    return kwargs


def retrieve_rag_context(query: str, machine_id: str) -> str:
    """Search Milvus and return concatenated SOP chunks."""
    query_vector = embed_model.encode(query).tolist()
    filter_expr = f"machine_id == '{machine_id}'" if machine_id else ""
    search_res = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        filter=filter_expr,
        limit=3,
        output_fields=["text"],
    )
    chunks = [hit["entity"]["text"] for hit in search_res[0]]
    logger.info(f"RAG | MILVUS | machine_id={machine_id} | {len(chunks)} chunks")
    if not chunks:
        return "No relevant SOP found for this machine."
    return "\n---\n".join(chunks)


def inject_rag_context(messages: list[dict], context_text: str) -> list[dict]:
    """Prepend or augment a system message with RAG context."""
    rag_preamble = (
        "You are a manufacturing assistant. Use the provided SOP context to "
        "answer the user's technical question precisely.\n\n"
        f"CONTEXT FROM SOPS:\n{context_text}"
    )
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = rag_preamble + "\n\n" + (messages[0]["content"] or "")
    else:
        messages.insert(0, {"role": "system", "content": rag_preamble})
    return messages


def _serialize_tool_call_delta(tc) -> dict:
    """Serialize one streaming ChoiceDeltaToolCall to a JSON-safe dict."""
    d: dict = {"index": tc.index}
    if tc.id is not None:
        d["id"] = tc.id
    if tc.type is not None:
        d["type"] = tc.type
    if tc.function is not None:
        fn: dict = {}
        if tc.function.name is not None:
            fn["name"] = tc.function.name
        if tc.function.arguments is not None:
            fn["arguments"] = tc.function.arguments
        if fn:
            d["function"] = fn
    return d


# --- Shared completion handler ---

async def handle_completion(
    body: ChatCompletionRequest,
    messages: list[dict],
    request: Request,
    label: str,
):
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())
    llm_kwargs = build_llm_kwargs(body, messages)

    if body.stream:
        return _handle_streaming(llm_kwargs, body.model, completion_id, created_ts, request, label)
    else:
        return _handle_non_streaming(llm_kwargs, body.model, completion_id, created_ts, label)


def _handle_non_streaming(llm_kwargs: dict, model: str, completion_id: str, created_ts: int, label: str):
    try:
        response = llm_client.chat.completions.create(**llm_kwargs)
    except Exception as e:
        logger.error(f"{label} | ERROR | {type(e).__name__}: {e}")
        raise HTTPException(status_code=502, detail=f"LLM backend error: {type(e).__name__}: {e}")

    choice = response.choices[0]
    msg = choice.message

    resp_msg = ResponseMessage(role="assistant")
    resp_msg.content = msg.content

    if hasattr(msg, "tool_calls") and msg.tool_calls:
        resp_msg.tool_calls = [
            ResponseToolCall(
                id=tc.id,
                type=tc.type,
                function=ToolCallFunction(
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                ),
            )
            for tc in msg.tool_calls
        ]

    finish_reason = choice.finish_reason or "stop"
    logger.info(f"{label} | SUCCESS | finish_reason={finish_reason}")

    return ChatCompletionResponse(
        id=completion_id,
        created=created_ts,
        model=model,
        choices=[
            ChatCompletionChoice(index=0, message=resp_msg, finish_reason=finish_reason)
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=getattr(response.usage, "prompt_tokens", 0),
            completion_tokens=getattr(response.usage, "completion_tokens", 0),
            total_tokens=getattr(response.usage, "total_tokens", 0),
        )
        if response.usage
        else None,
    )


def _handle_streaming(llm_kwargs: dict, model: str, completion_id: str, created_ts: int, request: Request, label: str):
    async def generate_sse():
        chunks_sent = 0
        response = None
        try:
            response = llm_client.chat.completions.create(**llm_kwargs)

            for chunk in response:
                if await request.is_disconnected():
                    logger.warning(f"{label} | CANCELLED | disconnected after {chunks_sent} chunks")
                    if response:
                        response.close()
                    return

                choice = chunk.choices[0]
                delta = choice.delta
                finish_reason = choice.finish_reason

                # Build delta dict
                delta_dict: dict = {}
                if hasattr(delta, "role") and delta.role:
                    delta_dict["role"] = delta.role
                if hasattr(delta, "content") and delta.content is not None:
                    delta_dict["content"] = delta.content
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    delta_dict["tool_calls"] = [
                        _serialize_tool_call_delta(tc) for tc in delta.tool_calls
                    ]

                sse_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta_dict,
                            "logprobs": None,
                            "finish_reason": finish_reason,
                        }
                    ],
                }
                yield f"data: {json.dumps(sse_chunk)}\n\n"
                chunks_sent += 1

            yield "data: [DONE]\n\n"
            logger.info(f"{label} | SUCCESS | stream complete, {chunks_sent} chunks")

        except GeneratorExit:
            logger.warning(f"{label} | CANCELLED | GeneratorExit after {chunks_sent} chunks")
            if response:
                response.close()
        except Exception as e:
            logger.error(f"{label} | ERROR | {type(e).__name__}: {e}")
            error_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"\n\n[Error: {type(e).__name__}: {e}]"},
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --- Endpoints ---

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    """
    Pure LLM proxy with full tool-calling support (no RAG).

    Forwards messages, tools, and tool_choice to the vLLM backend as-is.
    Returns tool_calls in the response when the model requests them,
    with finish_reason="tool_calls".
    """
    logger.info(
        f"PROXY | model={body.model} | stream={body.stream} "
        f"| tools={len(body.tools) if body.tools else 0} "
        f"| messages={len(body.messages)}"
    )
    messages = messages_to_dicts(body.messages)
    return await handle_completion(body, messages, request, "PROXY")


@app.post("/rag/{machine_id}/v1/chat/completions")
async def rag_chat_completions(
    machine_id: str, request: Request, body: ChatCompletionRequest
):
    """
    RAG-augmented chat completions with full tool-calling support.

    - Retrieves SOP context from Milvus using the last user message + machine_id
    - Injects context into the system prompt
    - Preserves the full conversation history (including tool results)
    - Forwards tools/tool_choice to vLLM
    - Returns tool_calls with finish_reason="tool_calls" when the model requests them
    """
    user_messages = [msg for msg in body.messages if msg.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found in request")

    user_query = user_messages[-1].content or ""
    query_preview = user_query[:100] + ("..." if len(user_query) > 100 else "")
    logger.info(
        f"RAG-TOOL | machine_id={machine_id} | stream={body.stream} "
        f"| tools={len(body.tools) if body.tools else 0} "
        f"| query={query_preview}"
    )

    context_text = retrieve_rag_context(user_query, machine_id)
    messages = messages_to_dicts(body.messages)
    messages = inject_rag_context(messages, context_text)

    return await handle_completion(body, messages, request, "RAG-TOOL")


@app.get("/v1/models")
@app.get("/rag/{machine_id}/v1/models")
async def list_models(machine_id: str = None):
    """Proxy model list from the vLLM backend."""
    try:
        models = llm_client.models.list()
        return {
            "object": "list",
            "data": [{"id": m.id, "object": "model"} for m in models],
        }
    except Exception as e:
        logger.error(f"MODELS | ERROR | {type(e).__name__}: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "port": 8081}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8081)
