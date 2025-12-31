import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import pypdf
import io
from bs4 import BeautifulSoup

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
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dimension=DIMENSION)
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

# --- INGESTION ENDPOINT ---
@app.post("/ingest")
async def ingest_doc(machine_id: str = Form(...), file: UploadFile = File(...)):
    try:
        filename = file.filename.lower() if file.filename else ""
        ext = os.path.splitext(filename)[1]

        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
            )

        content = await file.read()

        # Extract text based on file type
        if ext == ".pdf":
            full_text = extract_text_from_pdf(content)
        elif ext in (".html", ".htm"):
            full_text = extract_text_from_html(content)
        else:  # .txt
            full_text = extract_text_from_txt(content)

        # Chunking (Simple 500-word chunks with overlap)
        words = full_text.split()
        chunks = [" ".join(words[i:i+500]) for i in range(0, len(words), 400)]

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
        return {"status": "success", "chunks_ingested": len(chunks), "machine_id": machine_id, "file_type": ext}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- QUERY ENDPOINT ---
@app.post("/query")
async def query_rag(user_query: str = Form(...), machine_id: str = Form(None)):
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

    if not context_chunks:
        context_text = "No relevant SOP found for this machine."

    # 4. Generate Answer using the 20B LLM
    system_prompt = "You are a manufacturing assistant. Use the provided SOP context to answer the user's technical question precisely."
    prompt = f"CONTEXT FROM SOPS:\n{context_text}\n\nUSER QUESTION: {user_query}\n\nANSWER:"

    response = llm_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2 # Keep it factual for manufacturing
    )

    return {
        "answer": response.choices[0].message.content,
        "sources_found": len(context_chunks)
    }
