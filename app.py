import os
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from google import genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# -------------------------------------------------
# ENVIRONMENT VARIABLES (provided by Render)
# -------------------------------------------------

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

INDEX_NAME = "constitution-rag-demo"
NAMESPACE = "constitution-2024"

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
GEMINI_MODEL = "models/gemini-2.5-flash-lite"
TOP_K = 4

# -------------------------------------------------
# FASTAPI APP  ✅ THIS IS WHAT WAS MISSING
# -------------------------------------------------

app = FastAPI(
    title="Constitution RAG API",
    version="1.0"
)

# -------------------------------------------------
# CLIENT INITIALIZATION
# -------------------------------------------------

# Gemini client
genai_client = genai.Client(api_key=GEMINI_API_KEY)

# Embeddings (local, no API calls)
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL
)

# Pinecone vector store
vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
    namespace=NAMESPACE,
    pinecone_api_key=PINECONE_API_KEY
)

# -------------------------------------------------
# DATA MODELS
# -------------------------------------------------

class ChatRequest(BaseModel):
    question: str

class SourceChunk(BaseModel):
    page: int
    content: str

class ChatResponse(BaseModel):
    answer: str
    reasoning: str
    sources: List[SourceChunk]

# -------------------------------------------------
# PROMPT BUILDER
# -------------------------------------------------

def build_prompt(question: str, sources: List[SourceChunk]) -> str:
    context = "\n\n".join(
        [f"(Page {s.page}) {s.content}" for s in sources]
    )

    return f"""
You are a constitutional law assistant.

Rules:
- Answer ONLY from the provided context.
- Do NOT use outside knowledge.
- If the answer is not present, say:
  "The document does not contain this information."

Context:
{context}

Question:
{question}

Respond in this format:

Answer:
<answer>

Reasoning:
<reasoning>
"""

# -------------------------------------------------
# API ENDPOINTS
# -------------------------------------------------

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    docs = vectorstore.similarity_search(req.question, k=TOP_K)

    sources = [
        SourceChunk(
            page=int(d.metadata.get("page", -1)),
            content=d.page_content
        )
        for d in docs
    ]

    prompt = build_prompt(req.question, sources)

    response = genai_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    text = response.text or ""

    answer = ""
    reasoning = ""

    if "Answer:" in text and "Reasoning:" in text:
        answer = text.split("Answer:")[1].split("Reasoning:")[0].strip()
        reasoning = text.split("Reasoning:")[1].strip()
    else:
        answer = text.strip()
        reasoning = "Derived directly from the retrieved text."

    return ChatResponse(
        answer=answer,
        reasoning=reasoning,
        sources=sources
    )
