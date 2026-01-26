import os
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from google import genai

# -------------------------------------------------
# FASTAPI APP (MUST EXIST AT TOP LEVEL)
# -------------------------------------------------

app = FastAPI(
    title="Constitution RAG API",
    version="1.0"
)

# -------------------------------------------------
# GLOBAL PLACEHOLDERS (INITIALIZED ON STARTUP)
# -------------------------------------------------

embeddings = None
vectorstore = None
genai_client = None

# -------------------------------------------------
# STARTUP EVENT (CRITICAL FOR RENDER)
# -------------------------------------------------

@app.on_event("startup")
def startup_event():
    global embeddings, vectorstore, genai_client

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_pinecone import PineconeVectorStore

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not PINECONE_API_KEY or not GEMINI_API_KEY:
        raise RuntimeError("Missing required environment variables")

    # Gemini client
    genai_client = genai.Client(api_key=GEMINI_API_KEY)

    # Heavy model loading (must be here)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # Pinecone vector store
    vectorstore = PineconeVectorStore(
        index_name="constitution-rag-demo",
        embedding=embeddings,
        namespace="constitution-2024",
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
# HEALTH CHECK (IMPORTANT FOR RENDER)
# -------------------------------------------------

@app.get("/")
def health():
    return {"status": "ok"}

# -------------------------------------------------
# CHAT ENDPOINT
# -------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    docs = vectorstore.similarity_search(req.question, k=4)

    sources = [
        SourceChunk(
            page=int(d.metadata.get("page", -1)),
            content=d.page_content
        )
        for d in docs
    ]

    prompt = build_prompt(req.question, sources)

    response = genai_client.models.generate_content(
        model="models/gemini-2.5-flash-lite",
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
