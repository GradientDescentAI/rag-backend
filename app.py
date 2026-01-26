import os
import traceback
from typing import List

from fastapi import FastAPI, HTTPException
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
# STARTUP EVENT (RENDER-SAFE)
# -------------------------------------------------

@app.on_event("startup")
def startup_event():
    global embeddings, vectorstore, genai_client

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_pinecone import PineconeVectorStore

        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

        if not PINECONE_API_KEY or not GEMINI_API_KEY:
            raise RuntimeError("Missing required environment variables")

        # Gemini client
        genai_client = genai.Client(api_key=GEMINI_API_KEY)

        # Heavy model load (do NOT move to top-level)
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

        print("✅ Startup complete: Gemini + Pinecone ready")

    except Exception:
        print("❌ Startup failed")
        traceback.print_exc()
        raise

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
# HEALTH CHECK
# -------------------------------------------------

@app.get("/")
def health():
    return {"status": "ok"}

# -------------------------------------------------
# CHAT ENDPOINT (ROBUST + DEBUGGABLE)
# -------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        if vectorstore is None or genai_client is None:
            raise RuntimeError("Backend not initialized")

        # 1. Retrieve documents
        docs = vectorstore.similarity_search(req.question, k=4)

        sources: List[SourceChunk] = []

        for d in docs:
            page_raw = d.metadata.get("page", -1)
            try:
                page = int(float(page_raw))
            except Exception:
                page = -1

            sources.append(
                SourceChunk(
                    page=page,
                    content=d.page_content
                )
            )

        # 2. Build prompt
        prompt = build_prompt(req.question, sources)

        # 3. Gemini call
        response = genai_client.models.generate_content(
            model="models/gemini-2.5-flash-lite",
            contents=prompt
        )

        # 4. SAFE text extraction (Gemini SDK varies)
        text = ""
        if hasattr(response, "text") and response.text:
            text = response.text
        elif hasattr(response, "candidates"):
            text = response.candidates[0].content.parts[0].text
        else:
            raise RuntimeError("Gemini returned no usable text")

        # 5. Parse structured output
        if "Answer:" in text and "Reasoning:" in text:
            answer = text.split("Answer:")[1].split("Reasoning:")[0].strip()
            reasoning = text.split("Reasoning:")[1].strip()
        else:
            answer = text.strip()
            reasoning = "Derived directly from the retrieved constitutional text."

        return ChatResponse(
            answer=answer,
            reasoning=reasoning,
            sources=sources
        )

    except Exception as e:
        print("❌ ERROR IN /chat ENDPOINT")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
