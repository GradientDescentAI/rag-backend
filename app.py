import os
import traceback
from typing import List
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai

# -------------------------------------------------
# FASTAPI APP
# -------------------------------------------------

app = FastAPI(
    title="Constitution RAG API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # OK for demo; restrict later
    allow_credentials=True,
    allow_methods=["*"],   # Allows OPTIONS, POST, GET
    allow_headers=["*"],
)

# -------------------------------------------------
# GLOBALS (LAZY INIT)
# -------------------------------------------------

embeddings = None
vectorstore = None
genai_client = None
initialized = False

# -------------------------------------------------
# LAZY INITIALIZATION FUNCTION
# -------------------------------------------------

def ensure_initialized():
    global embeddings, vectorstore, genai_client, initialized

    if initialized:
        return

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_pinecone import PineconeVectorStore

        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

        if not PINECONE_API_KEY or not GEMINI_API_KEY:
            raise RuntimeError("Missing required environment variables")

        # Gemini client
        genai_client = genai.Client(api_key=GEMINI_API_KEY)

        # Embeddings (heavy)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Pinecone
        vectorstore = PineconeVectorStore(
            index_name="constitution-rag-384",
            embedding=embeddings,
            namespace="constitution-2024",
            pinecone_api_key=PINECONE_API_KEY
        )

        initialized = True
        print("✅ Lazy initialization complete")

    except Exception:
        print("❌ Lazy initialization failed")
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
# CHAT ENDPOINT
# -------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        ensure_initialized()

        docs = vectorstore.similarity_search(req.question, k=4)

        sources = []
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

        prompt = build_prompt(req.question, sources)

        response = genai_client.models.generate_content(
            model="models/gemini-2.5-flash-lite",
            contents=prompt
        )

        # Safe Gemini parsing
        if hasattr(response, "text") and response.text:
            text = response.text
        elif hasattr(response, "candidates"):
            text = response.candidates[0].content.parts[0].text
        else:
            raise RuntimeError("Gemini returned no usable text")

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
        print("❌ ERROR IN /chat")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
