import os
import traceback
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai

# -------------------------------------------------
# FASTAPI APP
# -------------------------------------------------

app = FastAPI(
    title="Legal Research RAG API",
    version="2.1",
    description=(
        "Source-bound legal research and analytical interpretation "
        "derived exclusively from authoritative legal texts."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # OK for demo; restrict later
    allow_credentials=True,
    allow_methods=["*"],
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
# LAZY INITIALIZATION
# -------------------------------------------------

def ensure_initialized():
    """
    Initializes embeddings, vector store, and Gemini client once.
    Preserves existing behaviour and avoids cold-start penalties.
    """
    global embeddings, vectorstore, genai_client, initialized

    if initialized:
        return

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_pinecone import PineconeVectorStore

        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

        if not PINECONE_API_KEY or not GEMINI_API_KEY:
            raise RuntimeError("Required environment variables are missing")

        # Gemini client
        genai_client = genai.Client(api_key=GEMINI_API_KEY)

        # Embeddings (UNCHANGED)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        # Pinecone Vector Store (UNCHANGED)
        vectorstore = PineconeVectorStore(
            index_name="constitution-rag-demo",
            embedding=embeddings,
            namespace="constitution-2024",
            pinecone_api_key=PINECONE_API_KEY,
        )

        initialized = True
        print("✅ Legal RAG system initialized")

    except Exception:
        print("❌ Initialization failed")
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

class RAGBlock(BaseModel):
    answer: str
    sources: List[SourceChunk]

class ChatResponse(BaseModel):
    pure_rag: RAGBlock
    reasoned_interpretation: RAGBlock

# -------------------------------------------------
# PROMPT BUILDERS (LANGUAGE-TIGHTENED)
# -------------------------------------------------

def build_pure_rag_prompt(question: str, sources: List[SourceChunk]) -> str:
    """
    Conservative, extractive, text-bound response.
    No inference beyond the material.
    """

    context = "\n\n".join(
        [f"(Page {s.page}) {s.content}" for s in sources]
    )

    return f"""
You are a legal research assistant.

Instructions:
- Confine your response strictly to the extracted material below.
- Do not infer, speculate, or introduce external legal principles.
- Maintain a formal legal research tone.
- If the material does not explicitly address the question, state:
  "The document does not contain this information."

Extracted Material:
{context}

Research Question:
{question}

Respond in the following format:

Answer:
<response>
"""

def build_reasoned_prompt(question: str, sources: List[SourceChunk]) -> str:
    """
    Controlled analytical interpretation.
    Reasoning must remain fully source-bound.
    """

    context = "\n\n".join(
        [f"(Page {s.page}) {s.content}" for s in sources]
    )

    return f"""
You are assisting in legal analysis based on authoritative source material.

Instructions:
- Derive analytical reasoning exclusively from the extracted passages.
- Do not rely on external statutes, case law, or doctrinal sources.
- Articulate the logic implicit in the material using formal legal reasoning.
- Where the text does not fully support a conclusion, explicitly state the limitation.
- Do not quote verbatim; summarize analytically.

Research Question:
{question}

Authoritative Source Material:
{context}

Task:
Provide a reasoned analytical interpretation strictly derived from the material above.
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

        # -------------------------------
        # RETRIEVAL (UNCHANGED)
        # -------------------------------
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

        # -------------------------------
        # BLOCK 1: SOURCE-BOUND ANSWER
        # -------------------------------
        pure_prompt = build_pure_rag_prompt(req.question, sources)

        pure_response = genai_client.models.generate_content(
            model="models/gemini-2.5-flash-lite",
            contents=pure_prompt
        )

        if hasattr(pure_response, "text") and pure_response.text:
            pure_text = pure_response.text
        else:
            pure_text = pure_response.candidates[0].content.parts[0].text

        if "Answer:" in pure_text:
            pure_answer = pure_text.split("Answer:")[1].strip()
        else:
            pure_answer = pure_text.strip()

        # -------------------------------
        # BLOCK 2: ANALYTICAL INTERPRETATION
        # -------------------------------
        reasoned_prompt = build_reasoned_prompt(req.question, sources)

        try:
            reasoned_response = genai_client.models.generate_content(
                model="models/gemini-2.5-flash-lite",
                contents=reasoned_prompt
            )

            if hasattr(reasoned_response, "text") and reasoned_response.text:
                reasoned_answer = reasoned_response.text.strip()
            else:
                reasoned_answer = (
                    reasoned_response.candidates[0]
                    .content.parts[0]
                    .text.strip()
                )

        except Exception:
            reasoned_answer = (
                "The extracted material does not provide sufficient basis "
                "for a reasoned analytical interpretation without introducing "
                "external legal considerations."
            )

        # -------------------------------
        # FINAL RESPONSE (UNCHANGED SHAPE)
        # -------------------------------
        return ChatResponse(
            pure_rag=RAGBlock(
                answer=pure_answer,
                sources=sources
            ),
            reasoned_interpretation=RAGBlock(
                answer=reasoned_answer,
                sources=sources
            )
        )

    except Exception as e:
        print("❌ Error in /chat endpoint")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
