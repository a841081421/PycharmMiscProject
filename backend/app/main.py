from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from typing import List
from pathlib import Path
from uuid import uuid4
from datetime import datetime

from .config import (
    UPLOAD_DIR, CHROMA_DIR, METADATA_FILE,
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
)
from .schemas import WebIngestRequest, ChatRequest, ChatResponse, SourceItem
from .services.document_loader import DocumentLoader
from .services.splitter import TextSplitter
from .services.vector_store import VectorStore
from .services.llm_client import LLMClient
from .services.rag_service import RAGService
from .services.metadata_store import MetadataStore

app = FastAPI(title="Local KB QA", version="0.1.0")

splitter = TextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
vector_store = VectorStore(str(CHROMA_DIR), EMBEDDING_MODEL)
metadata_store = MetadataStore(METADATA_FILE)
llm_client = LLMClient(LLM_API_KEY, LLM_BASE_URL, LLM_MODEL)
rag_service = RAGService(vector_store, llm_client, TOP_K)


def ingest_text(doc_id: str, doc_name: str, source: str, text: str, doc_type: str):
    chunks = splitter.split(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="文档内容为空，无法切片")

    ids = [f"{doc_id}:{i}" for i in range(len(chunks))]
    metas = [
        {
            "doc_id": doc_id,
            "doc_name": doc_name,
            "source": source,
            "chunk_index": i,
            "doc_type": doc_type
        }
        for i in range(len(chunks))
    ]

    vector_store.add_chunks(chunks, metas, ids)

    metadata_store.add_document({
        "doc_id": doc_id,
        "doc_name": doc_name,
        "source": source,
        "doc_type": doc_type,
        "chunk_count": len(chunks),
        "created_at": datetime.now().isoformat()
    })


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        doc_id = str(uuid4())
        save_path = UPLOAD_DIR / f"{doc_id}_{file.filename}"

        content = await file.read()
        save_path.write_bytes(content)

        loaded = DocumentLoader.load_file(Path(save_path))
        ingest_text(
            doc_id=doc_id,
            doc_name=loaded["doc_name"],
            source=loaded["source"],
            text=loaded["text"],
            doc_type=loaded["type"]
        )
        results.append({"doc_id": doc_id, "doc_name": file.filename})
    return {"message": "上传并入库成功", "documents": results}


@app.post("/documents/web")
def ingest_web(req: WebIngestRequest):
    doc_id = str(uuid4())
    loaded = DocumentLoader.load_web(req.url)

    ingest_text(
        doc_id=doc_id,
        doc_name=loaded["doc_name"],
        source=loaded["source"],
        text=loaded["text"],
        doc_type=loaded["type"]
    )
    return {"message": "网页入库成功", "doc_id": doc_id, "doc_name": req.url}


@app.get("/documents")
def list_documents():
    return {"documents": metadata_store.list_documents()}


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    vector_store.delete_document(doc_id)
    ok = metadata_store.delete_document(doc_id)
    if not ok:
        raise HTTPException(status_code=404, detail="文档不存在")
    return {"message": "删除成功", "doc_id": doc_id}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    answer, hits = rag_service.answer(req.question, [m.model_dump() for m in req.history or []])

    sources = []
    for h in hits:
        m = h["metadata"]
        sources.append(
            SourceItem(
                doc_id=m["doc_id"],
                doc_name=m["doc_name"],
                source=m["source"],
                chunk_index=int(m["chunk_index"]),
                score=float(h["score"]),
                content=h["content"]
            )
        )
    return ChatResponse(answer=answer, sources=sources)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    stream_gen, _ = rag_service.answer_stream(req.question, [m.model_dump() for m in req.history or []])

    def token_generator():
        for token in stream_gen:
            yield token

    return StreamingResponse(token_generator(), media_type="text/plain; charset=utf-8")