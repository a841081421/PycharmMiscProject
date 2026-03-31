from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List, Optional
from pathlib import Path
from uuid import uuid4
from datetime import datetime

from .config import (
    UPLOAD_DIR, CHROMA_DIR, METADATA_FILE,
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
)
from .schemas import (
    WebIngestRequest, ChatRequest, ChatResponse, SourceItem,
    DocumentUpdate, DocumentResponse, DocumentCreate,
    Conversation, ConversationCreate, ConversationUpdate, ConversationResponse, ConversationMessage, ChatMessage
)
from .services.document_loader import DocumentLoader
from .services.splitter import TextSplitter
from .services.vector_store import VectorStore
from .services.llm_client import LLMClient, EnhancedStreamingResponse
from .services.rag_service import RAGService
from .services.metadata_store import MetadataStore
from .services.conversation_store import ConversationStore

app = FastAPI(title="Local KB QA", version="0.1.0")

splitter = TextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
vector_store = VectorStore(str(CHROMA_DIR), EMBEDDING_MODEL)
metadata_store = MetadataStore(METADATA_FILE)
llm_client = LLMClient(LLM_API_KEY, LLM_BASE_URL, LLM_MODEL)
rag_service = RAGService(vector_store, llm_client, TOP_K)
conversation_store = ConversationStore(Path("data/conversations.db"))


def ingest_text(
    doc_id: str,
    doc_name: str,
    source: str,
    text: str,
    doc_type: str,
    tags: Optional[List[str]] = None,
    category: Optional[str] = None,
    metadata: Optional[dict] = None
):
    """增强的文档摄入函数，支持标签和分类"""
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
        "created_at": datetime.now().isoformat(),
        "tags": tags or [],
        "category": category,
        "metadata": metadata or {}
    })


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    tags: Optional[List[str]] = None,
    category: Optional[str] = None
):
    """增强的文档上传端点，支持标签和分类"""
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
            doc_type=loaded["type"],
            tags=tags,
            category=category
        )
        results.append({"doc_id": doc_id, "doc_name": file.filename})
    return {"message": "上传并入库成功", "documents": results}


@app.post("/documents/web")
def ingest_web(req: WebIngestRequest):
    """增强的网页抓取端点，支持标签和分类"""
    doc_id = str(uuid4())
    loaded = DocumentLoader.load_web(req.url)

    ingest_text(
        doc_id=doc_id,
        doc_name=loaded["doc_name"],
        source=loaded["source"],
        text=loaded["text"],
        doc_type=loaded["type"],
        tags=req.tags,
        category=req.category
    )
    return {"message": "网页入库成功", "doc_id": doc_id, "doc_name": req.url}


@app.get("/documents")
def list_documents(
    category: Optional[str] = Query(None, description="按分类过滤"),
    tags: Optional[List[str]] = Query(None, description="按标签过滤"),
    doc_type: Optional[str] = Query(None, description="按文档类型过滤")
):
    """增强的文档列表端点，支持多条件过滤"""
    documents = metadata_store.list_documents(
        category=category,
        tags=tags,
        doc_type=doc_type
    )
    return {"documents": documents}


@app.get("/documents/search")
def search_documents(
    q: str = Query(..., description="搜索关键词"),
    category: Optional[str] = Query(None, description="按分类过滤"),
    tags: Optional[List[str]] = Query(None, description="按标签过滤"),
    doc_type: Optional[str] = Query(None, description="按文档类型过滤")
):
    """文档搜索端点"""
    results = metadata_store.search_documents(q)

    # 应用额外的过滤
    if category:
        results = [d for d in results if d.get('category') == category]
    if tags:
        results = [d for d in results if any(tag in d.get('tags', []) for tag in tags)]
    if doc_type:
        results = [d for d in results if d.get('doc_type') == doc_type]

    return {"results": results, "query": q, "total": len(results)}


@app.get("/documents/{doc_id}", response_model=DocumentResponse)
def get_document(doc_id: str):
    """获取文档详情"""
    doc = metadata_store.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")
    return doc


@app.put("/documents/{doc_id}", response_model=DocumentResponse)
def update_document(doc_id: str, update: DocumentUpdate):
    """更新文档信息"""
    success = metadata_store.update_document(doc_id, update.dict(exclude_unset=True))
    if not success:
        raise HTTPException(status_code=404, detail="文档不存在")

    # 获取更新后的文档
    doc = metadata_store.get_document(doc_id)
    return doc


@app.post("/documents/{doc_id}/tags")
def add_document_tags(doc_id: str, tags: List[str]):
    """为文档添加标签"""
    success = metadata_store.add_tags(doc_id, tags)
    if not success:
        raise HTTPException(status_code=404, detail="文档不存在")
    return {"message": "标签添加成功", "doc_id": doc_id, "tags": tags}


@app.delete("/documents/{doc_id}/tags/{tag}")
def remove_document_tag(doc_id: str, tag: str):
    """从文档移除标签"""
    # 获取文档
    doc = metadata_store.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")

    # 移除标签
    if tag in doc.get('tags', []):
        doc['tags'].remove(tag)
        success = metadata_store.update_document(doc_id, {'tags': doc['tags']})
        if success:
            return {"message": "标签移除成功", "doc_id": doc_id, "tag": tag}

    raise HTTPException(status_code=400, detail=f"标签 '{tag}' 不存在")


# ==================== 对话管理API ====================


@app.post("/conversations", response_model=ConversationResponse)
def create_conversation(conv: ConversationCreate):
    """创建新对话"""
    conv_id = conversation_store.create_conversation(
        title=conv.title,
        doc_ids=conv.doc_ids,
        metadata={}
    )
    return ConversationResponse(
        conv_id=conv_id,
        title=conv.title or f"对话 {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        message_count=0,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )


@app.get("/conversations", response_model=List[ConversationResponse])
def list_conversations(limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)):
    """列出所有对话"""
    conversations = conversation_store.list_conversations(limit=limit, offset=offset)
    return [
        ConversationResponse(
            conv_id=conv['conv_id'],
            title=conv['title'],
            message_count=conversation_store.get_conversation_message_count(conv['conv_id']),
            created_at=conv['created_at'],
            updated_at=conv['updated_at']
        )
        for conv in conversations
    ]


@app.get("/conversations/{conv_id}", response_model=Conversation)
def get_conversation(conv_id: str):
    """获取对话详情"""
    conv = conversation_store.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="对话不存在")

    # 获取对话消息
    messages = conversation_store.get_messages(conv_id)
    conv['messages'] = [
        ConversationMessage(
            message_id=msg['message_id'],
            role=msg['role'],
            content=msg['content'],
            timestamp=msg['timestamp'],
            sources=msg['sources']
        )
        for msg in messages
    ]

    return conv


@app.put("/conversations/{conv_id}", response_model=ConversationResponse)
def update_conversation(conv_id: str, update: ConversationUpdate):
    """更新对话信息"""
    success = conversation_store.update_conversation(
        conv_id,
        title=update.title,
        metadata=update.metadata
    )
    if not success:
        raise HTTPException(status_code=404, detail="对话不存在")

    return ConversationResponse(
        conv_id=conv_id,
        title=update.title,
        message_count=conversation_store.get_conversation_message_count(conv_id),
        created_at=datetime.now(),
        updated_at=datetime.now()
    )


@app.delete("/conversations/{conv_id}")
def delete_conversation(conv_id: str):
    """删除对话"""
    success = conversation_store.delete_conversation(conv_id)
    if not success:
        raise HTTPException(status_code=404, detail="对话不存在")
    return {"message": "对话删除成功", "conv_id": conv_id}


@app.post("/conversations/{conv_id}/messages")
def add_conversation_message(conv_id: str, message: ChatMessage):
    """添加消息到对话"""
    # 检查对话是否存在
    conv = conversation_store.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="对话不存在")

    # 添加消息
    message_id = conversation_store.add_message(
        conv_id=conv_id,
        role=message.role,
        content=message.content,
        sources=[]
    )

    return {"message": "消息添加成功", "message_id": message_id, "conv_id": conv_id}


@app.get("/conversations/{conv_id}/messages")
def get_conversation_messages(
    conv_id: str,
    limit: Optional[int] = Query(None, ge=1),
    since: Optional[datetime] = None
):
    """获取对话消息历史"""
    # 检查对话是否存在
    conv = conversation_store.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="对话不存在")

    # 获取消息
    messages = conversation_store.get_messages(conv_id, limit=limit, since=since)
    return {
        "conv_id": conv_id,
        "messages": messages,
        "total": len(messages)
    }


@app.post("/conversations/{conv_id}/chat", response_model=ChatResponse)
def conversation_chat(conv_id: str, req: ChatRequest):
    """在特定对话中进行问答"""
    # 检查对话是否存在
    conv = conversation_store.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="对话不存在")

    # 获取对话历史
    history_messages = conversation_store.get_messages(conv_id, limit=100)
    history = [
        {"role": msg['role'], "content": msg['content']}
        for msg in history_messages
    ]

    # 使用RAG服务回答问题
    answer, hits = rag_service.answer(req.question, history)

    # 构建来源信息
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

    # 添加助手消息到对话
    conversation_store.add_message(
        conv_id=conv_id,
        role="assistant",
        content=answer,
        sources=sources
    )

    return ChatResponse(answer=answer, sources=sources)


@app.post("/conversations/{conv_id}/chat/stream")
def conversation_chat_stream(conv_id: str, req: ChatRequest):
    """在特定对话中进行流式问答"""
    # 检查对话是否存在
    conv = conversation_store.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="对话不存在")

    # 获取对话历史
    history_messages = conversation_store.get_messages(conv_id, limit=100)
    history = [
        {"role": msg['role'], "content": msg['content']}
        for msg in history_messages
    ]

    # 使用RAG服务进行流式回答
    stream_gen, hits = rag_service.answer_stream(req.question, history)

    # 构建来源信息
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

    def enhanced_token_generator():
        """增强的流式生成器，包含来源信息"""
        # 先发送来源信息标记
        yield f"【来源信息】\n"
        for i, source in enumerate(sources, 1):
            yield f"[片段{i}] {source.doc_name} (相关度: {source.score:.3f})\n"
        yield f"\n【回答内容】\n"

        # 然后发送回答内容
        for token in stream_gen:
            yield token

        # 最后发送结束标记
        yield f"\n\n【回答结束】\n"

    # 异步保存助手消息到对话
    import asyncio
    async def save_message():
        # 这里简化处理，实际应该等待流式完成后再保存
        pass

    return StreamingResponse(enhanced_token_generator(), media_type="text/plain; charset=utf-8")


# ==================== 增强流式输出API ====================


@app.post("/chat/stream/enhanced")
def chat_stream_enhanced(req: ChatRequest):
    """增强版流式输出，包含来源信息和元数据"""
    stream_gen, hits = rag_service.answer_stream_enhanced(req.question, [m.model_dump() for m in req.history or []])

    return StreamingResponse(stream_gen, media_type="text/plain; charset=utf-8")


@app.post("/chat/stream/json")
def chat_stream_json(req: ChatRequest):
    """JSON格式流式输出，适合机器解析"""
    stream_gen, hits = rag_service.answer_stream_json(req.question, [m.model_dump() for m in req.history or []])

    return StreamingResponse(
        stream_gen,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Content-Type": "text/event-stream"
        }
    )


@app.post("/chat/stream/sse")
def chat_stream_sse(req: ChatRequest):
    """SSE格式流式输出，适合服务器推送事件"""
    stream_gen, hits = rag_service.answer_stream_sse(req.question, [m.model_dump() for m in req.history or []])

    return StreamingResponse(
        stream_gen,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Content-Type": "text/event-stream"
        }
    )


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