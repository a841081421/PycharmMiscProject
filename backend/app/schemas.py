from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class WebIngestRequest(BaseModel):
    url: str
    tags: List[str] = Field(default_factory=list)
    category: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    history: Optional[List[ChatMessage]] = []


class SourceItem(BaseModel):
    doc_id: str
    doc_name: str
    source: str
    chunk_index: int
    score: float
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


# ==================== 数据模型增强 ====================


class DocumentBase(BaseModel):
    """文档基础模型"""
    doc_id: str
    doc_name: str
    source: str
    doc_type: str
    chunk_count: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    version: int = Field(default=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentCreate(BaseModel):
    """创建文档请求"""
    doc_name: str
    source: str
    doc_type: str
    text: str
    tags: List[str] = Field(default_factory=list)
    category: Optional[str] = None


class DocumentUpdate(BaseModel):
    """更新文档请求"""
    doc_name: Optional[str] = None
    tags: Optional[List[str]] = None
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentResponse(DocumentBase):
    """文档响应模型"""
    class Config:
        from_attributes = True


# ==================== 对话模型 ====================


class ConversationMessage(BaseModel):
    """对话消息"""
    message_id: str
    role: str  # user, assistant
    content: str
    timestamp: datetime
    sources: Optional[List[SourceItem]] = None


class Conversation(BaseModel):
    """对话会话"""
    conv_id: str
    title: str
    messages: List[ConversationMessage]
    created_at: datetime
    updated_at: datetime
    doc_ids: List[str] = Field(default_factory=list)  # 关联的文档ID
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationCreate(BaseModel):
    """创建对话请求"""
    title: Optional[str] = None
    doc_ids: List[str] = Field(default_factory=list)


class ConversationUpdate(BaseModel):
    """更新对话请求"""
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ConversationResponse(BaseModel):
    """对话响应"""
    conv_id: str
    title: str
    message_count: int
    created_at: datetime
    updated_at: datetime


# ==================== 增强召回结果模型 ====================


class EnhancedSourceItem(SourceItem):
    """增强的来源项"""
    context_before: Optional[str] = None  # 前文上下文
    context_after: Optional[str] = None   # 后文上下文
    relevance_score: float = Field(default=0.0)  # 相关度分数
    highlight_ranges: List[Dict[str, int]] = Field(default_factory=list)  # 高亮位置
    metadata: Dict[str, Any] = Field(default_factory=dict)  # 额外元数据


class RetrievalResult(BaseModel):
    """检索结果"""
    query: str
    sources: List[EnhancedSourceItem]
    retrieval_time: float  # 检索耗时（秒）
    total_chunks: int  # 总检索块数
    top_k: int  # 返回的top_k数量


class EnhancedChatResponse(BaseModel):
    """增强的聊天响应"""
    answer: str
    sources: List[EnhancedSourceItem]
    conversation_id: Optional[str] = None
    retrieval_time: Optional[float] = None
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None