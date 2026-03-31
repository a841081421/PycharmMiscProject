import json
from openai import OpenAI
from typing import List, Dict, Generator, Any, Optional
from datetime import datetime


class EnhancedStreamingResponse:
    """增强的流式响应格式"""

    def __init__(self, content: str, sources: Optional[List[Dict]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.content = content
        self.sources = sources or []
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps({
            "content": self.content,
            "sources": self.sources,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }, ensure_ascii=False)

    def to_sse(self) -> str:
        """转换为SSE格式"""
        data = {
            "choices": [{
                "delta": {"content": self.content},
                "finish_reason": None
            }],
            "sources": self.sources,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


class LLMClient:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, messages: List[Dict]) -> str:
        """普通聊天"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2
        )
        return resp.choices[0].message.content or ""

    def chat_stream(self, messages: List[Dict]) -> Generator[str, None, None]:
        """基础流式聊天"""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta

    def chat_stream_enhanced(self, messages: List[Dict],
                             sources: Optional[List[Dict]] = None,
                             metadata: Optional[Dict[str, Any]] = None) -> Generator[str, None, None]:
        """增强的流式聊天，包含来源信息和元数据"""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            stream=True
        )

        # 首先发送来源信息和元数据
        if sources or metadata:
            yield f"【知识库检索结果】\n"
            if sources:
                for i, source in enumerate(sources, 1):
                    yield f"[片段{i}] {source.get('doc_name', '未知')} "
                    yield f"(相关度: {source.get('score', 0):.3f})\n"
                    yield f"   {source.get('content', '')[:100]}...\n"
            if metadata:
                yield f"【元数据】{json.dumps(metadata, ensure_ascii=False)}\n"
            yield f"\n【开始回答】\n\n"

        # 然后发送流式内容
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta

        # 最后发送结束标记
        yield f"\n\n【回答结束】\n"

    def chat_stream_json(self, messages: List[Dict],
                         sources: Optional[List[Dict]] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> Generator[str, None, None]:
        """JSON格式的流式聊天，适合机器解析"""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            stream=True
        )

        # 发送开始标记
        yield self._format_json_chunk("", sources, metadata, is_start=True)

        # 发送流式内容
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield self._format_json_chunk(delta, is_continue=True)

        # 发送结束标记
        yield self._format_json_chunk("", is_finish=True)

    def _format_json_chunk(self, content: str,
                           sources: Optional[List[Dict]] = None,
                           metadata: Optional[Dict[str, Any]] = None,
                           is_start: bool = False,
                           is_continue: bool = False,
                           is_finish: bool = False) -> str:
        """格式化JSON格式的流式数据"""
        data = {
            "id": f"chatcmpl-{datetime.now().timestamp()}",
            "object": "chat.completion.chunk",
            "created": datetime.now().timestamp(),
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": "stop" if is_finish else None
            }]
        }

        if is_start:
            data["sources"] = sources or []
            data["metadata"] = metadata or {}

        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    def chat_stream_sse(self, messages: List[Dict],
                        sources: Optional[List[Dict]] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> Generator[str, None, None]:
        """SSE格式的流式聊天，适合服务器推送事件"""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            stream=True
        )

        # 发送开始事件
        yield self._format_sse_event("start", "", sources, metadata)

        # 发送内容事件
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield self._format_sse_event("content", delta)

        # 发送结束事件
        yield self._format_sse_event("finish", "")

    def _format_sse_event(self, event_type: str,
                          content: str = "",
                          sources: Optional[List[Dict]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """格式化SSE事件"""
        data = {
            "event": event_type,
            "data": {
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "model": self.model
            }
        }

        if event_type == "start":
            data["data"]["sources"] = sources or []
            data["data"]["metadata"] = metadata or {}

        return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"