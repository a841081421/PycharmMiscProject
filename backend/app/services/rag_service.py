from typing import List, Dict, Tuple, Generator, Any, Optional
from .vector_store import VectorStore
from .llm_client import LLMClient


class RAGService:
    def __init__(self, vector_store: VectorStore, llm_client: LLMClient, top_k: int):
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.top_k = top_k

    def retrieve(self, question: str) -> List[Dict]:
        """检索相关知识"""
        return self.vector_store.similarity_search(question, self.top_k)

    def _build_messages(self, question: str, history: List[Dict], hits: List[Dict]) -> List[Dict]:
        """构建发送给LLM的消息"""
        context_lines = []
        for i, hit in enumerate(hits, start=1):
            meta = hit["metadata"]
            context_lines.append(
                f"[片段{i}] 来源={meta.get('doc_name')} | chunk={meta.get('chunk_index')}\n"
                f"{hit['content']}"
            )

        context_text = "\n\n".join(context_lines) if context_lines else "无可用上下文"

        system_prompt = (
            "你是一个本地知识库问答助手。"
            "请严格根据给定上下文回答问题，不要编造。"
            "若上下文不足，明确说“根据当前知识库内容无法确定”。"
            "回答尽量简洁，并在结尾附上你使用了哪些片段编号，如：[片段1][片段3]。"
        )

        messages = [{"role": "system", "content": system_prompt}]
        for msg in history or []:
            if msg.get("role") in {"user", "assistant"}:
                messages.append(msg)

        user_prompt = f"问题：{question}\n\n可用上下文：\n{context_text}"
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _format_sources_for_stream(self, hits: List[Dict]) -> List[Dict]:
        """格式化来源信息，用于流式输出"""
        sources = []
        for hit in hits:
            meta = hit["metadata"]
            sources.append({
                "doc_id": meta.get("doc_id", ""),
                "doc_name": meta.get("doc_name", ""),
                "source": meta.get("source", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "content": hit.get("content", ""),
                "score": hit.get("score", 0.0),
                "metadata": meta
            })
        return sources

    def answer(self, question: str, history: List[Dict]) -> Tuple[str, List[Dict]]:
        """普通问答"""
        hits = self.retrieve(question)
        messages = self._build_messages(question, history, hits)
        answer = self.llm_client.chat(messages)
        return answer, hits

    def answer_stream(self, question: str, history: List[Dict]) -> Tuple[Generator[str, None, None], List[Dict]]:
        """基础流式问答"""
        hits = self.retrieve(question)
        messages = self._build_messages(question, history, hits)
        return self.llm_client.chat_stream(messages), hits

    def answer_stream_enhanced(self, question: str, history: List[Dict],
                               conv_id: Optional[str] = None) -> Tuple[Generator[str, None, None], List[Dict]]:
        """增强的流式问答，包含来源信息和元数据"""
        import time
        start_time = time.time()

        hits = self.retrieve(question)
        messages = self._build_messages(question, history, hits)

        # 构建元数据
        metadata = {
            "retrieval_time": time.time() - start_time,
            "total_chunks": len(hits),
            "top_k": self.top_k,
            "conversation_id": conv_id,
            "model": self.llm_client.model
        }

        # 格式化来源信息
        sources = self._format_sources_for_stream(hits)

        return self.llm_client.chat_stream_enhanced(messages, sources, metadata), hits

    def answer_stream_json(self, question: str, history: List[Dict],
                           conv_id: Optional[str] = None) -> Tuple[Generator[str, None, None], List[Dict]]:
        """JSON格式的流式问答，适合机器解析"""
        import time
        start_time = time.time()

        hits = self.retrieve(question)
        messages = self._build_messages(question, history, hits)

        # 构建元数据
        metadata = {
            "retrieval_time": time.time() - start_time,
            "total_chunks": len(hits),
            "top_k": self.top_k,
            "conversation_id": conv_id,
            "model": self.llm_client.model
        }

        # 格式化来源信息
        sources = self._format_sources_for_stream(hits)

        return self.llm_client.chat_stream_json(messages, sources, metadata), hits

    def answer_stream_sse(self, question: str, history: List[Dict],
                          conv_id: Optional[str] = None) -> Tuple[Generator[str, None, None], List[Dict]]:
        """SSE格式的流式问答，适合服务器推送事件"""
        import time
        start_time = time.time()

        hits = self.retrieve(question)
        messages = self._build_messages(question, history, hits)

        # 构建元数据
        metadata = {
            "retrieval_time": time.time() - start_time,
            "total_chunks": len(hits),
            "top_k": self.top_k,
            "conversation_id": conv_id,
            "model": self.llm_client.model
        }

        # 格式化来源信息
        sources = self._format_sources_for_stream(hits)

        return self.llm_client.chat_stream_sse(messages, sources, metadata), hits