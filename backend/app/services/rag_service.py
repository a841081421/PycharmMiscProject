from typing import List, Dict, Tuple, Generator, Any, Optional
from .vector_store import VectorStore
from .llm_client import LLMClient
from .enhanced_vector_store import EnhancedVectorStore
from .rerank_service import RerankService, SimpleRerankService
from .enhanced_prompt import EnhancedPromptBuilder


class RAGService:
    def __init__(
        self,
        vector_store: VectorStore,
        llm_client: LLMClient,
        top_k: int = 4,
        use_hybrid_search: bool = True,
        use_rerank: bool = True,
        use_enhanced_prompt: bool = True,
        prompt_template: str = "default",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化增强的RAG服务

        Args:
            vector_store: 向量存储实例
            llm_client: LLM客户端实例
            top_k: 返回前k个结果
            use_hybrid_search: 是否使用混合搜索
            use_rerank: 是否使用重排序
            use_enhanced_prompt: 是否使用增强的Prompt
            prompt_template: Prompt模板类型
            config: 额外配置
        """
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.top_k = top_k
        self.use_hybrid_search = use_hybrid_search
        self.use_rerank = use_rerank
        self.use_enhanced_prompt = use_enhanced_prompt
        self.config = config or {}

        # 初始化增强组件
        if use_hybrid_search and isinstance(vector_store, EnhancedVectorStore):
            self.search_method = vector_store.hybrid_search
        else:
            self.search_method = vector_store.similarity_search

        if use_rerank:
            # 尝试使用完整Rerank，如果失败则使用简单Rerank
            try:
                self.rerank_service = RerankService(
                    model_name=self.config.get("rerank_model", "BAAI/bge-reranker-base"),
                    cache_dir=self.config.get("model_cache_dir")
                )
            except Exception as e:
                print(f"⚠️  完整Rerank初始化失败，使用简单Rerank: {e}")
                self.rerank_service = SimpleRerankService()
        else:
            self.rerank_service = None

        if use_enhanced_prompt:
            self.prompt_builder = EnhancedPromptBuilder(prompt_template)
        else:
            self.prompt_builder = None

    def retrieve(self, question: str) -> List[Dict]:
        """
        检索相关知识，支持混合搜索和重排序

        Args:
            question: 查询语句

        Returns:
            检索结果列表
        """
        # 混合搜索或向量搜索
        search_results = self.search_method(
            question,
            top_k=self.top_k * 2 if self.use_rerank else self.top_k
        )

        # 重排序
        if self.use_rerank and self.rerank_service and search_results:
            reranked_results = self.rerank_service.rerank_with_metadata(
                question,
                search_results,
                top_k=self.top_k
            )
            return reranked_results

        return search_results[:self.top_k]

    def _build_messages(self, question: str, history: List[Dict], hits: List[Dict]) -> List[Dict]:
        """
        构建发送给LLM的消息

        Args:
            question: 用户问题
            history: 对话历史
            hits: 检索到的上下文片段

        Returns:
            消息列表
        """
        if self.use_enhanced_prompt and self.prompt_builder:
            # 使用增强的Prompt构建器
            return self.prompt_builder.build_messages(
                question=question,
                history=history,
                hits=hits,
                context_template=self.config.get("context_template", "default"),
                user_template=self.config.get("user_template", "default"),
                max_context_length=self.config.get("max_context_length", 4000),
                include_context_guide=self.config.get("include_context_guide", True)
            )
        else:
            # 使用原有的简单Prompt构建
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