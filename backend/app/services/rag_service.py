from typing import List, Dict, Tuple, Generator
from .vector_store import VectorStore
from .llm_client import LLMClient


class RAGService:
    def __init__(self, vector_store: VectorStore, llm_client: LLMClient, top_k: int):
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.top_k = top_k

    def retrieve(self, question: str) -> List[Dict]:
        return self.vector_store.similarity_search(question, self.top_k)

    def _build_messages(self, question: str, history: List[Dict], hits: List[Dict]) -> List[Dict]:
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

    def answer(self, question: str, history: List[Dict]) -> Tuple[str, List[Dict]]:
        hits = self.retrieve(question)
        messages = self._build_messages(question, history, hits)
        answer = self.llm_client.chat(messages)
        return answer, hits

    def answer_stream(self, question: str, history: List[Dict]) -> Tuple[Generator[str, None, None], List[Dict]]:
        hits = self.retrieve(question)
        messages = self._build_messages(question, history, hits)
        return self.llm_client.chat_stream(messages), hits