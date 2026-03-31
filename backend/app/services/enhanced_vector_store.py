import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import re
from collections import defaultdict


class EnhancedVectorStore:
    """增强的向量存储，支持Hybrid Search（关键词检索 + 向量检索）"""

    def __init__(self, persist_dir: str, embedding_model: str):
        """
        初始化增强向量存储

        Args:
            persist_dir: 持久化目录
            embedding_model: 嵌入模型名称
        """
        # 初始化ChromaDB
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name="kb_chunks")
        self.embedder = SentenceTransformer(embedding_model)

        # 关键词索引
        self.keyword_index: Dict[str, Set[str]] = defaultdict(set)
        self.chunk_lengths: Dict[str, int] = {}

        # 加载现有数据的关键词索引
        self._build_keyword_index_from_existing()

        print(f"✅ 增强向量存储初始化成功，目录: {persist_dir}")

    def _build_keyword_index_from_existing(self):
        """从现有数据构建关键词索引"""
        try:
            # 获取所有数据
            results = self.collection.get(include=["documents", "metadatas"])
            documents = results.get("documents", [])
            metadatas = results.get("metadatas", [])

            for doc_id, doc, metadata in zip(
                results.get("ids", []),
                documents,
                metadatas
            ):
                if doc:
                    self._add_to_keyword_index(doc_id, doc)

            print(f"✅ 从现有数据构建了 {len(self.keyword_index)} 个关键词的索引")

        except Exception as e:
            print(f"⚠️  构建关键词索引时出错: {e}")

    def _add_to_keyword_index(self, chunk_id: str, text: str):
        """添加文本到关键词索引"""
        # 分词（简单的基于空格和标点符号的分词）
        words = self._tokenize(text.lower())
        self.chunk_lengths[chunk_id] = len(text)

        for word in words:
            if len(word) > 2:  # 忽略太短的词
                self.keyword_index[word].add(chunk_id)

    def _remove_from_keyword_index(self, chunk_id: str, text: str):
        """从关键词索引中移除文本"""
        words = self._tokenize(text.lower())

        for word in words:
            if len(word) > 2 and word in self.keyword_index:
                self.keyword_index[word].discard(chunk_id)

                # 如果这个词没有关联任何chunk，删除这个词
                if not self.keyword_index[word]:
                    del self.keyword_index[word]

    def _tokenize(self, text: str) -> List[str]:
        """简单的分词函数"""
        # 使用正则表达式提取单词
        words = re.findall(r'\b\w+\b', text)
        return words

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """生成文本嵌入"""
        vectors = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return vectors.tolist()

    def add_chunks(self, chunks: List[str], metadatas: List[Dict], ids: List[str]) -> None:
        """
        添加文档块到存储

        Args:
            chunks: 文档块列表
            metadatas: 元数据列表
            ids: ID列表
        """
        # 添加到ChromaDB
        embeddings = self._embed(chunks)
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

        # 添加到关键词索引
        for chunk_id, chunk in zip(ids, chunks):
            self._add_to_keyword_index(chunk_id, chunk)

    def similarity_search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        向量相似度搜索

        Args:
            query: 查询语句
            top_k: 返回前k个结果

        Returns:
            搜索结果列表
        """
        q_emb = self._embed([query])[0]
        result = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        hits = []
        for d, m, dist in zip(docs, metas, dists):
            score = float(1 - dist) if dist is not None else 0.0
            hits.append({
                "content": d,
                "metadata": m,
                "score": score
            })

        return hits

    def keyword_search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        关键词搜索

        Args:
            query: 查询语句
            top_k: 返回前k个结果

        Returns:
            搜索结果列表
        """
        query_words = self._tokenize(query.lower())
        chunk_scores = defaultdict(float)

        # 计算每个chunk的关键词匹配分数
        for word in query_words:
            if word in self.keyword_index:
                # 使用BM25-like的评分
                doc_freq = len(self.keyword_index[word])
                idf = np.log((len(self.chunk_lengths) - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

                for chunk_id in self.keyword_index[word]:
                    # TF部分（简单的计数）
                    chunk_content = self.collection.get(ids=[chunk_id], include=["documents"])["documents"][0]
                    tf = chunk_content.lower().count(word)

                    # BM25-like评分
                    k1 = 1.5
                    b = 0.75
                    avg_doc_len = np.mean(list(self.chunk_lengths.values())) if self.chunk_lengths else 1
                    doc_len = self.chunk_lengths.get(chunk_id, 1)
                    tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))

                    chunk_scores[chunk_id] += idf * tf_norm

        # 排序并获取top_k
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for chunk_id, score in sorted_chunks:
            chunk_data = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
            results.append({
                "content": chunk_data["documents"][0],
                "metadata": chunk_data["metadatas"][0],
                "score": float(score)
            })

        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 4,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4,
        use_rerank: bool = False,
        rerank_top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        混合搜索：关键词检索 + 向量检索

        Args:
            query: 查询语句
            top_k: 最终返回前k个结果
            vector_weight: 向量检索权重
            keyword_weight: 关键词检索权重
            use_rerank: 是否使用重排序
            rerank_top_k: 重排序前k个候选

        Returns:
            搜索结果列表
        """
        # 向量检索
        vector_results = self.similarity_search(query, top_k=rerank_top_k if use_rerank else top_k)

        # 关键词检索
        keyword_results = self.keyword_search(query, top_k=rerank_top_k if use_rerank else top_k)

        # 归一化分数
        vector_results = self._normalize_scores(vector_results, method="minmax")
        keyword_results = self._normalize_scores(keyword_results, method="minmax")

        # 合并结果
        combined_scores = defaultdict(float)

        # 向量结果加权
        for hit in vector_results:
            chunk_id = hit["metadata"].get("chunk_id", "")
            combined_scores[chunk_id] += hit["score"] * vector_weight

        # 关键词结果加权
        for hit in keyword_results:
            chunk_id = hit["metadata"].get("chunk_id", "")
            combined_scores[chunk_id] += hit["score"] * keyword_weight

        # 排序
        sorted_chunks = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        # 如果不需要重排序，直接返回top_k
        if not use_rerank:
            results = []
            for chunk_id, score in sorted_chunks[:top_k]:
                # 获取完整信息
                chunk_data = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
                results.append({
                    "content": chunk_data["documents"][0],
                    "metadata": chunk_data["metadatas"][0],
                    "score": float(score),
                    "vector_score": next((h["score"] for h in vector_results if h["metadata"].get("chunk_id") == chunk_id), 0.0),
                    "keyword_score": next((h["score"] for h in keyword_results if h["metadata"].get("chunk_id") == chunk_id), 0.0)
                })
            return results

        # 重排序阶段
        # 获取候选文档
        candidate_docs = []
        candidate_hits = []
        for chunk_id, score in sorted_chunks[:rerank_top_k]:
            chunk_data = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
            candidate_docs.append(chunk_data["documents"][0])
            candidate_hits.append({
                "content": chunk_data["documents"][0],
                "metadata": chunk_data["metadatas"][0],
                "combined_score": float(score)
            })

        # 这里可以调用RerankService进行重排序
        # 由于RerankService可能在其他文件，我们暂时用启发式方法
        if candidate_docs:
            # 使用简单的启发式重排序
            reranked_results = self._simple_rerank(query, candidate_hits, top_k)
            return reranked_results

        return []

    def _normalize_scores(self, hits: List[Dict[str, Any]], method: str = "minmax") -> List[Dict[str, Any]]:
        """归一化分数"""
        if not hits:
            return hits

        scores = [hit["score"] for hit in hits]

        if method == "minmax":
            min_score = min(scores)
            max_score = max(scores)
            if max_score > min_score:
                for hit in hits:
                    hit["score"] = (hit["score"] - min_score) / (max_score - min_score)
            else:
                # 所有分数相同
                for hit in hits:
                    hit["score"] = 1.0
        elif method == "sigmoid":
            for hit in hits:
                hit["score"] = 1 / (1 + np.exp(-hit["score"]))

        return hits

    def _simple_rerank(self, query: str, hits: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
        """简单的启发式重排序"""
        query_lower = query.lower()

        for hit in hits:
            content = hit.get("content", "").lower()
            combined_score = hit.get("combined_score", 0.0)

            # 关键词匹配增强
            keyword_bonus = 0.0
            query_words = query_lower.split()
            for word in query_words:
                if word in content:
                    keyword_bonus += 0.1

            # 长度惩罚（避免过长的片段）
            length_penalty = 0.0
            content_length = len(content)
            if content_length > 1000:
                length_penalty = -0.1
            elif content_length < 50:
                length_penalty = -0.05

            # 综合评分
            hit["final_score"] = combined_score + keyword_bonus + length_penalty

        # 排序并返回top_k
        sorted_hits = sorted(hits, key=lambda x: x.get("final_score", 0), reverse=True)
        return sorted_hits[:top_k]

    def delete_document(self, doc_id: str) -> None:
        """删除文档"""
        # 从ChromaDB删除
        self.collection.delete(where={"doc_id": doc_id})

        # 从关键词索引删除
        # 需要找到所有包含该doc_id的chunk
        results = self.collection.get(where={"doc_id": doc_id}, include=["documents"])
        for chunk_id, doc in zip(results.get("ids", []), results.get("documents", [])):
            self._remove_from_keyword_index(chunk_id, doc)

    def update_chunk(self, chunk_id: str, new_content: str, new_metadata: Dict) -> None:
        """更新单个chunk"""
        # 获取旧内容
        old_data = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        old_content = old_data["documents"][0] if old_data["documents"] else ""

        # 更新ChromaDB
        new_embedding = self._embed([new_content])[0]
        self.collection.update(
            ids=[chunk_id],
            documents=[new_content],
            embeddings=[new_embedding],
            metadatas=[new_metadata]
        )

        # 更新关键词索引
        self._remove_from_keyword_index(chunk_id, old_content)
        self._add_to_keyword_index(chunk_id, new_content)

    def get_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        total_chunks = self.collection.count()
        total_keywords = len(self.keyword_index)
        avg_chunk_length = np.mean(list(self.chunk_lengths.values())) if self.chunk_lengths else 0

        return {
            "total_chunks": total_chunks,
            "total_keywords": total_keywords,
            "avg_chunk_length": float(avg_chunk_length),
            "keyword_index_size": total_keywords
        }


# 测试代码
if __name__ == "__main__":
    # 创建测试实例
    store = EnhancedVectorStore("test_data", "sentence-transformers/all-MiniLM-L6-v2")

    # 添加测试数据
    test_chunks = [
        "Python是一种流行的编程语言，用于Web开发、数据科学和人工智能。",
        "Java是另一种广泛使用的编程语言，特别适用于企业级应用。",
        "机器学习是人工智能的一个分支，让计算机从数据中学习。",
        "深度学习使用神经网络来解决复杂的问题。",
        "Web开发涉及创建和维护网站和Web应用程序。"
    ]

    test_metadatas = [
        {"doc_id": "doc1", "doc_name": "Python介绍", "chunk_index": 0},
        {"doc_id": "doc1", "doc_name": "Python介绍", "chunk_index": 1},
        {"doc_id": "doc2", "doc_name": "机器学习基础", "chunk_index": 0},
        {"doc_id": "doc2", "doc_name": "机器学习基础", "chunk_index": 1},
        {"doc_id": "doc3", "doc_name": "Web开发指南", "chunk_index": 0}
    ]

    test_ids = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]

    store.add_chunks(test_chunks, test_metadatas, test_ids)

    # 测试统计信息
    stats = store.get_stats()
    print(f"\n=== 存储统计 ===")
    print(f"总chunks数: {stats['total_chunks']}")
    print(f"总关键词数: {stats['total_keywords']}")
    print(f"平均chunk长度: {stats['avg_chunk_length']:.1f}")

    # 测试混合搜索
    query = "Python编程"
    print(f"\n=== 混合搜索测试 (查询: '{query}') ===")

    # 向量搜索
    vector_results = store.similarity_search(query, top_k=3)
    print(f"\n向量搜索结果:")
    for i, hit in enumerate(vector_results, 1):
        print(f"{i}. {hit['content'][:50]}... (分数: {hit['score']:.3f})")

    # 关键词搜索
    keyword_results = store.keyword_search(query, top_k=3)
    print(f"\n关键词搜索结果:")
    for i, hit in enumerate(keyword_results, 1):
        print(f"{i}. {hit['content'][:50]}... (分数: {hit['score']:.3f})")

    # 混合搜索
    hybrid_results = store.hybrid_search(query, top_k=3)
    print(f"\n混合搜索结果:")
    for i, hit in enumerate(hybrid_results, 1):
        print(f"{i}. {hit['content'][:50]}... (最终分数: {hit.get('final_score', hit['score']):.3f})")

    # 测试重排序
    print(f"\n=== 重排序测试 ===")
    reranked_results = store._simple_rerank(query, hybrid_results, top_k=2)
    for i, hit in enumerate(reranked_results, 1):
        print(f"{i}. {hit['content'][:50]}... (重排分数: {hit.get('final_score', hit['score']):.3f})")

    # 清理测试数据
    import shutil
    if Path("test_data").exists():
        shutil.rmtree("test_data")