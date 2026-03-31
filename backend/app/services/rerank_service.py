import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path


class RerankService:
    """Rerank服务，使用bge-reranker模型提升召回精度"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", cache_dir: Optional[str] = None):
        """
        初始化Rerank服务

        Args:
            model_name: 模型名称，支持 BAAI/bge-reranker-base, BAAI/bge-reranker-large
            cache_dir: 模型缓存目录
        """
        self.model_name = model_name

        # 设置缓存目录
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

        # 加载模型和分词器
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                use_fast=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                num_labels=1  # reranker模型通常只有一个输出
            )

            # 如果有GPU，使用GPU
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

            self.model.eval()  # 设置为评估模式
            print(f"✅ Rerank服务初始化成功，模型: {model_name}, 设备: {self.device}")

        except Exception as e:
            print(f"❌ Rerank服务初始化失败: {e}")
            raise

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 4,
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        重排序文档

        Args:
            query: 查询语句
            documents: 待排序的文档列表
            top_k: 返回前k个结果
            batch_size: 批量推理的大小

        Returns:
            排序后的文档列表，包含分数
        """
        if not documents:
            return []

        # 确保top_k不超过文档数量
        top_k = min(top_k, len(documents))

        # 构建查询-文档对
        pairs = [[query, doc] for doc in documents]

        # 批量推理
        all_scores = []

        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]

                # 分词
                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )

                # 移动到相应设备
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # 推理
                outputs = self.model(**inputs)
                scores = outputs.logits.squeeze(-1)  # [batch_size]

                # 转换为CPU并添加到结果列表
                all_scores.extend(scores.cpu().numpy().tolist())

        # 按分数排序（降序）
        sorted_indices = np.argsort(all_scores)[::-1]

        # 构建结果列表
        results = []
        for idx in sorted_indices[:top_k]:
            results.append({
                "content": documents[idx],
                "score": float(all_scores[idx]),
                "original_index": int(idx)
            })

        return results

    def rerank_with_metadata(
        self,
        query: str,
        hits: List[Dict[str, Any]],
        top_k: int = 4,
        score_key: str = "score"
    ) -> List[Dict[str, Any]]:
        """
        重排序带有元数据的hits

        Args:
            query: 查询语句
            hits: 待排序的hits列表，每个hit包含content和metadata
            top_k: 返回前k个结果
            score_key: 原始分数的键名

        Returns:
            排序后的hits列表
        """
        if not hits:
            return []

        # 提取文档内容
        documents = [hit.get("content", "") for hit in hits]

        # 调用rerank
        reranked_results = self.rerank(query, documents, top_k)

        # 将结果映射回原始hits
        results = []
        for reranked_item in reranked_results:
            original_idx = reranked_item["original_index"]
            original_hit = hits[original_idx].copy()

            # 更新分数
            original_hit[score_key] = reranked_item["score"]
            original_hit["rerank_score"] = reranked_item["score"]

            results.append(original_hit)

        return results

    def compute_similarity_matrix(
        self,
        queries: List[str],
        documents: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        计算查询-文档相似度矩阵

        Args:
            queries: 查询语句列表
            documents: 文档列表
            batch_size: 批量推理大小

        Returns:
            相似度矩阵，形状为 [len(queries), len(documents)]
        """
        num_queries = len(queries)
        num_docs = len(documents)

        # 构建所有查询-文档对
        all_pairs = []
        pair_indices = []  # 记录每个pair对应的query_idx和doc_idx

        for q_idx, query in enumerate(queries):
            for d_idx, doc in enumerate(documents):
                all_pairs.append([query, doc])
                pair_indices.append((q_idx, d_idx))

        # 批量推理
        all_scores = []

        with torch.no_grad():
            for i in range(0, len(all_pairs), batch_size):
                batch_pairs = all_pairs[i:i + batch_size]

                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )

                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                scores = outputs.logits.squeeze(-1).cpu().numpy()

                all_scores.extend(scores.tolist())

        # 构建相似度矩阵
        similarity_matrix = np.zeros((num_queries, num_docs))

        for (q_idx, d_idx), score in zip(pair_indices, all_scores):
            similarity_matrix[q_idx, d_idx] = score

        return similarity_matrix


class SimpleRerankService:
    """简单的Rerank服务，使用启发式方法，不需要额外模型"""

    def __init__(self):
        """初始化简单Rerank服务"""
        print("✅ 简单Rerank服务初始化成功")

    def simple_rerank(
        self,
        query: str,
        hits: List[Dict[str, Any]],
        top_k: int = 4,
        score_key: str = "score"
    ) -> List[Dict[str, Any]]:
        """
        使用启发式方法重排序

        Args:
            query: 查询语句
            hits: 待排序的hits列表
            top_k: 返回前k个结果
            score_key: 原始分数的键名

        Returns:
            排序后的hits列表
        """
        if not hits:
            return []

        query_lower = query.lower()

        # 计算额外的启发式分数
        for hit in hits:
            content = hit.get("content", "").lower()
            metadata = hit.get("metadata", {})

            # 基础分数
            base_score = hit.get(score_key, 0.0)

            # 关键词匹配分数
            keyword_score = 0.0
            query_words = query_lower.split()
            for word in query_words:
                if word in content:
                    keyword_score += 0.1

            # 上下文相关性分数
            context_score = 0.0
            if "doc_name" in metadata:
                doc_name = metadata["doc_name"].lower()
                if any(word in doc_name for word in query_words):
                    context_score += 0.05

            # 综合分数
            hit["heuristic_score"] = base_score + keyword_score + context_score

        # 按启发式分数排序
        sorted_hits = sorted(hits, key=lambda x: x.get("heuristic_score", 0), reverse=True)

        # 返回top_k
        return sorted_hits[:top_k]


# 测试代码
if __name__ == "__main__":
    # 测试简单Rerank服务
    simple_rerank = SimpleRerankService()

    query = "如何安装Python"
    hits = [
        {
            "content": "Python的安装步骤很简单",
            "score": 0.8,
            "metadata": {"doc_name": "Python安装指南"}
        },
        {
            "content": "Java的安装需要配置环境变量",
            "score": 0.7,
            "metadata": {"doc_name": "Java安装教程"}
        },
        {
            "content": "Python的pip工具用于包管理",
            "score": 0.6,
            "metadata": {"doc_name": "Python工具介绍"}
        }
    ]

    results = simple_rerank.simple_rerank(query, hits, top_k=2)
    print("\n=== 简单Rerank测试结果 ===")
    for i, hit in enumerate(results, 1):
        print(f"{i}. 内容: {hit['content'][:30]}...")
        print(f"   分数: {hit['score']:.3f} -> {hit['heuristic_score']:.3f}")