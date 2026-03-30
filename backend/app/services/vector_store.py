from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np


class VectorStore:
    def __init__(self, persist_dir: str, embedding_model: str):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name="kb_chunks")
        self.embedder = SentenceTransformer(embedding_model)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        vectors = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return vectors.tolist()

    def add_chunks(self, chunks: List[str], metadatas: List[Dict], ids: List[str]) -> None:
        embeddings = self._embed(chunks)
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def similarity_search(self, query: str, top_k: int = 4):
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

    def delete_document(self, doc_id: str) -> None:
        self.collection.delete(where={"doc_id": doc_id})