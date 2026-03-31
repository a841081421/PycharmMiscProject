import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid


class MetadataStore:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """确保元数据文件存在"""
        if not self.file_path.exists():
            self._write({"documents": {}})

    def _read(self) -> Dict:
        """读取元数据"""
        try:
            with self.file_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {"documents": {}}

    def _write(self, data: Dict) -> None:
        """写入元数据"""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with self.file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_document(self, doc_data: Dict[str, Any]) -> str:
        """添加文档"""
        data = self._read()
        doc_id = doc_data.get('doc_id', str(uuid.uuid4()))

        document = {
            'doc_id': doc_id,
            'doc_name': doc_data['doc_name'],
            'source': doc_data['source'],
            'doc_type': doc_data['doc_type'],
            'chunk_count': doc_data['chunk_count'],
            'created_at': doc_data.get('created_at', datetime.now().isoformat()),
            'updated_at': doc_data.get('updated_at'),
            'tags': doc_data.get('tags', []),
            'category': doc_data.get('category'),
            'version': doc_data.get('version', 1),
            'metadata': doc_data.get('metadata', {})
        }

        data["documents"][doc_id] = document
        self._write(data)
        return doc_id

    def update_document(self, doc_id: str, update_data: Dict[str, Any]) -> bool:
        """更新文档"""
        data = self._read()
        if doc_id not in data["documents"]:
            return False

        doc = data["documents"][doc_id]

        # 更新字段
        if 'doc_name' in update_data:
            doc['doc_name'] = update_data['doc_name']
        if 'tags' in update_data:
            doc['tags'] = update_data['tags']
        if 'category' in update_data:
            doc['category'] = update_data['category']
        if 'metadata' in update_data:
            doc['metadata'].update(update_data['metadata'])

        doc['updated_at'] = datetime.now().isoformat()
        doc['version'] = doc.get('version', 1) + 1

        self._write(data)
        return True

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """获取文档"""
        data = self._read()
        return data["documents"].get(doc_id)

    def list_documents(self,
                      category: Optional[str] = None,
                      tags: Optional[List[str]] = None,
                      doc_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出文档，支持过滤"""
        data = self._read()
        docs = list(data["documents"].values())

        # 按分类过滤
        if category:
            docs = [d for d in docs if d.get('category') == category]

        # 按标签过滤
        if tags:
            docs = [d for d in docs if any(tag in d.get('tags', []) for tag in tags)]

        # 按文档类型过滤
        if doc_type:
            docs = [d for d in docs if d.get('doc_type') == doc_type]

        # 按更新时间排序
        return sorted(docs, key=lambda x: x.get('updated_at', x['created_at']), reverse=True)

    def search_documents(self, query: str) -> List[Dict[str, Any]]:
        """搜索文档"""
        data = self._read()
        query = query.lower()
        results = []

        for doc in data["documents"].values():
            # 在名称、标签、分类中搜索
            search_text = f"{doc['doc_name']} {' '.join(doc.get('tags', []))} {doc.get('category', '')}".lower()
            if query in search_text:
                results.append(doc)

        return sorted(results, key=lambda x: x.get('updated_at', x['created_at']), reverse=True)

    def add_tags(self, doc_id: str, tags: List[str]) -> bool:
        """添加标签"""
        data = self._read()
        if doc_id not in data["documents"]:
            return False

        doc = data["documents"][doc_id]
        existing_tags = set(doc.get('tags', []))
        new_tags = existing_tags.union(set(tags))
        doc['tags'] = list(new_tags)
        doc['updated_at'] = datetime.now().isoformat()

        self._write(data)
        return True

    def delete_document(self, doc_id: str) -> bool:
        """删除文档"""
        data = self._read()
        if doc_id in data["documents"]:
            del data["documents"][doc_id]
            self._write(data)
            return True
        return False