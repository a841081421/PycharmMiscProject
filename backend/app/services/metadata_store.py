import json
from pathlib import Path
from typing import Dict, List


class MetadataStore:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        if not self.file_path.exists():
            self._write({"documents": []})

    def _read(self) -> Dict:
        with self.file_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, data: Dict) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with self.file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def list_documents(self) -> List[Dict]:
        return self._read().get("documents", [])

    def add_document(self, doc_meta: Dict) -> None:
        data = self._read()
        data["documents"].append(doc_meta)
        self._write(data)

    def delete_document(self, doc_id: str) -> bool:
        data = self._read()
        old_len = len(data["documents"])
        data["documents"] = [d for d in data["documents"] if d["doc_id"] != doc_id]
        self._write(data)
        return len(data["documents"]) < old_len

    def get_document(self, doc_id: str):
        for d in self.list_documents():
            if d["doc_id"] == doc_id:
                return d
        return None