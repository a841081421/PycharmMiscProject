from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader
from docx import Document as DocxDocument
from bs4 import BeautifulSoup
import requests


class DocumentLoader:
    @staticmethod
    def load_file(file_path: Path) -> Dict:
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return DocumentLoader._load_pdf(file_path)
        if suffix == ".txt":
            return DocumentLoader._load_txt(file_path)
        if suffix == ".docx":
            return DocumentLoader._load_docx(file_path)

        raise ValueError(f"不支持的文件类型: {suffix}")

    @staticmethod
    def load_web(url: str) -> Dict:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = " ".join(soup.get_text(separator="\n").split())
        return {
            "source": url,
            "doc_name": url,
            "text": text,
            "type": "web"
        }

    @staticmethod
    def _load_pdf(file_path: Path) -> Dict:
        reader = PdfReader(str(file_path))
        pages: List[str] = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        text = "\n".join(pages).strip()
        return {
            "source": str(file_path),
            "doc_name": file_path.name,
            "text": text,
            "type": "pdf"
        }

    @staticmethod
    def _load_txt(file_path: Path) -> Dict:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        return {
            "source": str(file_path),
            "doc_name": file_path.name,
            "text": text,
            "type": "txt"
        }

    @staticmethod
    def _load_docx(file_path: Path) -> Dict:
        doc = DocxDocument(str(file_path))
        text = "\n".join([p.text for p in doc.paragraphs if p.text]).strip()
        return {
            "source": str(file_path),
            "doc_name": file_path.name,
            "text": text,
            "type": "docx"
        }