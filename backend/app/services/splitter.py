from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List


class TextSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]
        )

    def split(self, text: str) -> List[str]:
        return [c.strip() for c in self.splitter.split_text(text) if c.strip()]