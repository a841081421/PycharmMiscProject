from pathlib import Path
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"
METADATA_FILE = DATA_DIR / "metadata" / "documents.json"

for p in [UPLOAD_DIR, CHROMA_DIR, METADATA_FILE.parent]:
    p.mkdir(parents=True, exist_ok=True)

LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-71b399274f6a436080106f4a1151e912").strip()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat").strip()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip()

TOP_K = int(os.getenv("TOP_K", "4"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))