import os
from pathlib import Path
from dotenv import load_dotenv

# backend/
BACKEND_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = BACKEND_DIR / ".env"

# Explicitly load the backend/.env file.
load_dotenv(dotenv_path=ENV_FILE, override=True)

GOOGLE_API_KEY = (
    os.getenv("GOOGLE_API_KEY", "").strip()
    or os.getenv("GEMINI_API_KEY", "").strip()
)

CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.1-flash-lite").strip()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001").strip()
# auto uses Gemini until a daily embedding quota is exhausted, then falls back
# to local sentence-transformers embeddings for that document.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "auto").strip().lower()
if EMBEDDING_PROVIDER not in {"auto", "gemini", "local"}:
    EMBEDDING_PROVIDER = "auto"
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip()
TOP_K = int(os.getenv("TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "20"))
# Gemini embedding requests support at most 100 contents per batch.
EMBEDDING_BATCH_SIZE = min(int(os.getenv("EMBEDDING_BATCH_SIZE", "100")), 100)
if EMBEDDING_BATCH_SIZE < 1:
    EMBEDDING_BATCH_SIZE = 100
# Free-tier Gemini embedding quota is request-based, so multi-batch PDFs can
# receive HTTP 429 even though each batch contains <=100 inputs.
EMBEDDING_MAX_RETRIES = max(int(os.getenv("EMBEDDING_MAX_RETRIES", "6")), 0)
EMBEDDING_RETRY_BASE_SECONDS = max(float(os.getenv("EMBEDDING_RETRY_BASE_SECONDS", "5")), 1.0)
EMBEDDING_MAX_RETRY_DELAY_SECONDS = max(float(os.getenv("EMBEDDING_MAX_RETRY_DELAY_SECONDS", "70")), 5.0)
# Gemini free-tier embedding RPM is consumed by individual texts in a batch.
# Keep a small safety margin below the published 100 RPM quota.
EMBEDDING_RPM_LIMIT = max(int(os.getenv("EMBEDDING_RPM_LIMIT", "90")), 1)
EMBEDDING_RPM_WINDOW_SECONDS = max(float(os.getenv("EMBEDDING_RPM_WINDOW_SECONDS", "60")), 1.0)

DATA_DIR = BACKEND_DIR / "data"
INDEX_FILE = DATA_DIR / "index.json"
FRONTEND_FILE = BACKEND_DIR.parent / "frontend" / "index.html"

RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.40"))
MEMORY_FILE = DATA_DIR / "memory.json"
EMBEDDING_QUOTA_STATE_FILE = DATA_DIR / "embedding_quota_state.json"
