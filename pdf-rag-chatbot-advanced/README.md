# PDF AI Assistant — RAG Chatbot with Memory

A Windows-friendly full-stack PDF question-answering chatbot built with **FastAPI**, **Google Gemini**, **NumPy**, **pypdf**, and a single HTML/CSS/JavaScript frontend.

The application combines two capabilities:

1. **Grounded PDF Q&A** — upload one or more PDFs, retrieve the most relevant text chunks, and generate an answer using Gemini from that retrieved context.
2. **General conversation + lightweight memory** — greetings and normal chat work without a PDF, while simple user facts such as a name and recent conversation turns are persisted locally.

> This project is designed as an assignment/demo application. The local JSON vector store and local memory are intentionally simple and easy to understand.

---

## 1. Features

### PDF / RAG features
- Upload one or multiple PDF files.
- Validate PDF extension and maximum file size.
- Extract text page-by-page using `pypdf`.
- Normalize extracted text.
- Split text into overlapping chunks.
- Generate embeddings with `gemini-embedding-001`.
- Persist vectors and metadata in a local JSON index.
- Retrieve relevant chunks using cosine similarity with NumPy.
- Apply a relevance threshold before answering document questions.
- Detect requested IDs that are not present in retrieved context.
- Generate grounded answers with `gemini-3.1-flash-lite`.
- Return source filename, page, and similarity score.
- Show document/chunk statistics in the UI.
- Clear the knowledge base without clearing chatbot memory.

### Chat / memory features
- Friendly greetings such as `Hi`, `Hello`, and `How are you?`.
- General-purpose answers when a question is not document-specific.
- Remember a simple user name, for example:
  - `My name is Dev`
  - `Call me Dev`
- Recall the remembered name:
  - `What is my name?`
  - `Who am I?`
- Keep the last few conversation turns as local context.
- Persist memory in `backend/data/memory.json`.
- Clear memory independently from the PDF knowledge base.

### UI / developer experience
- Bright, colorful responsive frontend.
- Drag-and-drop PDF upload.
- Friendly error/toast messages.
- Source pills for grounded answers.
- One-click Windows launcher.
- Automatic virtual-environment creation.
- Automatic dependency installation/checking.
- Gemini API key setup through `backend/.env`.
- No Node.js/npm requirement.
- Backend serves the frontend directly.
- `Cache-Control: no-store` headers help avoid stale frontend files during development.

---

## 2. Architecture

```text
                         Browser
                            |
                            | HTTP
                            v
                  +---------------------+
                  |      FastAPI        |
                  |     backend/app     |
                  +----------+----------+
                             |
              +--------------+---------------+
              |                              |
              v                              v
       PDF Upload / RAG                Chat + Memory
              |                              |
              v                              v
          pypdf text                  memory.json
              |
              v
        Chunking + overlap
              |
              v
      Gemini embeddings
              |
              v
       index.json vectors
              |
              v
     Cosine similarity search
              |
              v
       Relevant PDF context
              |
              v
       Gemini chat model
              |
              v
       Answer + PDF sources
```

The frontend is located at `frontend/index.html` and is served by FastAPI from `/`.

---

## 3. Project Structure

```text
pdf-rag-chatbot-final/
├── START_PROJECT.bat
├── STOP_PROJECT.bat
├── README.md
├── IMPLEMENTATION.md
├── frontend/
│   └── index.html
└── backend/
    ├── requirements.txt
    ├── .env.example
    ├── SETUP_API_KEY.bat
    ├── START_BACKEND.bat
    ├── app/
    │   ├── __init__.py
    │   ├── config.py
    │   ├── rag.py
    │   └── main.py
    └── data/
        ├── .gitkeep
        ├── index.json          # created after PDF indexing
        └── memory.json         # created after memory/chat activity
```

`index.json` and `memory.json` are runtime-generated files and should not contain secrets.

---

## 4. Requirements

- Windows 10/11
- Python 3.11+ available as `py` or `python`
- Internet access for package installation and Gemini API requests
- A Gemini API key

Node.js and npm are **not required**.

---

## 5. Quick Start — Windows

### Option A: One-click startup

1. Extract the project ZIP.
2. Open the project folder.
3. Double-click:

```text
START_PROJECT.bat
```

The launcher:

1. Detects Python.
2. Creates `backend/.venv` if required.
3. Installs/checks packages from `backend/requirements.txt`.
4. Checks `backend/.env`.
5. Prompts for a Gemini API key if one is missing.
6. Starts FastAPI.
7. Waits for `/api/health` to become available.
8. Opens `http://127.0.0.1:8000`.

### Option B: Manual startup

```bat
cd backend
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

Never commit the real `backend/.env` file or expose the Gemini API key in screenshots, source control, or chat.

---

## 6. Environment Configuration

`backend/.env.example` contains the supported settings:

```env
GOOGLE_API_KEY=PASTE_YOUR_GEMINI_API_KEY_HERE
CHAT_MODEL=gemini-3.1-flash-lite
EMBEDDING_MODEL=gemini-embedding-001
TOP_K=5
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
MAX_FILE_SIZE_MB=20
```

The application also accepts `GEMINI_API_KEY` as an alias for `GOOGLE_API_KEY`.

Additional application settings:

```env
RELEVANCE_THRESHOLD=0.40
```

`RELEVANCE_THRESHOLD` controls whether the top retrieved result is considered sufficiently relevant for a PDF-grounded answer.

---

## 7. How to Use

### Upload PDFs

1. Open the application.
2. Select one or more PDF files, or drag them into the upload area.
3. Wait for indexing to finish.
4. The document/chunk counters update after successful indexing.

### Ask a PDF question

Example:

```text
What is the information for ID 2?
```

The application retrieves relevant chunks and answers from those chunks. Sources show the PDF filename and page number.

### Ask about data that is not in the PDF

If the PDF contains IDs `1`, `2`, and `3` and you ask:

```text
What is the data for ID 006?
```

the application checks whether the requested ID appears in the retrieved context. If it does not, it avoids presenting an unrelated retrieved chunk as the answer and gives a friendly response explaining that the requested information was not found.

### Use normal chat

```text
Hi
```

or:

```text
My name is Dev
```

Then:

```text
What is my name?
```

The chatbot can return the remembered name from local memory.

---

## 8. API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Serve frontend |
| GET | `/api/health` | Health/API-key status |
| GET | `/api/stats` | PDF/document/chunk statistics |
| POST | `/api/upload` | Upload and index PDFs |
| POST | `/api/ask` | Ask a PDF/general chatbot question |
| GET | `/api/memory` | Read current memory summary |
| DELETE | `/api/memory` | Clear remembered facts/history |
| DELETE | `/api/knowledge-base` | Clear PDF index |

### Ask request

```json
{
  "question": "What is the information for ID 2?"
}
```

### Ask response

```json
{
  "answer": "...",
  "sources": [
    {
      "file_name": "example.pdf",
      "page": 1,
      "score": 0.812
    }
  ]
}
```

General-chat and memory-only responses return an empty `sources` array because they are not grounded in PDF chunks.

---

## 9. RAG Processing Flow

```text
PDF
 ↓
pypdf extraction
 ↓
Page-aware text
 ↓
Normalization
 ↓
Overlapping chunks
 ↓
Gemini embedding model
 ↓
Local JSON vector index

User question
 ↓
Question embedding
 ↓
Cosine similarity
 ↓
Top-K chunks
 ↓
Relevance / ID validation
 ↓
Gemini grounded prompt
 ↓
Answer + sources
```

For implementation details, see **[IMPLEMENTATION.md](IMPLEMENTATION.md)**.

---

## 10. Memory Model

Memory is intentionally lightweight and local.

```json
{
  "facts": {
    "name": "Dev"
  },
  "history": [
    {"role": "user", "content": "My name is Dev"},
    {"role": "assistant", "content": "Nice to meet you, Dev!"}
  ]
}
```

The memory implementation currently:

- Extracts names from common phrases.
- Stores the name under `facts.name`.
- Stores up to 12 total recent turns.
- Uses the latest 10 turns when constructing a general-chat prompt.
- Saves data after each memory update.
- Does not add memory content to the PDF vector index.

This is **application-level local memory**, not a sophisticated long-term semantic memory system.

---

## 11. PDF Limitations

`pypdf` extracts embedded/selectable text. Scanned or image-only PDFs may contain no extractable text.

If a PDF is scanned, the application may report:

```text
No readable text was found in the PDF(s). Scanned/image-only PDFs need OCR.
```

An OCR pipeline can be added later using an OCR engine before chunking.

---

## 12. Troubleshooting

### Old UI is still displayed

If the new frontend appears in Incognito but not in a normal Chrome window, the issue is likely browser cache.

Try:

```text
Ctrl + Shift + R
```

If necessary, clear site data from Chrome DevTools → Application → Storage.

The backend also serves the frontend with `no-cache` headers to reduce stale development content.

### API key missing

Confirm this file exists:

```text
backend/.env
```

and contains:

```env
GOOGLE_API_KEY=YOUR_NEW_KEY
```

Restart the application after changing the key.

### Port 8000 is already in use

Stop the previous backend using:

```text
STOP_PROJECT.bat
```

or terminate the process that owns port `8000` before starting again.

### No tests / no PDF answer

Check:

- The PDF is readable/selectable.
- The upload completed successfully.
- `/api/stats` reports documents and chunks.
- The question actually relates to uploaded content.
- The requested record/ID exists in the PDF.

---

## 13. Security Notes

This project is a local/demo application and is not production-hardened.

For production, add at least:

- Authentication and authorization.
- Per-user or per-tenant knowledge-base isolation.
- Secure server-side API key management.
- HTTPS.
- File malware scanning.
- Strong file-type/content validation.
- Rate limiting.
- Request size limits.
- Structured database/vector store.
- Object storage for PDFs.
- Background processing for large documents.
- Audit logging.
- Secret management through a proper secret manager.
- Access controls for memory data.

The current CORS configuration is permissive for local development.

---

## 14. Production Upgrade Path

A production architecture could replace the simple local JSON components with:

```text
Frontend
   ↓
Authenticated API
   ↓
Background document processor
   ↓
Object storage + PostgreSQL/pgvector or Qdrant
   ↓
Embedding service
   ↓
Retriever / reranker
   ↓
LLM
   ↓
Grounded answer + citations
```

The memory layer could also move from `memory.json` to a user-scoped database with explicit retention and privacy controls.

---

## 15. License / Assignment Use

This project is structured as an educational/assignment-friendly implementation. Adapt the code and configuration to the requirements of your environment.


## Embedding quota and retry handling

PDF chunks are sent to Gemini in batches of at most 100 inputs because the embedding API rejects larger batch requests. Gemini's free tier can also enforce a request-per-minute quota for embeddings. A large PDF can therefore receive HTTP 429 (`RESOURCE_EXHAUSTED`) even when the batch size is valid.

This version automatically retries 429/quota errors using Google's suggested retry delay when available, with exponential backoff as a fallback. The retry settings are configurable with `EMBEDDING_MAX_RETRIES`, `EMBEDDING_RETRY_BASE_SECONDS`, and `EMBEDDING_MAX_RETRY_DELAY_SECONDS`.

Important: retries can make a large upload take several minutes on a free-tier quota. This is a Gemini API quota limitation, not a PDF parsing error. For frequent or large-document ingestion, use a project/API plan with a higher embedding quota.

### Embedding quota handling

Gemini embedding batch requests have a maximum of 100 inputs, but the free-tier embedding RPM quota can also count the individual texts contained in a batch. This project therefore defaults to `EMBEDDING_RPM_LIMIT=90` to leave a small safety margin. The backend automatically paces batches and retries 429 quota responses using Google's suggested retry delay when available.

PDF indexing is processed as a background job. The browser receives a job ID immediately and polls `/api/upload/status/{job_id}` while indexing continues. This prevents long embedding waits from appearing as a browser-level `Failed to fetch` error.

For large PDFs on the Gemini free tier, indexing may take a few minutes because the quota is intentionally respected rather than bypassed.


## Local embedding fallback and daily quota detection

`EMBEDDING_PROVIDER=auto` uses Gemini embeddings first. When Gemini explicitly reports daily quota exhaustion (for example, `EmbedContentRequestsPerDayPerUserPerProject-FreeTier`, a daily-quota message, or a server retry delay of at least one hour), the backend stops retrying that daily-limit error immediately and switches that document to local `sentence-transformers/all-MiniLM-L6-v2` embeddings. The local model is downloaded on first use, so the first fallback may need internet access and can take longer.

You can set `EMBEDDING_PROVIDER=local` to use local embeddings for all documents, or `EMBEDDING_PROVIDER=gemini` to disable fallback. Gemini is still used for answer generation, so `GOOGLE_API_KEY` is still required for chat responses. Local embeddings do not consume Gemini embedding quota.

**Do not mix embedding models in one knowledge base.** Vectors from Gemini and the local model use different dimensions/embedding spaces. If the knowledge base already contains Gemini vectors and daily quota is exhausted, the app refuses to silently mix them. Clear the knowledge base, then re-upload the PDFs with `EMBEDDING_PROVIDER=local` (or `auto`) to rebuild a consistent index. The app stores the provider with each indexed chunk and uses that same provider for question embeddings.


### Gemini embedding quota fallback

`EMBEDDING_PROVIDER=auto` uses Gemini embeddings until Gemini reports a daily embedding quota exhaustion. At that point the application records the quota state for the current day and immediately switches to the local `sentence-transformers/all-MiniLM-L6-v2` model. It does not perform repeated retries against a daily quota. `EMBEDDING_PROVIDER=local` always uses the local model. Local and Gemini vectors are never mixed in the same knowledge base.
