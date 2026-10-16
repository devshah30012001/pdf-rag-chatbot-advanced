from pathlib import Path
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from uuid import uuid4
from datetime import datetime, timezone

UPLOAD_JOBS = {}

from . import config
from .rag import RAG

app = FastAPI(title="PDF AI Assistant", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
rag = None


def get_rag():
    global rag
    if rag is None:
        if not config.GOOGLE_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="Gemini API key is missing. Add GOOGLE_API_KEY=... to backend/.env and restart the application."
            )
        try:
            rag = RAG()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    return rag


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(config.FRONTEND_FILE, headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache", "Expires": "0"})


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "api_key_configured": bool(config.GOOGLE_API_KEY),
    }


@app.get("/api/stats")
def stats():
    return get_rag().store.stats()


def _new_job(file_name: str):
    job_id = uuid4().hex
    UPLOAD_JOBS[job_id] = {
        "job_id": job_id,
        "file_name": file_name,
        "status": "queued",
        "progress": 0,
        "message": "Waiting to start...",
        "chunks_created": 0,
        "error": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return job_id


def _run_upload_job(job_id: str, file_name: str, data: bytes):
    job = UPLOAD_JOBS[job_id]
    job["status"] = "processing"
    job["message"] = "Extracting PDF text and creating embeddings..."
    try:
        service = get_rag()
        count = service.index_pdf(file_name, data)
        if not count:
            raise RuntimeError(
                "No readable text was found. Scanned/image-only PDFs need OCR."
            )
        job.update(
            status="completed",
            progress=100,
            message=f"Indexed successfully — {count} chunks created.",
            chunks_created=count,
        )
    except Exception as exc:
        job.update(
            status="failed",
            progress=0,
            message="Indexing failed.",
            error=str(exc),
        )


@app.post("/api/upload")
async def upload(background_tasks: BackgroundTasks, files: list[UploadFile] = File(...)):
    max_bytes = config.MAX_FILE_SIZE_MB * 1024 * 1024
    jobs = []

    for file in files:
        name = (file.filename or "").strip()
        if not name.lower().endswith(".pdf"):
            raise HTTPException(400, f"{name or 'Selected file'} is not a PDF.")

        data = await file.read()
        if len(data) > max_bytes:
            raise HTTPException(400, f"{name} is larger than {config.MAX_FILE_SIZE_MB} MB.")

        safe_name = Path(name).name
        job_id = _new_job(safe_name)
        background_tasks.add_task(_run_upload_job, job_id, safe_name, data)
        jobs.append({"job_id": job_id, "file_name": safe_name})

    return {"message": "Upload accepted", "jobs": jobs}


@app.get("/api/upload/status/{job_id}")
def upload_status(job_id: str):
    job = UPLOAD_JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Upload job not found.")
    return job


@app.post("/api/ask")
def ask(request: AskRequest):
    try:
        answer, sources = get_rag().ask(request.question.strip())
        return {"answer": answer, "sources": sources}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Could not answer the question: {exc}") from exc


@app.get("/api/memory")
def memory():
    service = get_rag()
    return {"facts": service.memory.data.get("facts", {}), "stats": service.memory.stats()}


@app.delete("/api/memory")
def clear_memory():
    get_rag().memory.clear()
    return {"message": "Conversation memory cleared"}


@app.delete("/api/knowledge-base")
def clear():
    get_rag().store.clear()
    return {"message": "Knowledge base cleared"}
