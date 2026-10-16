import json
import re
import time
from collections import deque
from datetime import date
from io import BytesIO
from pathlib import Path

import numpy as np
from google import genai
from pypdf import PdfReader

from . import config


ID_PATTERN = re.compile(r"\b(?:id|identifier)\s*[:#-]?\s*([A-Za-z0-9_-]+)\b", re.IGNORECASE)
NAME_PATTERNS = [
    re.compile(r"\bmy name is\s+([A-Za-z][A-Za-z .'-]{0,60})\b", re.IGNORECASE),
    re.compile(r"\bcall me\s+([A-Za-z][A-Za-z .'-]{0,60})\b", re.IGNORECASE),
    re.compile(r"\bi am\s+([A-Za-z][A-Za-z .'-]{0,60})\b", re.IGNORECASE),
    re.compile(r"\bi'm\s+([A-Za-z][A-Za-z .'-]{0,60})\b", re.IGNORECASE),
]


def normalize(text: str) -> str:
    text = text.replace("\x00", " ")
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def split_text(text: str) -> list[str]:
    if not text:
        return []
    size, overlap = config.CHUNK_SIZE, config.CHUNK_OVERLAP
    if overlap >= size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            candidates = [text.rfind("\n", start, end), text.rfind(". ", start, end), text.rfind(" ", start, end)]
            boundary = max(candidates)
            if boundary > start + size * 0.55:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


class Store:
    def __init__(self):
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.items = []
        self.load()

    def load(self):
        if config.INDEX_FILE.exists():
            try:
                self.items = json.loads(config.INDEX_FILE.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self.items = []

    def save(self):
        config.INDEX_FILE.write_text(json.dumps(self.items), encoding="utf-8")

    def add(self, texts, vectors, metadata):
        providers = {item.get("metadata", {}).get("embedding_provider", "gemini") for item in self.items}
        incoming = {meta.get("embedding_provider", "gemini") for meta in metadata}
        if providers and incoming and providers != incoming:
            raise RuntimeError(
                "The knowledge base already contains embeddings from a different provider. "
                "Clear the knowledge base before switching between Gemini and local embeddings; "
                "vectors from different models cannot be compared reliably."
            )
        for text, vector, meta in zip(texts, vectors, metadata):
            self.items.append({"text": text, "vector": vector, "metadata": meta})
        self.save()

    def embedding_provider(self):
        if not self.items:
            return None
        return self.items[0].get("metadata", {}).get("embedding_provider", "gemini")

    def search(self, query_vector, k):
        if not self.items:
            return []
        q = np.asarray(query_vector, dtype=np.float32)
        qn = np.linalg.norm(q)
        if qn == 0:
            return []
        scored = []
        for item in self.items:
            v = np.asarray(item["vector"], dtype=np.float32)
            vn = np.linalg.norm(v)
            if vn == 0:
                continue
            score = float(np.dot(q, v) / (qn * vn))
            scored.append({**item, "score": score})
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:k]

    def clear(self):
        self.items = []
        if config.INDEX_FILE.exists():
            config.INDEX_FILE.unlink()

    def stats(self):
        return {"documents": len({x["metadata"]["file_name"] for x in self.items}), "chunks": len(self.items)}


class Memory:
    """Small persistent local memory for user facts and recent conversation turns."""

    def __init__(self):
        self.path = config.MEMORY_FILE
        self.data = {"facts": {}, "history": []}
        self.load()

    def load(self):
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                self.data.update(loaded)
            except (json.JSONDecodeError, OSError):
                pass

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    def remember_name(self, question: str):
        for pattern in NAME_PATTERNS:
            match = pattern.search(question.strip())
            if match:
                name = match.group(1).strip(" .,!?:;")
                if 1 <= len(name) <= 60:
                    # Avoid accidentally storing common phrases such as "I am looking...".
                    lowered = name.lower()
                    if lowered not in {"looking", "trying", "asking", "learning", "testing", "not sure"}:
                        self.data["facts"]["name"] = name
                        self.save()
                        return name
        return None

    def get_name(self):
        return self.data.get("facts", {}).get("name")

    def add_turn(self, role: str, content: str):
        history = self.data.setdefault("history", [])
        history.append({"role": role, "content": content[:4000]})
        self.data["history"] = history[-12:]
        self.save()

    def recent_history(self):
        return self.data.get("history", [])[-10:]

    def clear(self):
        self.data = {"facts": {}, "history": []}
        if self.path.exists():
            self.path.unlink()

    def stats(self):
        return {"remembered_facts": len(self.data.get("facts", {})), "conversation_turns": len(self.data.get("history", []))}


class RAG:
    def __init__(self):
        if not config.GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is missing in backend/.env")
        self.client = genai.Client(api_key=config.GOOGLE_API_KEY) if config.GOOGLE_API_KEY else None
        self.store = Store()
        self.memory = Memory()
        self.local_embedder = None
        self.active_embedding_provider = self.store.embedding_provider() or (
            "local" if config.EMBEDDING_PROVIDER == "local" else "gemini"
        )
        # Local pacing prevents the free-tier embedding RPM quota from being
        # exhausted by a large batch. Gemini meters batched embedding inputs.
        self._embedding_usage = deque()

    @staticmethod
    def _is_daily_quota_error(exc: Exception) -> bool:
        message = str(exc).lower()
        markers = (
            "embedcontentrequestsperday",
            "requestsperday",
            "requests per day",
            "perdayperuserperproject",
            "daily quota",
            "quota exceeded for metric: generativelanguage.googleapis.com/embed_content_free_tier_requests",
        )
        return any(marker in message for marker in markers)

    def _daily_quota_blocked(self):
        """Return True when Gemini daily embedding quota was previously exhausted today."""
        try:
            if not config.EMBEDDING_QUOTA_STATE_FILE.exists():
                return False
            state = json.loads(config.EMBEDDING_QUOTA_STATE_FILE.read_text(encoding="utf-8"))
            return state.get("date") == date.today().isoformat() and state.get("daily_quota_exhausted") is True
        except (OSError, json.JSONDecodeError):
            return False

    def _mark_daily_quota_exhausted(self):
        config.EMBEDDING_QUOTA_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        config.EMBEDDING_QUOTA_STATE_FILE.write_text(
            json.dumps({"date": date.today().isoformat(), "daily_quota_exhausted": True}, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _daily_quota_message(exc: Exception) -> str:
        message = str(exc)
        if "PerDay" in message or "per day" in message.lower() or "daily quota" in message.lower():
            return (
                "GEMINI_DAILY_EMBEDDING_QUOTA_EXHAUSTED: Gemini's daily embedding quota is exhausted. "
                "The application will use local embeddings instead; no further Gemini embedding retries will be made today."
            )
        return ""

    def _local_model(self):
        if self.local_embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Local embedding fallback is not installed. Run START_PROJECT.bat again "
                    "or install requirements from backend/requirements.txt."
                ) from exc
            print(f"Loading local embedding model: {config.LOCAL_EMBEDDING_MODEL}")
            self.local_embedder = SentenceTransformer(config.LOCAL_EMBEDDING_MODEL)
        return self.local_embedder

    def _embed_local(self, texts):
        if not texts:
            return []
        model = self._local_model()
        vectors = model.encode(texts, normalize_embeddings=False, show_progress_bar=False)
        return [vector.tolist() for vector in vectors]

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        message = str(exc).upper()
        return (
            "RESOURCE_EXHAUSTED" in message
            or "429" in message
            or "TOO MANY REQUESTS" in message
            or "QUOTA EXCEEDED" in message
        )

    @staticmethod
    def _retry_delay_seconds(exc: Exception, attempt: int) -> float:
        """Use Google's suggested retry delay when present, otherwise back off."""
        message = str(exc)
        match = re.search(r"retryDelay[^0-9]*(\d+)s", message, re.IGNORECASE)
        if match:
            return min(float(match.group(1)) + 1.0, config.EMBEDDING_MAX_RETRY_DELAY_SECONDS)
        return min(
            config.EMBEDDING_RETRY_BASE_SECONDS * (2 ** attempt),
            config.EMBEDDING_MAX_RETRY_DELAY_SECONDS,
        )

    def _wait_for_embedding_quota(self, item_count):
        """Pace embedding inputs so the local process stays below the RPM quota."""
        limit = config.EMBEDDING_RPM_LIMIT
        window = config.EMBEDDING_RPM_WINDOW_SECONDS
        if item_count > limit:
            raise RuntimeError(
                f"Embedding batch has {item_count} items but EMBEDDING_RPM_LIMIT is {limit}. "
                "Set EMBEDDING_RPM_LIMIT to at least the batch size."
            )

        while True:
            now = time.monotonic()
            while self._embedding_usage and now - self._embedding_usage[0][0] >= window:
                self._embedding_usage.popleft()

            used = sum(count for _, count in self._embedding_usage)
            if used + item_count <= limit:
                self._embedding_usage.append((now, item_count))
                return

            wait = max(0.1, window - (now - self._embedding_usage[0][0]) + 0.5)
            print(
                f"Embedding quota pacing: {used}/{limit} inputs used in the last "
                f"{window:.0f}s. Waiting {wait:.0f}s before the next batch..."
            )
            time.sleep(wait)

    def _embed_batch(self, batch, batch_number, total_batches):
        """Embed one Gemini batch, stopping immediately on a daily-quota response."""
        last_error = None
        for attempt in range(config.EMBEDDING_MAX_RETRIES + 1):
            try:
                self._wait_for_embedding_quota(len(batch))
                return self.client.models.embed_content(
                    model=config.EMBEDDING_MODEL,
                    contents=batch,
                )
            except Exception as exc:
                last_error = exc
                if self._is_daily_quota_error(exc):
                    self._mark_daily_quota_exhausted()
                    raise RuntimeError(self._daily_quota_message(exc)) from exc
                if not self._is_rate_limit_error(exc) or attempt >= config.EMBEDDING_MAX_RETRIES:
                    break
                delay = self._retry_delay_seconds(exc, attempt)
                print(f"Embedding batch {batch_number}/{total_batches} hit a temporary Gemini rate limit. Retrying in {delay:.0f}s (attempt {attempt + 1}/{config.EMBEDDING_MAX_RETRIES})...")
                time.sleep(delay)
        raise RuntimeError(
            f"Embedding batch {batch_number}/{total_batches} failed after {config.EMBEDDING_MAX_RETRIES + 1} attempts: {last_error}"
        ) from last_error

    def embed(self, texts, progress_callback=None, provider=None):
        """Create embeddings in batches of at most 100 and retry 429 quota errors.

        Gemini has a maximum batch size of 100 embedding inputs. The free tier also
        limits the number of embedding API requests per minute, so a multi-batch PDF
        can receive HTTP 429 even when every individual batch is valid. Rate-limit
        errors are retried with the server-provided delay (or exponential backoff).
        """
        if not texts:
            return []

        provider = provider or self.active_embedding_provider
        if provider == "local":
            return self._embed_local(texts)

        batch_size = min(config.EMBEDDING_BATCH_SIZE, 100, config.EMBEDDING_RPM_LIMIT)
        total_batches = (len(texts) + batch_size - 1) // batch_size
        all_embeddings = []

        for batch_number, start in enumerate(range(0, len(texts), batch_size), start=1):
            batch = texts[start:start + batch_size]
            response = self._embed_batch(batch, batch_number, total_batches)

            embeddings = [item.values for item in response.embeddings]
            if len(embeddings) != len(batch):
                raise RuntimeError(
                    f"Embedding batch {batch_number}/{total_batches} returned "
                    f"{len(embeddings)} embeddings for {len(batch)} items."
                )

            all_embeddings.extend(embeddings)

            if progress_callback:
                progress_callback(batch_number, total_batches, len(batch), len(all_embeddings), len(texts))

        return all_embeddings

    def index_pdf(self, file_name: str, data: bytes) -> int:
        reader = PdfReader(BytesIO(data))
        texts, metadata = [], []
        for page_no, page in enumerate(reader.pages, start=1):
            text = normalize(page.extract_text() or "")
            for chunk in split_text(text):
                texts.append(chunk)
                metadata.append({"file_name": file_name, "page": page_no})
        if not texts:
            return 0

        requested_provider = config.EMBEDDING_PROVIDER
        existing_provider = self.store.embedding_provider()
        if existing_provider:
            provider = existing_provider
        elif requested_provider == "local":
            provider = "local"
        elif requested_provider == "auto" and self._daily_quota_blocked():
            provider = "local"
        else:
            provider = "gemini"

        def report_progress(batch_number, total_batches, batch_count, completed, total):
            print(f"[{file_name}] Embedding batch {batch_number}/{total_batches} ({completed}/{total} chunks)")

        try:
            if provider == "gemini":
                try:
                    vectors = self.embed(texts, progress_callback=report_progress, provider="gemini")
                except Exception as exc:
                    is_daily_quota = self._is_daily_quota_error(exc) or "GEMINI_DAILY_EMBEDDING_QUOTA_EXHAUSTED" in str(exc)
                    can_fallback = requested_provider == "auto" and is_daily_quota
                    if not can_fallback:
                        raise
                    if existing_provider and existing_provider != "local":
                        raise RuntimeError(
                            "Gemini's daily embedding quota is exhausted, but the existing knowledge base "
                            "uses Gemini embeddings. To switch safely, clear the knowledge base and upload "
                            "the PDFs again; local and Gemini vectors cannot be mixed."
                        ) from exc
                    self._mark_daily_quota_exhausted()
                    print(f"[{file_name}] Gemini daily embedding quota exhausted. Falling back to local embeddings now and for the rest of today.")
                    provider = "local"
                    vectors = self.embed(texts, progress_callback=report_progress, provider="local")
            else:
                vectors = self.embed(texts, progress_callback=report_progress, provider="local")
        except Exception as exc:
            raise RuntimeError(
                f"Could not create embeddings for {file_name}. The document was not added to the knowledge base. {exc}"
            ) from exc

        for meta in metadata:
            meta["embedding_provider"] = provider
        self.store.add(texts, vectors, metadata)
        self.active_embedding_provider = provider
        return len(texts)

    def _looks_like_pdf_question(self, question: str) -> bool:
        q = question.lower()
        keywords = ("pdf", "document", "page", "chapter", "section", "according to", "uploaded", "file", "id", "identifier", "record")
        return any(word in q for word in keywords)

    def _exact_id_is_missing(self, question: str, results) -> bool:
        match = ID_PATTERN.search(question)
        if not match:
            return False
        requested = match.group(1).lower()
        context = "\n".join(r["text"] for r in results).lower()
        return not re.search(rf"\b(?:id|identifier)\s*[:#-]?\s*{re.escape(requested)}\b", context)

    def _general_answer(self, question: str):
        name = self.memory.get_name()
        memory_text = f"The user's remembered name is {name}." if name else "No user name is currently remembered."
        history = "\n".join(f"{x['role']}: {x['content']}" for x in self.memory.recent_history())
        prompt = f"""You are a friendly general-purpose AI assistant that also supports PDF questions.

{memory_text}

Recent conversation:
{history or '(none)'}

User message:
{question}

Respond naturally and helpfully. For greetings, be warm and conversational. Use remembered user facts when relevant. If the user asks about information that is not in the PDFs, do not pretend it is in the PDFs; instead explain politely that it was not found and offer to help generally. Keep the response concise unless the user asks for detail."""
        response = self.client.models.generate_content(model=config.CHAT_MODEL, contents=prompt)
        return (response.text or "").strip() or "Sure — how can I help?"

    def ask(self, question: str):
        question = question.strip()
        remembered_name = self.memory.remember_name(question)

        # Direct memory questions should work even when no PDF is uploaded.
        if re.search(r"\b(what(?:'s| is) my name|do you remember my name|who am i)\b", question, re.IGNORECASE):
            name = self.memory.get_name()
            answer = f"Your name is {name}. 😊" if name else "I don't know your name yet. Tell me by saying, for example, 'My name is Dev'."
            self.memory.add_turn("user", question)
            self.memory.add_turn("assistant", answer)
            return answer, []

        if remembered_name:
            answer = f"Nice to meet you, {remembered_name}! 😊 I'll remember your name for our future messages."
            self.memory.add_turn("user", question)
            self.memory.add_turn("assistant", answer)
            return answer, []

        # Greetings and normal conversation should not require a PDF.
        casual = re.match(r"^(hi|hello|hey|good morning|good afternoon|good evening|thanks|thank you|how are you|what can you do)\b", question, re.IGNORECASE)
        if casual:
            answer = self._general_answer(question)
            self.memory.add_turn("user", question)
            self.memory.add_turn("assistant", answer)
            return answer, []

        if not self.store.items:
            answer = self._general_answer(question)
            self.memory.add_turn("user", question)
            self.memory.add_turn("assistant", answer)
            return answer, []

        provider = self.store.embedding_provider() or self.active_embedding_provider
        qvec = self.embed([question], provider=provider)[0]
        results = self.store.search(qvec, config.TOP_K)
        if not results:
            answer = self._general_answer(question)
            self.memory.add_turn("user", question)
            self.memory.add_turn("assistant", answer)
            return answer, []

        best_score = results[0]["score"]
        if best_score < config.RELEVANCE_THRESHOLD or self._exact_id_is_missing(question, results):
            answer = self._general_answer(question)
            self.memory.add_turn("user", question)
            self.memory.add_turn("assistant", answer)
            return answer, []

        context = "\n\n---\n\n".join(
            f"[SOURCE {i}] File: {r['metadata']['file_name']} | Page: {r['metadata']['page']}\n{r['text']}"
            for i, r in enumerate(results, 1)
        )
        name = self.memory.get_name()
        prompt = f"""You are a PDF question-answering assistant.

Answer the user's document-related question ONLY from the supplied PDF context. Do not invent facts or use outside knowledge for document claims. If the context does not support the answer, politely say you couldn't find it in the uploaded PDFs and suggest uploading the relevant document. The user's remembered name is {name or 'unknown'}, but do not use it as evidence for PDF facts.

PDF CONTEXT:
{context}

QUESTION:
{question}"""
        response = self.client.models.generate_content(model=config.CHAT_MODEL, contents=prompt)
        answer = (response.text or "").strip() or "I couldn't find enough information in the uploaded PDFs to answer that."
        sources = [
            {"file_name": r["metadata"]["file_name"], "page": r["metadata"]["page"], "score": round(r["score"], 3)}
            for r in results
        ]
        self.memory.add_turn("user", question)
        self.memory.add_turn("assistant", answer)
        return answer, sources
