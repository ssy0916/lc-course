import os
import secrets

from fastapi import APIRouter, Depends, Header, HTTPException

from models.schemas import KnowledgeDocumentRequest
from services.knowledge_base import knowledge_store

router = APIRouter()


def require_ingest_api_key(
    api_key: str | None = Header(default=None, alias="X-Knowledge-API-Key"),
) -> None:
    expected = os.getenv("KNOWLEDGE_INGEST_API_KEY", "").strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Knowledge document ingestion is disabled until an API key is configured.",
        )
    if api_key is None or not secrets.compare_digest(
        api_key.encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing knowledge API key.")


@router.post("/knowledge/documents")
def ingest_knowledge_document(
    request: KnowledgeDocumentRequest,
    _: None = Depends(require_ingest_api_key),
) -> dict[str, int | str]:
    chunk_count = knowledge_store.ingest(request.source, request.content)
    return {"source": request.source, "chunks_indexed": chunk_count}
