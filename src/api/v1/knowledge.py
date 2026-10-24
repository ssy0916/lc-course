from fastapi import APIRouter

from models.schemas import KnowledgeDocumentRequest
from services.knowledge_base import knowledge_store

router = APIRouter()


@router.post("/knowledge/documents")
def ingest_knowledge_document(request: KnowledgeDocumentRequest) -> dict[str, int | str]:
    chunk_count = knowledge_store.ingest(request.source, request.content)
    return {"source": request.source, "chunks_indexed": chunk_count}
