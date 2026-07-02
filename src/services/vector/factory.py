from src.config import Settings
from src.services.vector.base import BaseVectorStore
from src.services.vector.qdrant_store import QdrantStore


def create_vector_store(settings: Settings) -> BaseVectorStore:
    return QdrantStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )
