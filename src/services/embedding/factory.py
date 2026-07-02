from src.config import Settings
from src.services.embedding.base import BaseEmbedder
from src.services.embedding.openai_embedder import OpenAIEmbedder


def create_embedder(settings: Settings) -> BaseEmbedder:
    return OpenAIEmbedder(
        api_key=settings.openai_api_key,
        model=settings.openai_embed_model,
        embedding_dim=settings.embedding_dim,
        batch_size=settings.ingest_batch_size,
    )
