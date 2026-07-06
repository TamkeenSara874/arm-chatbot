"""Unit tests for src/services/vector/qdrant_store.py's collection-creation
race handling.

Regression coverage for a real bug found deploying to a fresh Qdrant Cloud
cluster: ensure_collections() previously did a plain
`if not collection_exists(): create_collection()` check, which has a TOCTOU
gap -- with multiple concurrent uvicorn workers starting against a brand-new
(empty) Qdrant instance, two workers can both see "missing" and both call
create_collection(), and the loser gets an uncaught 409 Conflict that crashed
the whole app's startup.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

from src.services.vector.qdrant_store import _create_collection_if_missing, ensure_collections


def _unexpected_response(status_code: int) -> UnexpectedResponse:
    return UnexpectedResponse(
        status_code=status_code,
        reason_phrase="Conflict" if status_code == 409 else "Error",
        content=b'{"status":{"error":"boom"}}',
        headers=httpx.Headers(),
    )


class TestCreateCollectionIfMissing:
    @pytest.mark.asyncio
    async def test_skips_create_when_already_exists(self) -> None:
        qdrant = MagicMock()
        qdrant.collection_exists = AsyncMock(return_value=True)
        qdrant.create_collection = AsyncMock()

        await _create_collection_if_missing(qdrant, "review_chunks")

        qdrant.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_creates_when_missing(self) -> None:
        qdrant = MagicMock()
        qdrant.collection_exists = AsyncMock(return_value=False)
        qdrant.create_collection = AsyncMock()

        await _create_collection_if_missing(qdrant, "review_chunks", vectors_config="x")

        qdrant.create_collection.assert_awaited_once_with(
            collection_name="review_chunks", vectors_config="x"
        )

    @pytest.mark.asyncio
    async def test_tolerates_409_from_concurrent_creation(self) -> None:
        """The exact race: another worker created it between our exists()
        check and this create_collection() call."""
        qdrant = MagicMock()
        qdrant.collection_exists = AsyncMock(return_value=False)
        qdrant.create_collection = AsyncMock(side_effect=_unexpected_response(409))

        await _create_collection_if_missing(qdrant, "review_chunks")

        # No exception raised -- that's the assertion.

    @pytest.mark.asyncio
    async def test_reraises_non_409_errors(self) -> None:
        qdrant = MagicMock()
        qdrant.collection_exists = AsyncMock(return_value=False)
        qdrant.create_collection = AsyncMock(side_effect=_unexpected_response(500))

        with pytest.raises(UnexpectedResponse):
            await _create_collection_if_missing(qdrant, "review_chunks")


class TestEnsureCollections:
    @pytest.mark.asyncio
    async def test_creates_all_four_collections_when_none_exist(self) -> None:
        settings = MagicMock(
            qdrant_url="http://localhost:6333",
            qdrant_api_key="",
            qdrant_collection_reviews="review_chunks",
            qdrant_collection_corrections="correction_embeddings",
            qdrant_collection_session_memory="session_memory",
            qdrant_collection_chat_cache="chat_cache",
            embedding_dim=1536,
        )
        fake_client = MagicMock()
        fake_client.collection_exists = AsyncMock(return_value=False)
        fake_client.create_collection = AsyncMock()
        fake_client.close = AsyncMock()

        with patch("qdrant_client.AsyncQdrantClient", return_value=fake_client):
            await ensure_collections(settings)

        assert fake_client.create_collection.await_count == 4
        fake_client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_closes_client_even_on_failure(self) -> None:
        settings = MagicMock(
            qdrant_url="http://localhost:6333",
            qdrant_api_key="",
            qdrant_collection_reviews="review_chunks",
            qdrant_collection_corrections="correction_embeddings",
            qdrant_collection_session_memory="session_memory",
            qdrant_collection_chat_cache="chat_cache",
            embedding_dim=1536,
        )
        fake_client = MagicMock()
        fake_client.collection_exists = AsyncMock(return_value=False)
        fake_client.create_collection = AsyncMock(side_effect=_unexpected_response(500))
        fake_client.close = AsyncMock()

        with (
            patch("qdrant_client.AsyncQdrantClient", return_value=fake_client),
            pytest.raises(UnexpectedResponse),
        ):
            await ensure_collections(settings)

        fake_client.close.assert_awaited_once()
