from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, HTTPException, status
from jose import jwt
from pydantic import BaseModel

from src.api.dependencies import AuthToken, DbSession
from src.config import get_settings
from src.models.db_entities import RestaurantCredential
from src.utils.restaurant_auth import verify_restaurant_key

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class TokenRequest(BaseModel):
    restaurant_id: int
    restaurant_key: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    restaurant_id: int
    expires_in_hours: int


@router.post("/token", response_model=TokenResponse)
async def issue_restaurant_token(
    body: TokenRequest,
    _: AuthToken,
    db: DbSession,
) -> TokenResponse:
    settings = get_settings()
    if body.restaurant_id < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="restaurant_id must be a positive integer",
        )

    # API_KEY above only proves "this is a legitimate arm-chatbot client app"
    # -- it's the same for every tenant and, in the current frontend, shipped
    # to the browser. restaurant_key is the actual per-tenant check: without
    # it, anyone holding API_KEY could mint a JWT for any restaurant_id.
    credential = await db.get(RestaurantCredential, body.restaurant_id)
    if credential is None or not verify_restaurant_key(body.restaurant_key, credential.key_hash):
        # Same 401 whether the restaurant_id doesn't exist or the key is
        # wrong -- distinguishing them would let an unauthenticated caller
        # enumerate valid restaurant_ids.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid restaurant_id or restaurant_key",
        )

    expires = datetime.now(UTC) + timedelta(hours=settings.jwt_expiry_hours)
    payload = {
        "sub": f"restaurant:{body.restaurant_id}",
        "restaurant_id": body.restaurant_id,
        "exp": expires,
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return TokenResponse(
        access_token=token,
        restaurant_id=body.restaurant_id,
        expires_in_hours=settings.jwt_expiry_hours,
    )
