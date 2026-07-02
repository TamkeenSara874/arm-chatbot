from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, HTTPException, status
from jose import jwt
from pydantic import BaseModel

from src.api.dependencies import AuthToken
from src.config import get_settings

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class TokenRequest(BaseModel):
    restaurant_id: int


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    restaurant_id: int
    expires_in_hours: int


@router.post("/token", response_model=TokenResponse)
async def issue_restaurant_token(
    body: TokenRequest,
    _: AuthToken,
) -> TokenResponse:
    settings = get_settings()
    if body.restaurant_id < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="restaurant_id must be a positive integer",
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
