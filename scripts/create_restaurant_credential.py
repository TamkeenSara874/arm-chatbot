"""Provision (or reissue) the credential required to mint a JWT for one restaurant.

Usage: python scripts/create_restaurant_credential.py <restaurant_id>

Prints the plaintext key ONCE -- only its SHA-256 hash is stored, so there is
no way to recover it later. Hand the printed key to that restaurant's
owner/operator out of band. Rerunning for the same restaurant_id overwrites
the existing credential, immediately invalidating the old key.
"""

import asyncio
import sys

from src.models.db_entities import RestaurantCredential
from src.services.database import get_session_factory
from src.utils.restaurant_auth import generate_restaurant_key, hash_restaurant_key


async def main(restaurant_id: int) -> None:
    key = generate_restaurant_key()
    key_hash = hash_restaurant_key(key)

    session_factory = get_session_factory()
    async with session_factory() as db:
        existing = await db.get(RestaurantCredential, restaurant_id)
        if existing is not None:
            existing.key_hash = key_hash
        else:
            db.add(RestaurantCredential(restaurant_id=restaurant_id, key_hash=key_hash))
        await db.commit()

    print(f"restaurant_id={restaurant_id}")
    print(f"restaurant_key={key}")
    print("Store this now -- it cannot be recovered, only reissued by rerunning this script.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/create_restaurant_credential.py <restaurant_id>")
        sys.exit(1)
    asyncio.run(main(int(sys.argv[1])))
