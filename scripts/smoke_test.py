"""Post-deployment smoke test. Run after 'make run' and 'make seed'.

Usage: python scripts/smoke_test.py [--url http://localhost:8000]
"""

import asyncio
import argparse
import sys
import httpx


async def main(base_url: str, api_key: str) -> None:
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=30) as client:

        # 1. Liveness
        r = await client.get("/health")
        assert r.status_code == 200, f"liveness failed: {r.text}"
        print("PASS /health")

        # 2. Readiness
        r = await client.get("/health/ready")
        assert r.status_code == 200, f"readiness failed: {r.text}"
        body = r.json()
        assert body["status"] == "ready", f"not ready: {body}"
        print("PASS /health/ready")

        print("\nSmoke test passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--api-key", default="change-me-local-dev-key")
    args = ap.parse_args()
    asyncio.run(main(args.url, args.api_key))
