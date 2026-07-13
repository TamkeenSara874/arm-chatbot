"""One-off runner: drives dashboard_grounded_qa_dataset.csv through the live
/chat/query API and records the real answer/timing/cost/evidence for each
question. Reuses tests/e2e/eval_client.py's SSE-handling client rather than
re-implementing it. Not a pytest module -- run directly:

    python tests/fixtures/run_dashboard_qa_dataset.py
"""

from __future__ import annotations

import asyncio
import csv
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.e2e.eval_client import send_query  # noqa: E402

RESTAURANT_ID = 1
JWT = Path(sys.argv[1]).read_text().strip() if len(sys.argv) > 1 else None
BASE_URL = "http://localhost:8000"
DATASET_PATH = Path(__file__).parent / "dashboard_grounded_qa_dataset.csv"
RESULTS_PATH = Path(__file__).parent / "dashboard_grounded_qa_results.csv"

# Fixed column order, not derived from whatever keys happen to be on the first
# row -- bug_status is hand-curated (added out-of-band, not by this script)
# and every fresh result dict below carries it forward from the existing row
# rather than dropping it, so a re-run never silently strips manually-written
# annotations or crashes DictWriter on a fieldname mismatch between old rows
# (with bug_status) and freshly-overwritten ones (without it).
FIELDNAMES = [
    "id",
    "category",
    "question_type",
    "question",
    "grounded_in",
    "answer",
    "evidence_count",
    "evidence_sources",
    "confidence",
    "caveats",
    "complexity",
    "model_used",
    "cached",
    "server_latency_ms",
    "wall_clock_ms",
    "cost_usd",
    "session_id",
    "message_id",
    "error",
    "bug_status",
]


async def main() -> None:
    if not JWT:
        print("Usage: python run_dashboard_qa_dataset.py <path-to-jwt-file>")
        sys.exit(1)

    with open(DATASET_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    only_ids: set[str] | None = None
    if len(sys.argv) > 2:
        only_ids = set(sys.argv[2].split(","))
        rows = [r for r in rows if r["id"] in only_ids]

    # Merge into any existing results file rather than overwriting it, so
    # re-running just the failed ids doesn't discard the ones that already
    # succeeded (and were already paid for).
    existing: dict[str, dict] = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                existing[r["id"]] = r

    # rate_limit_chat is 10/minute -- space requests well under that (one
    # every 7s) regardless of how fast an individual request itself completes,
    # since a fast cache-hit/count-query response otherwise lets the loop
    # burn through the whole per-minute budget in seconds.
    results = []
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        for i, row in enumerate(rows):
            if i > 0:
                await asyncio.sleep(7.0)
            qid, category, question = row["id"], row["category"], row["question"]
            print(f"[{qid}/{len(rows)}] {category}: {question[:70]}...", flush=True)

            session_resp = await client.post(
                "/api/v1/chat/sessions",
                json={"restaurant_id": RESTAURANT_ID},
                headers={"Authorization": f"Bearer {JWT}"},
            )
            session_resp.raise_for_status()
            session_id = session_resp.json()["session_id"]

            # Carry forward any hand-curated bug_status note already on this
            # row rather than dropping it when the row gets overwritten below.
            prior_bug_status = existing.get(qid, {}).get("bug_status", "")

            wall_start = time.perf_counter()
            try:
                result = await send_query(client, JWT, session_id, RESTAURANT_ID, question)
                wall_ms = round((time.perf_counter() - wall_start) * 1000)
                results.append(
                    {
                        "id": qid,
                        "category": category,
                        "question_type": row["question_type"],
                        "question": question,
                        "grounded_in": row["grounded_in"],
                        "answer": result.answer,
                        "evidence_count": len(result.evidence),
                        "evidence_sources": json.dumps([e.get("source") for e in result.evidence]),
                        "confidence": result.confidence,
                        "caveats": result.caveats or "",
                        "complexity": result.complexity,
                        "model_used": result.model_used,
                        "cached": result.cached,
                        "server_latency_ms": result.latency_ms,
                        "wall_clock_ms": wall_ms,
                        "cost_usd": result.cost_usd,
                        "session_id": session_id,
                        "message_id": result.message_id,
                        "error": "",
                        "bug_status": prior_bug_status,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                wall_ms = round((time.perf_counter() - wall_start) * 1000)
                print(f"    ERROR: {exc}", flush=True)
                results.append(
                    {
                        "id": qid,
                        "category": category,
                        "question_type": row["question_type"],
                        "question": question,
                        "grounded_in": row["grounded_in"],
                        "answer": "",
                        "evidence_count": 0,
                        "evidence_sources": "",
                        "confidence": "",
                        "caveats": "",
                        "complexity": "",
                        "model_used": "",
                        "cached": "",
                        "server_latency_ms": "",
                        "wall_clock_ms": wall_ms,
                        "cost_usd": "",
                        "session_id": session_id,
                        "message_id": "",
                        "error": str(exc),
                        "bug_status": prior_bug_status,
                    }
                )

            # Merge this run's results into `existing` and write incrementally
            # so partial progress survives an interruption.
            for r in results:
                existing[r["id"]] = r
            merged = sorted(existing.values(), key=lambda r: int(r["id"]))
            with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as out:
                writer = csv.DictWriter(out, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerows(merged)

    final_rows = sorted(existing.values(), key=lambda r: int(r["id"]))

    def _to_float(v: object) -> float:
        try:
            return float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0

    total_cost = sum(_to_float(r["cost_usd"]) for r in final_rows)
    errors = sum(1 for r in final_rows if r["error"])
    print(
        f"\nDone. {len(final_rows)} total questions on file, {errors} with errors, total cost ${total_cost:.4f}"
    )
    print(f"Results written to {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
