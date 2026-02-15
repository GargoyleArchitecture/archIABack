import argparse
import sys

from fastapi.testclient import TestClient

from src.main import app


def run_test(query: str, session_id: str, expect_source: str | None, expect_phrase: str | None) -> int:
    client = TestClient(app)

    resp = client.post(
        "/message",
        data={
            "message": query,
            "session_id": session_id,
        },
    )
    if resp.status_code != 200:
        print(f"[FAIL] HTTP {resp.status_code}: {resp.text}")
        return 2

    data = resp.json()
    rag = data.get("rag_trace") or {}
    attempted = bool(rag.get("attempted"))
    hit_count = int(rag.get("hit_count") or 0)
    sources = rag.get("sources") or []
    end_msg = data.get("endMessage") or ""

    print("[INFO] rag_trace.attempted =", attempted)
    print("[INFO] rag_trace.hit_count =", hit_count)
    print("[INFO] rag_trace.sources   =", sources)

    if not attempted:
        print("[FAIL] RAG was not attempted for this query.")
        return 3

    if expect_source:
        if not any(expect_source.lower() in s.lower() for s in sources):
            print(f"[FAIL] Expected source fragment not found: {expect_source}")
            return 4

    if expect_phrase:
        if expect_phrase.lower() not in end_msg.lower():
            print(f"[FAIL] Expected phrase not found in response: {expect_phrase}")
            return 5

    print("[PASS] RAG access test succeeded.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Backend-only RAG access test.")
    parser.add_argument("--query", required=True, help="User question to send to /message")
    parser.add_argument("--session-id", default="rag-test-session", help="Session id for the request")
    parser.add_argument("--expect-source", help="Substring expected in rag_trace.sources")
    parser.add_argument("--expect-phrase", help="Substring expected in endMessage")
    args = parser.parse_args()

    return run_test(
        query=args.query,
        session_id=args.session_id,
        expect_source=args.expect_source,
        expect_phrase=args.expect_phrase,
    )


if __name__ == "__main__":
    raise SystemExit(main())
