"""Inspecciona el estado persistido del checkpointer para un thread_id.

Uso:
    python back/scripts/inspect_thread.py phase2-smoke-persistence-thread-persist

Demuestra que el AsyncSqliteSaver persiste GraphState entre reinicios y
turnos. Imprime: numero de mensajes, snippet del primer y ultimo mensaje,
campos clave del estado (mode, user_id, current_asr, quality_attribute).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

DB_PATH = Path(__file__).resolve().parents[1] / "state_db" / "graph_checkpoints.db"


def _shorten(s: str, n: int = 200) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + "..."


async def inspect(thread_id: str) -> None:
    if not DB_PATH.exists():
        raise SystemExit(f"DB no encontrada: {DB_PATH}")
    print(f"DB: {DB_PATH}")
    print(f"thread_id: {thread_id}")
    print("-" * 70)

    async with AsyncSqliteSaver.from_conn_string(str(DB_PATH)) as saver:
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        tup = await saver.aget_tuple(config)
        if tup is None:
            print("No hay estado persistido para ese thread_id.")
            return

        state = tup.checkpoint.get("channel_values", {})
        msgs = state.get("messages") or []
        print(f"Total mensajes en GraphState.messages: {len(msgs)}")
        if msgs:
            first = msgs[0]
            last = msgs[-1]
            print(f"  [primero] {type(first).__name__}: {_shorten(getattr(first, 'content', str(first)), 160)}")
            print(f"  [ultimo]  {type(last).__name__}: {_shorten(getattr(last, 'content', str(last)), 160)}")

        print("\nCampos clave del estado:")
        for key in ("mode", "user_id", "language", "intent", "quality_attribute",
                    "current_asr", "arch_stage", "user_profile", "mode_suggestion"):
            val = state.get(key)
            print(f"  {key:20s} = {_shorten(repr(val), 120)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("thread_id", help="thread_id (= session_id) a inspeccionar")
    args = parser.parse_args()
    asyncio.run(inspect(args.thread_id))


if __name__ == "__main__":
    main()
