"""Verificacion del AsyncSqliteSaver para F1-T3.

Apunta al mismo `state_db/graph_checkpoints.db` que usa el lifespan de FastAPI.
Modos:
    --write : abre el saver, persiste un checkpoint bajo un thread_id fijo.
    --read  : abre el saver en un proceso distinto y verifica que recupera
              el checkpoint persistido por el modo --write.

Demostrar que `--read` recupera los datos despues de salir y volver a entrar
prueba los criterios de F1-T3:
- `state_db/graph_checkpoints.db` existe y es escrito por el checkpointer.
- El grafo persiste el thread_id entre reinicios de proceso.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

DB_PATH = Path(__file__).resolve().parents[1] / "state_db" / "graph_checkpoints.db"
THREAD_ID = "smoke-thread-1"


def _build_minimal_checkpoint() -> dict:
    """Estructura minima aceptada por el checkpointer de LangGraph."""
    return {
        "v": 1,
        "id": "smoke-checkpoint-1",
        "ts": "2026-04-29T00:00:00Z",
        "channel_values": {"messages": []},
        "channel_versions": {"messages": 1},
        "versions_seen": {"__input__": {}},
        "pending_sends": [],
    }


async def write_checkpoint() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(DB_PATH)) as saver:
        config = {"configurable": {"thread_id": THREAD_ID, "checkpoint_ns": ""}}
        checkpoint = _build_minimal_checkpoint()
        await saver.aput(config, checkpoint, {"source": "smoke"}, {})
    size = DB_PATH.stat().st_size
    print(f"[write] OK | path={DB_PATH} | size={size} bytes | thread_id={THREAD_ID}")


async def read_checkpoint() -> None:
    if not DB_PATH.exists():
        raise SystemExit(f"[read] FAIL: db not found at {DB_PATH}")
    async with AsyncSqliteSaver.from_conn_string(str(DB_PATH)) as saver:
        config = {"configurable": {"thread_id": THREAD_ID, "checkpoint_ns": ""}}
        tup = await saver.aget_tuple(config)
        if tup is None:
            raise SystemExit(f"[read] FAIL: no checkpoint for thread_id={THREAD_ID}")
        print(
            f"[read] OK | thread_id={THREAD_ID} | "
            f"checkpoint_id={tup.checkpoint['id']} | "
            f"ts={tup.checkpoint['ts']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="persistir checkpoint")
    group.add_argument("--read", action="store_true", help="leer checkpoint persistido")
    args = parser.parse_args()
    if args.write:
        asyncio.run(write_checkpoint())
    else:
        asyncio.run(read_checkpoint())


if __name__ == "__main__":
    main()
