"""F1-T4: Verificacion de lectura cross-thread del Store.

Estos tests no requieren `pytest-asyncio`. Cada caso ejecuta su corutina via
`asyncio.run(...)` desde una funcion sincrona estandar de pytest.
"""
import asyncio

from langgraph.store.memory import InMemoryStore


def test_store_cross_thread_profile_roundtrip():
    """Escribe un perfil bajo ("user", "u1", "profile") y lo lee de vuelta sin
    referencia al thread_id del checkpointer (cross-thread por construccion)."""

    async def run():
        store = InMemoryStore()
        ns = ("user", "u1", "profile")
        payload = {"strengths": ["X"], "weaknesses": ["Y"]}
        await store.aput(ns, key="profile", value=payload)
        item = await store.aget(ns, key="profile")
        assert item is not None
        assert item.value == payload

    asyncio.run(run())


def test_store_namespaces_are_isolated_per_user():
    """Verifica que dos usuarios distintos no se ven entre si."""

    async def run():
        store = InMemoryStore()
        await store.aput(("user", "u1", "profile"), key="profile", value={"v": 1})
        await store.aput(("user", "u2", "profile"), key="profile", value={"v": 2})
        item1 = await store.aget(("user", "u1", "profile"), key="profile")
        item2 = await store.aget(("user", "u2", "profile"), key="profile")
        assert item1 is not None and item1.value == {"v": 1}
        assert item2 is not None and item2.value == {"v": 2}

    asyncio.run(run())


def test_store_routines_namespace_supported():
    """Verifica que el namespace ("user", user_id, "routines") es escribible y
    legible, en linea con la convencion documentada en docs_back/store.md."""

    async def run():
        store = InMemoryStore()
        ns = ("user", "u1", "routines")
        await store.aput(ns, key="r1", value={"title": "Reto 1"})
        item = await store.aget(ns, key="r1")
        assert item is not None
        assert item.value["title"] == "Reto 1"

    asyncio.run(run())
