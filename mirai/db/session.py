from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

# Default URL, overridden by MiraiConfig at startup
_engine = None
_async_session = None


def _get_engine(database_url: str = "sqlite+aiosqlite:///./mirai.db"):
    global _engine, _async_session
    if _engine is None:
        _engine = create_async_engine(database_url, echo=False)
        _async_session = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)
    return _engine


async def init_db(database_url: str = "sqlite+aiosqlite:///./mirai.db") -> None:
    engine = _get_engine(database_url)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    if _async_session is None:
        _get_engine()
    assert _async_session is not None
    async with _async_session() as session:
        yield session
