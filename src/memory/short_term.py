"""Short-term memory: the LangGraph checkpointer thread.

Each conversation is one `thread_id` in an AsyncSqliteSaver (data/checkpoints.db). The full graph
state, including the message history, the running summary and session facts, is checkpointed
after every node, so later turns of the same conversation see earlier ones even across process
restarts, and an approval interrupt can be resumed.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.config import DATA_DIR

CHECKPOINT_DB = DATA_DIR / "checkpoints.db"


@asynccontextmanager
async def open_checkpointer(path=CHECKPOINT_DB) -> AsyncIterator[AsyncSqliteSaver]:
    path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        await saver.setup()
        yield saver


def thread_config(session_id: str, run_id: str, recursion_limit: int) -> dict:
    # NB: never put `run_id` in config metadata; LangGraph treats it as the run identity and
    # replays the previous result for a second turn on the same thread.
    return {"configurable": {"thread_id": session_id}, "recursion_limit": recursion_limit,
            "metadata": {"copilot_run_id": run_id, "session_id": session_id}}
