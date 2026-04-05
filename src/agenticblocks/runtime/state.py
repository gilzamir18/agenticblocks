from __future__ import annotations
import asyncio
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from pydantic import BaseModel

class NodeStatus(Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    DONE      = "done"
    FAILED    = "failed"
    SKIPPED   = "skipped"

@dataclass
class NodeResult:
    node_id: str
    status: NodeStatus
    output: BaseModel | None = None
    error: Exception | None = None
    duration_ms: float = 0.0

@dataclass
class ExecutionContext:
    run_id: str
    results: dict[str, NodeResult] = field(default_factory=dict)
    store: dict[str, Any] = field(default_factory=dict)

    # Note: asyncio.Lock was removed since this dataclass needs to be shallow copied sometimes
    # or just simple. We'll use a normal dictionary since it's mostly thread-safe in CPython for simple ops,
    # or handle locking in executor to keep State serializable.
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def set_result(self, node_id: str, result: NodeResult) -> None:
        async with self._lock:
            self.results[node_id] = result

    def get_output(self, node_id: str) -> BaseModel | None:
        r = self.results.get(node_id)
        return r.output if r else None

_current_ctx: ContextVar[ExecutionContext] = ContextVar("execution_ctx")

def get_ctx() -> ExecutionContext:
    return _current_ctx.get()
