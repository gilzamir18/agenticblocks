from __future__ import annotations
import asyncio
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from pydantic import BaseModel


class NodeStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"
    SKIPPED = "skipped"


@dataclass
class NodeResult:
    node_id:     str
    status:      NodeStatus
    output:      BaseModel | None = None
    error:       Exception | None = None
    duration_ms: float = 0.0


@dataclass
class CycleResult:
    """
    Result of a CycleGroup after full execution.

    Stored in the ExecutionContext under the cycle name.
    The `output` field holds the last output of the producer block
    (entry_block or the last non-condition block in the cycle).
    """
    cycle_name:  str
    iterations:  int
    validated:   bool
    output:      BaseModel | None = None


@dataclass
class ExecutionContext:
    run_id:        str
    results:       dict[str, NodeResult]  = field(default_factory=dict)
    cycle_results: dict[str, CycleResult] = field(default_factory=dict)
    store:         dict[str, Any]         = field(default_factory=dict)

    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def set_result(self, node_id: str, result: NodeResult) -> None:
        async with self._lock:
            self.results[node_id] = result

    async def set_cycle_result(self, cycle_name: str, result: CycleResult) -> None:
        async with self._lock:
            self.cycle_results[cycle_name] = result

    def get_output(self, node_id: str) -> BaseModel | None:
        """Returns the output for a node OR cycle group by name."""
        # Check cycle results first
        if node_id in self.cycle_results:
            return self.cycle_results[node_id].output
        r = self.results.get(node_id)
        return r.output if r else None


_current_ctx: ContextVar[ExecutionContext] = ContextVar("execution_ctx")


def get_ctx() -> ExecutionContext:
    return _current_ctx.get()
