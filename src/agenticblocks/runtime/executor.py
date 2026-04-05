from __future__ import annotations
import asyncio
import time
import uuid
from typing import Callable
import networkx as nx

from agenticblocks.core.block import Block
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.state import ExecutionContext, NodeResult, NodeStatus, _current_ctx


class WorkflowExecutor:
    def __init__(
        self,
        graph: WorkflowGraph,
        on_node_start: Callable[[str], None] | None = None,
        on_node_end:   Callable[[NodeResult], None] | None = None,
    ):
        self.graph = graph
        self.on_node_start = on_node_start
        self.on_node_end   = on_node_end

    async def run(
        self,
        initial_input: dict | None = None,
        run_id: str | None = None,
    ) -> ExecutionContext:
        ctx = ExecutionContext(run_id=run_id or str(uuid.uuid4()))
        ctx.store["__input__"] = initial_input or {}

        token = _current_ctx.set(ctx)
        try:
            self._validate()
            waves = self._build_waves()
            for wave in waves:
                await self._execute_wave(wave, ctx)
        finally:
            _current_ctx.reset(token)

        return ctx

    def _validate(self) -> None:
        g = self.graph.graph
        if not nx.is_directed_acyclic_graph(g):
            cycles = list(nx.simple_cycles(g))
            raise ValueError(f"Ciclos detectados no grafo: {cycles}")
        self.graph.validate_connections()

    def _build_waves(self) -> list[list[str]]:
        g = self.graph.graph
        in_degree = {n: g.in_degree(n) for n in g.nodes}
        waves: list[list[str]] = []

        remaining = set(g.nodes)
        while remaining:
            wave = [n for n in remaining if in_degree[n] == 0]
            if not wave:
                raise RuntimeError("Dependências circulares detectadas.")
            waves.append(wave)
            for node in wave:
                remaining.remove(node)
                for successor in g.successors(node):
                    in_degree[successor] -= 1
        return waves

    async def _execute_wave(self, wave: list[str], ctx: ExecutionContext) -> None:
        tasks = [self._execute_node(node_id, ctx) for node_id in wave]
        await asyncio.gather(*tasks, return_exceptions=False)

    async def _execute_node(self, node_id: str, ctx: ExecutionContext) -> None:
        block: Block = self.graph.graph.nodes[node_id]["block"]

        if self.on_node_start:
            self.on_node_start(node_id)

        t0 = time.monotonic()
        try:
            input_data = self._collect_inputs(node_id, ctx)
            # Create instance of the input schema
            input_schema_class = block.input_schema()
            
            # Simple conversion logic
            if issubclass(input_schema_class, dict):
                 typed_input = input_data
            else:
                 typed_input = input_schema_class(**input_data)
                 
            output = await block.run(typed_input)
            result = NodeResult(
                node_id=node_id,
                status=NodeStatus.DONE,
                output=output,
                duration_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception as exc:
            result = NodeResult(
                node_id=node_id,
                status=NodeStatus.FAILED,
                error=exc,
                duration_ms=(time.monotonic() - t0) * 1000,
            )
            # await to save the failure
            await ctx.set_result(node_id, result)
            raise exc

        await ctx.set_result(node_id, result)
        if self.on_node_end:
            self.on_node_end(result)

    def _collect_inputs(self, node_id: str, ctx: ExecutionContext) -> dict:
        g = self.graph.graph
        predecessors = list(g.predecessors(node_id))

        if not predecessors:
            return ctx.store.get("__input__", {})

        if len(predecessors) == 1:
            prev_output = ctx.get_output(predecessors[0])
            return prev_output.model_dump() if prev_output else {}

        merged: dict = {}
        for pred in predecessors:
            prev_output = ctx.get_output(pred)
            if prev_output:
                merged.update(prev_output.model_dump())
        return merged
