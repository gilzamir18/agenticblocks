import sys
import os

# Add local path to sys.path if not running from proper module
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import asyncio
from agentblocks.core.graph import WorkflowGraph
from agentblocks.runtime.executor import WorkflowExecutor
from agentblocks.runtime.state import NodeResult, NodeStatus
from agentblocks.blocks.llm.mock_llm import FetchDataBlock, ParseBlock, EnrichBlock, LLMCallBlock

def on_start(node_id: str) -> None:
    print(f"  → iniciando: {node_id}")

def on_end(result: NodeResult) -> None:
    icon = "✓" if result.status == NodeStatus.DONE else "✗"
    print(f"  {icon} {result.node_id} ({result.duration_ms:.1f}ms)")

async def main():
    graph = WorkflowGraph()

    # Montagem estilo Lego
    a = graph.add_block(FetchDataBlock(name="fetch"))
    b = graph.add_block(ParseBlock(name="parse"))
    c = graph.add_block(EnrichBlock(name="enrich"))
    d = graph.add_block(LLMCallBlock(name="summarize"))

    graph.connect(a, b)
    graph.connect(a, c)   # b e c rodam em paralelo
    graph.connect(b, d)
    graph.connect(c, d)

    executor = WorkflowExecutor(
        graph,
        on_node_start=on_start,
        on_node_end=on_end,
    )

    print("Executando workflow...")
    ctx = await executor.run(initial_input={"url": "https://exemplo.com"})

    final = ctx.get_output("summarize")
    print(f"\nResumo Final: {final.message if final else 'ERROR'}")

if __name__ == "__main__":
    asyncio.run(main())
