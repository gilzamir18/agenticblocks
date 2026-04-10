"""
07_validator_loop.py — Ciclo produtor→validador expresso nativamente no grafo.

O WorkflowExecutor detecta o CycleGroup e executa o loop com feedback
automático — sem nenhum bloco orquestrador especial.

Produtor  : LLMAgentBlock  →  gera um email formal
Validador : @as_tool        →  função Python que verifica estrutura e formalidade
Downstream: bloco de impressão conectado à saída do ciclo
"""

import asyncio
import os

from agenticblocks import as_tool
from agenticblocks.blocks.llm.agent import LLMAgentBlock
from agenticblocks.core.block import Block
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor
from agenticblocks.runtime.state import NodeResult, NodeStatus
from pydantic import BaseModel

MODEL = os.getenv("AGENTICBLOCKS_MODEL", "gpt-4o-mini")


# ── Validador: função pura, reutilizável em qualquer grafo ───────────────────

@as_tool
def validate_email(content: str) -> dict:
    """
    Validates that the email:
    - Has at least 3 paragraphs (separated by blank lines)
    - Uses no informal language
    Returns: {"is_valid": bool, "feedback": str}
    """
    informal_markers = [
        "oi ", "olá,", "olá!", "valeu", "vlw", "blz", "beleza",
        "galera", "pessoal", "😊", "👍",
    ]
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    found = [w for w in informal_markers if w.lower() in content.lower()]
    if found:
        return {
            "is_valid": False,
            "feedback": (
                f"Informal language detected: {found}. "
                "Use formal, professional communication."
            ),
        }

    if len(paragraphs) < 3:
        return {
            "is_valid": False,
            "feedback": (
                f"Email has only {len(paragraphs)} paragraph(s). "
                "Structure it in at least 3: opening, body, and closing."
            ),
        }

    return {"is_valid": True, "feedback": ""}


# ── Downstream bloco de saída ─────────────────────────────────────────────────

class PrintInput(BaseModel):
    # Matches AgentOutput from LLMAgentBlock
    response: str = ""
    tool_calls_made: int = 0


class PrintOutput(BaseModel):
    done: bool = True


class PrintBlock(Block[PrintInput, PrintOutput]):
    """Simply prints the final validated email."""
    name: str = "print_result"
    description: str = "Prints the final result."

    async def run(self, input: PrintInput) -> PrintOutput:
        print("\n" + "=" * 60)
        print("Final validated email:\n")
        print(input.response)
        print("=" * 60)
        return PrintOutput(done=True)


# ── Construção do grafo ───────────────────────────────────────────────────────

async def main():
    writer    = LLMAgentBlock(
        name="writer",
        model=MODEL,
        system_prompt=(
            "You are a professional corporate writer. "
            "Write formal, well-structured emails in Brazilian Portuguese."
        ),
        max_tool_calls=0,
    )
    printer   = PrintBlock(name="print_result")

    graph = WorkflowGraph()
    graph.add_block(writer)
    graph.add_block(validate_email)  # FunctionBlock from @as_tool
    graph.add_block(printer)

    # Declare the cycle: writer → validate_email
    # The executor will loop with feedback until validate_email returns is_valid=True
    graph.add_cycle(
        name="refine_email",
        edges=[("writer", "validate_email")],
        condition_block="validate_email",
        max_iterations=3,
    )

    # Connect the cycle output to a downstream node (fully homogeneous!)
    graph.connect("refine_email", "print_result")

    executor = WorkflowExecutor(
        graph,
        on_node_start=lambda nid: print(f"\n>> Starting node: {nid}"),
        on_node_end=lambda r: print(
            f"  OK {r.node_id} ({r.duration_ms:.0f}ms)"
        ) if r.status == NodeStatus.DONE else None,
    )

    print("=" * 60)
    print("Starting workflow with cycle...")
    print("=" * 60)

    ctx = await executor.run(initial_input={
        "prompt": (
            "Write an email to the team informing that Friday's meeting "
            "has been postponed to Monday at 10 AM, room 204."
        )
    })

    cycle_result = ctx.cycle_results.get("refine_email")
    if cycle_result:
        print(
            f"\n[Done] Cycles: {cycle_result.iterations} | "
            f"Validated: {cycle_result.validated}"
        )


if __name__ == "__main__":
    asyncio.run(main())
