"""
07_validator_loop.py — Producer -> Validator cycle expressed natively in the graph.

The WorkflowExecutor detects the CycleGroup and runs the feedback loop
automatically — no special orchestrator block required.

Producer  : LLMAgentBlock  ->  generates a formal email
Validator : @as_tool        ->  pure Python function that checks structure and formality
Result    : read directly from the ExecutionContext after execution
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import get_model

from agenticblocks import as_tool
from agenticblocks.blocks.llm.agent import LLMAgentBlock
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor


# -- Validador: funcao pura, reutilizavel em qualquer grafo ------------------

@as_tool
def validate_email(content: str) -> dict:
    """
    Validates that the email:
    - Has at least 3 paragraphs (separated by blank lines)
    - Uses no informal language
    Returns: {"is_valid": bool, "feedback": str}
    """
    informal_markers = [
        "oi ", "ola,", "ola!", "valeu", "vlw", "blz", "beleza",
        "galera", "pessoal",
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


# -- Graph construction -------------------------------------------------------

async def main():
    writer = LLMAgentBlock(
        name="writer",
        model=get_model(),
        system_prompt=(
            "You are a professional corporate writer. "
            "Write formal, well-structured emails in Brazilian Portuguese."
        ),
        max_tool_calls=0,
    )

    graph = WorkflowGraph()
    graph.add_block(writer)
    graph.add_block(validate_email)  # FunctionBlock from @as_tool

    # Declare the cycle: writer -> validate_email
    # sequence=[] is the shortcut for linear chains (equivalent to edges=[("writer", "validate_email")])
    graph.add_cycle(
        name="refine_email",
        sequence=["writer", "validate_email"],
        condition_block="validate_email",
        max_iterations=3,
    )

    executor = WorkflowExecutor(graph)

    print("=" * 60)
    print("Starting workflow with cycle...")
    print("=" * 60)

    ctx = await executor.run(initial_input={
        "prompt": (
            "Write an email to the team informing that Friday's meeting "
            "has been postponed to Monday at 10 AM, room 204."
        )
    })

    # The cycle result is available directly in the context —
    # no extra orchestrator block needed.
    cr = ctx.cycle_results.get("refine_email")
    if cr:
        print(f"\n[Done] Iterations: {cr.iterations} | Validated: {cr.validated}")
        print("\n--- Final email ---\n")
        print(cr.output.response)


if __name__ == "__main__":
    asyncio.run(main())
