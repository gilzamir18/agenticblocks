"""
07_validator_loop.py — Ciclo produtor->validador expresso nativamente no grafo.

O WorkflowExecutor detecta o CycleGroup e executa o loop com feedback
automatico -- sem nenhum bloco orquestrador especial.

Produtor  : LLMAgentBlock  ->  gera um email formal
Validador : @as_tool        ->  funcao Python que verifica estrutura e formalidade
Resultado : lido diretamente do ExecutionContext apos execucao
"""

import asyncio
import os

from agenticblocks import as_tool
from agenticblocks.blocks.llm.agent import LLMAgentBlock
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor

MODEL = os.getenv("AGENTICBLOCKS_MODEL", "ollama/granite4:1b")


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


# -- Construcao do grafo ------------------------------------------------------

async def main():
    writer = LLMAgentBlock(
        name="writer",
        model=MODEL,
        system_prompt=(
            "You are a professional corporate writer. "
            "Write formal, well-structured emails in Brazilian Portuguese."
        ),
        max_tool_calls=0,
    )

    graph = WorkflowGraph()
    graph.add_block(writer)
    graph.add_block(validate_email)  # FunctionBlock from @as_tool

    # Declara o ciclo: writer -> validate_email
    # O executor itera com feedback ate validate_email retornar is_valid=True
    graph.add_cycle(
        name="refine_email",
        edges=[("writer", "validate_email")],
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

    # O resultado do ciclo esta disponivel diretamente no contexto --
    # nenhum bloco extra necessario.
    cr = ctx.cycle_results.get("refine_email")
    if cr:
        print(f"\n[Done] Iterations: {cr.iterations} | Validated: {cr.validated}")
        print("\n--- Final email ---\n")
        print(cr.output.response)


if __name__ == "__main__":
    asyncio.run(main())
