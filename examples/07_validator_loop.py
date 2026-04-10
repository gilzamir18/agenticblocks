"""
07_validator_loop.py — ValidatorLoop sem classes extras.

Produtor  : LLMAgentBlock  →  escreve um email formal
Validador : @as_tool        →  função Python simples que checa a estrutura

O ValidatorLoopBlock orquestra o ciclo automaticamente.
"""

import asyncio
import os

from agenticblocks import as_tool
from agenticblocks.blocks.llm.agent import LLMAgentBlock
from agenticblocks.blocks.flow.validator_loop import ValidatorLoopBlock, ValidatorLoopInput

MODEL = os.getenv("AGENTICBLOCKS_MODEL", "gpt-4o-mini")


# ── Validador: função pura, sem classe ───────────────────────────────────────
@as_tool
def validar_email(content: str) -> dict:
    """
    Valida se o email:
    - Tem pelo menos 3 parágrafos (separados por linha em branco)
    - Não usa linguagem informal (checagem simples por palavras-chave)
    Retorna: {"is_valid": bool, "feedback": str}
    """
    informal = ["oi", "ola", "olá", "valeu", "vlw", "blz", "beleza", "galera", "pessoal", "😊", "👍"]
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    found_informal = [w for w in informal if w.lower() in content.lower()]
    if found_informal:
        return {
            "is_valid": False,
            "feedback": f"Linguagem informal detectada: {found_informal}. Use comunicação formal e profissional.",
        }

    if len(paragraphs) < 3:
        return {
            "is_valid": False,
            "feedback": (
                f"O email tem apenas {len(paragraphs)} parágrafo(s). "
                "Estruture em pelo menos 3: introdução, corpo e conclusão."
            ),
        }

    return {"is_valid": True, "feedback": ""}


# ── Produtor: LLMAgentBlock padrão ───────────────────────────────────────────
escritor = LLMAgentBlock(
    name="EscritorDeEmails",
    model=MODEL,
    system_prompt=(
        "Você é um redator corporativo. "
        "Escreva emails formais, claros e bem estruturados em português do Brasil."
    ),
    max_tool_calls=0,
)


# ── Loop produtor → validador ─────────────────────────────────────────────────
loop = ValidatorLoopBlock(
    name="email_loop",
    producer=escritor,
    validator=validar_email,   # @as_tool — sem classe extra
    max_iterations=3,
)


async def main():
    entrada = ValidatorLoopInput(
        prompt=(
            "Escreva um email para a equipe avisando que a reunião de sexta-feira "
            "foi adiada para segunda-feira às 10h, sala 204."
        )
    )

    print("=" * 60)
    print("Iniciando ValidatorLoop...")
    print("=" * 60)

    resultado = await loop.run(input=entrada)

    print("\n" + "=" * 60)
    print(f"Iterações: {resultado.iterations} | Validado: {resultado.validated}")
    print("=" * 60)
    print("\n📧 Email Final:\n")
    print(resultado.result)


if __name__ == "__main__":
    asyncio.run(main())
