"""
validator_loop.py — Bloco de orquestração que implementa um ciclo
Produtor → Validador com feedback iterativo.

Fluxo:
    1. Chama A(y)  → x
    2. Chama VALIDATOR(x) → ValidationResult
    3. Se válido  → retorna x como saída final.
    4. Se inválido → atualiza y com feedback de VALIDATOR e volta ao passo 1.
    5. Repete até validar ou atingir max_iterations.
"""

from typing import Any, Type
from pydantic import BaseModel

from agenticblocks.core.block import Block


# ---------------------------------------------------------------------------
# Modelos de I/O do ValidatorLoop
# ---------------------------------------------------------------------------

class ValidatorLoopInput(BaseModel):
    prompt: str


class ValidationResult(BaseModel):
    """
    Retorno esperado do bloco VALIDADOR.

    Campos:
        is_valid: True se a saída do produtor é aceita.
        feedback: Mensagem de feedback a ser enviada ao produtor em caso de falha.
                  Ignorado se is_valid=True.
    """
    is_valid: bool
    feedback: str = ""


class ValidatorLoopOutput(BaseModel):
    result: str
    iterations: int
    validated: bool


# ---------------------------------------------------------------------------
# Bloco Orquestrador
# ---------------------------------------------------------------------------

class ValidatorLoopBlock(Block[ValidatorLoopInput, ValidatorLoopOutput]):
    """
    Orquestra um ciclo de produção com feedback iterativo:

        A(y) → x  →  VALIDATOR(x)  →  válido? → fim
                                     ↑       ↓ não
                                     └── y += feedback(x)

    O Bloco Produtor deve expor input_schema() com campo `prompt: str`.
    O Bloco Validador deve expor input_schema() com campo `content: str`
    e retornar um modelo compatível com ValidationResult (is_valid, feedback).
    """

    description: str = "Loop de produção com validação iterativa e feedback."
    producer: Any   # Block[AgentInput-like, AgentOutput-like]
    validator: Any  # Block[ValidatorInput-like, ValidationResult-like]
    max_iterations: int = 5

    model_config = {"arbitrary_types_allowed": True}

    async def run(self, input: ValidatorLoopInput) -> ValidatorLoopOutput:
        current_prompt = input.prompt
        last_output_str = ""
        iterations = 0

        for iteration in range(1, self.max_iterations + 1):
            iterations = iteration
            print(f"\n[ValidatorLoop] Iteração {iteration}/{self.max_iterations}")

            # ── Passo 1: Produtor gera saída ──────────────────────────────
            producer_schema: Type[BaseModel] = self.producer.input_schema()
            producer_input = producer_schema(prompt=current_prompt)
            producer_result = await self.producer.run(input=producer_input)

            # Extrai texto: suporta AgentOutput.response, FunctionOutput.result e genérico
            if hasattr(producer_result, "response"):
                last_output_str = producer_result.response
            elif hasattr(producer_result, "result"):
                last_output_str = str(producer_result.result)
            else:
                last_output_str = str(producer_result.model_dump())

            print(f"[ValidatorLoop] Saída do produtor: {last_output_str[:120]}...")

            # ── Passo 2: Validador avalia a saída ─────────────────────────
            validator_schema: Type[BaseModel] = self.validator.input_schema()
            validator_input = validator_schema(content=last_output_str)
            validator_result = await self.validator.run(input=validator_input)

            # Normaliza o resultado do validador para (is_valid, feedback).
            # Suporta 3 formatos:
            #   a) ValidationResult                    → .is_valid / .feedback
            #   b) FunctionOutput(result={...})        → @as_tool retornando dict
            #   c) AgentOutput(response="{...}")       → LLMAgentBlock retornando JSON
            is_valid, feedback = self._extract_validation(validator_result)

            print(f"[ValidatorLoop] Válido: {is_valid}" +
                  (f" | Feedback: {feedback}" if not is_valid else ""))

            if is_valid:
                return ValidatorLoopOutput(
                    result=last_output_str,
                    iterations=iterations,
                    validated=True,
                )

            # ── Passo 3: Atualiza o prompt com o feedback ─────────────────
            current_prompt = (
                f"{input.prompt}\n\n"
                f"--- Tentativa {iteration} (rejeitada) ---\n"
                f"Sua resposta anterior foi:\n{last_output_str}\n\n"
                f"Feedback do validador:\n{feedback}\n\n"
                f"Por favor, corrija sua resposta levando em conta o feedback acima."
            )

        # Esgotou iterações sem validação
        print(f"[ValidatorLoop] Limite de {self.max_iterations} iterações atingido.")
        return ValidatorLoopOutput(
            result=last_output_str,
            iterations=iterations,
            validated=False,
        )

    @staticmethod
    def _extract_validation(result: Any) -> tuple[bool, str]:
        """
        Extrai (is_valid, feedback) de qualquer formato de saída do validador:
        - ValidationResult / qualquer BaseModel com is_valid
        - FunctionOutput(result=dict)         — @as_tool retornando dict
        - AgentOutput / qualquer BaseModel com response contendo JSON
        """
        import json, re

        # a) Modelo com is_valid diretamente (ValidationResult)
        if hasattr(result, "is_valid"):
            return result.is_valid, getattr(result, "feedback", "")

        # b) FunctionOutput(result=dict) — @as_tool retornando dict
        raw = getattr(result, "result", None)
        if isinstance(raw, dict):
            return bool(raw.get("is_valid", False)), raw.get("feedback", "")

        # c) AgentOutput / FunctionOutput(result=str) — tenta parsear JSON
        text = ""
        if hasattr(result, "response"):
            text = result.response or ""
        elif raw is not None:
            text = str(raw)

        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return bool(data.get("is_valid", False)), data.get("feedback", "")
            except json.JSONDecodeError:
                pass

        # Fallback: inválido com mensagem de erro
        return False, f"Não foi possível interpretar o resultado do validador: {text[:200]}"

