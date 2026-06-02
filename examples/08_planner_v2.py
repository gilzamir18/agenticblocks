"""08_planner_v2.py — TasteFast chatbot refactored with PlannerChatBlock.

Compared to 08_planner.py, ``make_turn_block`` has been replaced by the
library-provided ``PlannerChatBlock``, which is a first-class Block.  The
shared ``chat_history`` list is injected at construction time; no closure
boilerplate is needed.
"""

import asyncio
from agenticblocks import as_tool, PlannerChatBlock
from agenticblocks.blocks.llm.agent import LLMAgentBlock
from agenticblocks.blocks.patterns.plan_executor import PlanExecutorBlock
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor


# ── Domain data ────────────────────────────────────────────────────────────

quantidade = {
    "cheeseburger": 10,
    "Pão de Queijo": 10,
    "Suco de laranja": 1,
    "Suco de goiaba": 0,
    "Suco de Limão": 2,
}

preco = {
    "cheeseburger": 12.0,
    "Pão de Queijo": 8.0,
    "Suco de laranja": 9.0,
    "Suco de goiaba": 8.0,
    "Suco de Limão": 6.0,
}


# ── Domain tools ───────────────────────────────────────────────────────────

@as_tool(name="consultar_item")
def consultar_item(item: str) -> str:
    item_upper = item.strip().upper()
    for k in quantidade:
        if item_upper == k.upper() or item_upper in k.upper() or k.upper() in item_upper:
            qty = quantidade[k]
            price = preco[k]
            if qty > 0:
                return f"{k}: {qty} unidade(s) disponível(is), R$ {price:.2f} cada."
            return f"{k}: indisponível no momento."
    return f"Item '{item}' não encontrado no cardápio."


@as_tool(name="get_cardapio")
def get_cardapio() -> str:
    lines = ["Cardápio do TasteFast:"]
    for item in quantidade:
        qty = quantidade[item]
        price = preco[item]
        status = f"{qty} disponível(is)" if qty > 0 else "indisponível"
        lines.append(f"  - {item} | R$ {price:.2f} | {status}")
    return "\n".join(lines)


# ── Conversation state ─────────────────────────────────────────────────────

chat_history: list[str] = []


# ── Output tool ────────────────────────────────────────────────────────────

@as_tool(name="print_agent_response", description="Delivers the final reply to the customer.")
def print_agent_response(response: str) -> str:
    print(f"Agent: {response}")
    chat_history.append(f"Agent: {response}")
    return "ok"


# ── Validator ──────────────────────────────────────────────────────────────

_VALID_ITEMS = {k.lower() for k in quantidade}
_SUSPICIOUS = [
    "frango", "salada", "alface", "tomate", "batata", "refrigerante",
    "pizza", "lasanha", "sushi", "sorvete",
]


def validate_reply(reply: str, observations: list) -> tuple[bool, str]:
    if not reply or not reply.strip():
        return False, "Empty response."
    reply_low = reply.lower()
    invasao = [
        s for s in _SUSPICIOUS
        if s in reply_low and not any(s in item for item in _VALID_ITEMS)
    ]
    if invasao:
        return False, (
            f"Response mentions items NOT in the menu: {invasao}. "
            "Use ONLY items from REAL DATA TO USE."
        )
    return True, "ok"


# ── Chat flow controls ─────────────────────────────────────────────────────

@as_tool(name="get_user_input")
def get_user_input() -> dict:
    print("Você: ", end="", flush=True)
    return {"user_message": input().strip()}


_EXITS = {"sair", "tchau", "fim", "/bye", "exit", "quit"}


@as_tool(name="check_done")
def check_done(last_message: str = "") -> dict:
    for line in reversed(chat_history):
        if line.startswith("User:"):
            last_user = line[len("User:"):].strip().lower()
            if last_user in _EXITS:
                return {"is_valid": True, "feedback": "encerrado"}
            break
    return {"is_valid": False, "feedback": "continuar"}


def build_chat_prompt(orig, iteration, producer, feedback):
    return feedback or ""


# ── Main ───────────────────────────────────────────────────────────────────

async def main():
    executor_agent = LLMAgentBlock(
        name="executor_agent",
        model="ollama/mistral-nemo:latest",
        description="Writes the final reply to the customer.",
        system_prompt=(
            "Você é a voz do TasteFast (lanchonete).\n"
            "REGRAS ABSOLUTAS:\n"
            "1. Mencione SOMENTE itens, preços e disponibilidade de 'DADOS REAIS A USAR'.\n"
            "2. Copie nomes e preços literalmente.\n"
            "3. Se 'DADOS REAIS A USAR' está vazio, peça mais informações.\n"
            "4. Entregue a resposta chamando 'print_agent_response' UMA única vez.\n"
            "5. Seja breve, amigável, em português brasileiro."
        ),
        tools=[print_agent_response],
        model_kargs={"temperature": 0.3},
    )

    planner_agent = LLMAgentBlock(
        name="planner_agent",
        model="ollama/mistral-nemo:latest",
        description="Generates JSON plan.",
        system_prompt=(
            "Você é o Planejador do TasteFast. NÃO fala com o cliente. NÃO chama "
            "ferramentas. Sua ÚNICA saída é um JSON válido.\n\n"
            "Formato OBRIGATÓRIO (apenas JSON, nada mais):\n"
            "{\n"
            '  "thought": "raciocínio breve",\n'
            '  "steps": [ {"action": "<nome>", "args": { ... }} ]\n'
            "}\n\n"
            "AÇÕES DISPONÍVEIS:\n"
            '  - "get_cardapio":   args = {}.\n'
            '  - "consultar_item": args = {"item": "<nome>"}.\n'
            '  - "reply":          args = {"message": "<briefing>"}. SEMPRE o último passo.\n\n'
            "REGRAS:\n"
            "1. SEMPRE termine com 'reply'.\n"
            "2. Nunca invente ações fora da lista.\n"
            "3. Saída: APENAS o JSON, sem ```, sem comentários."
        ),
        tools=[],
        max_iterations=1,
        model_kargs={"temperature": 0.0},
    )

    plan_executor = PlanExecutorBlock(
        executor_agent=executor_agent,
        tools=[get_cardapio, consultar_item],
        validator_fn=validate_reply,
        max_reply_retries=2,
        reply_prompt_template=(
            "BRIEFING DO PLANNER:\n{briefing}\n\n"
            "DADOS REAIS A USAR (copie nomes e preços EXATAMENTE):\n{observations}\n\n"
            "HISTÓRICO RECENTE:\n{history}\n\n"
            "{extra_instruction}\n"
            "Agora chame 'print_agent_response' UMA vez com a mensagem final ao cliente."
        ),
    )

    # ── Single declaration — no closure needed ─────────────────────────────
    turn_block = PlannerChatBlock(
        planner=planner_agent,
        executor=plan_executor,
        history=chat_history,          # shared list: block appends "User:" / "Agent:" lines
        user_prefix="User",
        agent_prefix="Agent",
        planner_prompt_template=(
            "HISTÓRICO RECENTE:\n{history}\n\n"
            "MENSAGEM DO USUÁRIO: {user_message}\n\n"
            "Produza APENAS o JSON do plano. Sem texto antes ou depois."
        ),
    )

    print("Bem-vindo ao TasteFast! Digite 'sair' para encerrar.\n")

    graph = WorkflowGraph()
    graph.add_block(get_user_input)
    graph.add_block(turn_block)
    graph.add_block(check_done)

    graph.add_cycle(
        name="chat_loop",
        sequence=["get_user_input", "plan_and_execute_turn", "check_done"],
        condition_block="check_done",
        max_iterations=1000,
        augment_fn=build_chat_prompt,
    )

    executor = WorkflowExecutor(graph, verbose=False)
    await executor.run(initial_input={"prompt": "início"})


if __name__ == "__main__":
    asyncio.run(main())
