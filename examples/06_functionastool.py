from agenticblocks import as_tool
from agenticblocks.blocks.llm.agent import LLMAgentBlock

# async def — chamado diretamente com await
@as_tool
async def buscar_clima(cidade: str) -> str:
    """Retorna o clima atual de uma cidade."""
    return f"Ensolarado em {cidade}"

# def síncrono — roda em thread pool via asyncio.to_thread
@as_tool(name="hora_atual", description="Retorna a hora atual do sistema.")
def hora_atual() -> str:
    import datetime
    return datetime.datetime.now().strftime("%H:%M:%S")

# Uso idêntico ao de qualquer outro Block
agente = LLMAgentBlock(
    name="assistente",
    tools=[buscar_clima, hora_atual]
)
