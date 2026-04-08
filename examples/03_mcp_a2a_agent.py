import asyncio
import os

from pydantic import BaseModel
from agenticblocks.core.block import Block
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor
from agenticblocks.blocks.llm.agent import LLMAgentBlock
import sys
import os

# 1. Definindo o Agente Pesquisador (Sub-Agente a ser chamado por A2A)
class ResearcherInput(BaseModel):
    query: str

class ResearcherOutput(BaseModel):
    findings: str

class ResearcherBlock(Block[ResearcherInput, ResearcherOutput]):
    name: str = "researcher_agent"
    description: str = "Pesquisa informações em arquivo ou banco de dados dado uma query."
    
    async def run(self, input: ResearcherInput) -> ResearcherOutput:
        print(f"\n[A2A Call] 🕵️ Pesquisador invocado buscando por: '{input.query}'")
        await asyncio.sleep(0.5) # Simula o processamento pesado ou o MCP call  
        return ResearcherOutput(findings=f"Encontrei dados sobre '{input.query}': Sucesso Absoluto.")


# 2. Configurando o Grafo e o Diretor (Agent MCP)
async def main():
    # Detectamos o caminho absoluto correto para invocar o servidor Python como subprocesso
    python_exe = sys.executable
    server_path = os.path.join(os.path.dirname(__file__), "mcp_server_estoque.py")
    llm_model = "ollama/granite4:1b" #if don't work change to "gemini/gemini-3-flash-preview"

    graph = WorkflowGraph()
    
    # Instanciamos nosso bloco A2A subordinado
    researcher = ResearcherBlock()
    
    # Instanciamos o Agente Principal (Delegador)
    director_agent = LLMAgentBlock(
        name="director_agent",
        model=llm_model,
        system_prompt="Você é o Diretor. Use a ferramenta researcher_agent para buscar dados exatos sempre antes de responder.",
        tools=[researcher], # Aqui acontece a mágica (A2A Bridge e Pydantic Schema Automático)!
        max_iterations=5 # Evita loops com LLMs locais
    )
    
    graph.add_block(director_agent)
    executor = WorkflowExecutor(graph)
    llm_model = "ollama/granite4:1b"

    if llm_model.startswith("gemini"):
        if not os.getenv("GEMINI_API_KEY"):
            print("⚠️ Lembrete: Defina GEMINI_API_KEY para a resposta final fluir do LiteLLM.")
    
    try:
        ctx = await executor.run(initial_input={"prompt": "Por favor investigue a operação de hoje."})
        output = ctx.get_output("director_agent")
        
        print("\n[Resposta Final do LLM Diretor]:")
        print(output.response)
        print(f"- Tool Calls transparentes feitas (A2A): {output.tool_calls_made}")
    except Exception as e:
        print(f"\n[Fim Antecipado (Erro API)]: {e}")

if __name__ == "__main__":
    asyncio.run(main())
