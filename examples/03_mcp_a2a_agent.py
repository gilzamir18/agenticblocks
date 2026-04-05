import asyncio
import os

from pydantic import BaseModel
from agenticblocks.core.block import Block
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor
from agenticblocks.blocks.llm.agent import LLMAgentBlock

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
    graph = WorkflowGraph()
    
    # Instanciamos nosso bloco A2A subordinado
    researcher = ResearcherBlock()
    
    # Instanciamos o Agente Principal (Delegador)
    director_agent = LLMAgentBlock(
        name="director_agent",
        model="gpt-4o-mini",
        system_prompt="Você é o Diretor. Use a ferramenta researcher_agent para buscar dados exatos sempre antes de responder.",
        tools=[researcher] # Aqui acontece a mágica (A2A Bridge e Pydantic Schema Automático)!
    )
    
    graph.add_block(director_agent)
    executor = WorkflowExecutor(graph)
    
    print("Iniciando Workflow de Agentes MCP/A2A...\n")
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY não encontrada. Exibiremos apenas a estrutura, mas o LiteLLM lançará exceção de key na chamada final.")
    
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
