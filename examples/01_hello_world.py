import sys
import os



import asyncio
from pydantic import BaseModel
from agentblocks.core.block import Block
from agentblocks.core.graph import WorkflowGraph
from agentblocks.runtime.executor import WorkflowExecutor

# 1. Definindo as Entradas e Saídas (Pydantic Models)
class HelloInput(BaseModel):
    name: str

class HelloOutput(BaseModel):
    greeting: str

# 2. Criando o Bloco (a peça de Lego)
class HelloWorldBlock(Block[HelloInput, HelloOutput]):
    name: str = "default_hello"
    
    async def run(self, input: HelloInput) -> HelloOutput:
        # Lógica real do bloco
        msg = f"Hello, {input.name}! Welcome to AgentBlocks."
        print(f"[Executando Bloco] Processando mensagem para {input.name}...")
        return HelloOutput(greeting=msg)

# 3. Executando o Workflow
async def main():
    # Instanciando o grafo
    graph = WorkflowGraph()

    # Adicionando a peça ao "tabuleiro" (grafo)
    bloco_hello = HelloWorldBlock(name="hello_world_1")
    graph.add_block(bloco_hello)

    # Configurando o executor
    executor = WorkflowExecutor(graph)

    print("Iniciando Workflow...\n")
    
    # Execução inicializando com os dados primários (initial_input)
    # Aqui, a chave "name" será validada pelo esquema HelloInput!
    ctx = await executor.run(initial_input={"name": "Mundo Compositivo"})

    # O Contexto armazena todo o estado da execução. Podemos coletar por Node_ID.
    resultado_final = ctx.get_output("hello_world_1")
    
    print("\n[Resultado Final]:")
    print(resultado_final.model_dump())

if __name__ == "__main__":
    asyncio.run(main())
