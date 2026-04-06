import asyncio
import time
from pydantic import BaseModel
from agenticblocks.core.block import Block
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor

# 1. Primeira etapa do pipeline - Processamento Simples
class BlockAInput(BaseModel):
    message: str

class BlockAOutput(BaseModel):
    data: str

class BlockA(Block[BlockAInput, BlockAOutput]):
    description: str = "Bloco de Teste Inicial"
    
    async def run(self, input: BlockAInput) -> BlockAOutput:
        # Apenas simula uma pequena transformação de texto
        return BlockAOutput(data=f"A processou: '{input.message}'")

# 2. Segunda etapa do pipeline - Modificação e Conclusão
class BlockBInput(BaseModel):
    data: str

class BlockBOutput(BaseModel):
    final_output: str

class BlockB(Block[BlockBInput, BlockBOutput]):
    description: str = "Bloco de Teste Final"
    
    async def run(self, input: BlockBInput) -> BlockBOutput:
        # Transforma o texto para maiúsculo
        return BlockBOutput(final_output=f"B finalizou com: {input.data.upper()}")

# 3. Execução
async def main():
    # Instanciamos os blocos básicos
    block_a = BlockA(name="block_a")
    block_b = BlockB(name="block_b")
    
    # Criamos o grafo
    graph = WorkflowGraph()
    graph.add_block(block_a)
    graph.add_block(block_b)
    
    # Conectamos A para B
    graph.connect("block_a", "block_b")
    
    # Prepara o executor
    executor = WorkflowExecutor(graph)
    print("Iniciando Workflow de Teste de Performance (Sem LLM)...")
    
    # Medimos o tempo
    start_time = time.time()
    
    # Rodamos o executor
    # Passamos o Input do primeiro bloco do grafo (neste caso, "block_a")
    ctx = await executor.run(initial_input={"message": "Testando performance de execucao nativa"})
    
    end_time = time.time()
    
    print("\n[✓] Execução finalizada!")
    print(f"⏱️  Tempo total percorrido: {end_time - start_time:.4f} segundos")
    print(f"📦 Resultado Final (Block B): {ctx.get_output('block_b').model_dump()}")

if __name__ == "__main__":
    asyncio.run(main())
