from pydantic import BaseModel
from typing import TypeVar, Generic, List
from agenticblocks.core.block import Block, Input, Output

class AgentBlock(Block[Input, Output]):
    """
    Classe base para Agentes independentes do modelo cognitivo (LLM ou não-LLM).
    Um Agente é caracterizado por possuir um ciclo (loop) de decisão próprio 
    e um conjunto de "Componentes" ou "Ferramentas" (Sub-Blocos) acopláveis.
    """
    tools: List[Block] = []
    
    async def run(self, input: Input) -> Output:
        """
        Subclasses devem implementar o próprio Loop de pensamento ou heurística aqui.
        """
        raise NotImplementedError("Crie e injete o ciclo cognitivo do seu agente nesta etapa.")
