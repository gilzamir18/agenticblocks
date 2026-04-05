import networkx as nx
from .block import Block
from pydantic import BaseModel

class WorkflowGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_block(self, block: Block) -> str:
        node_id = block.name
        if node_id in self.graph.nodes:
            raise ValueError(f"Já existe um bloco com o nome {node_id}")
        self.graph.add_node(node_id, block=block)
        return node_id

    def connect(self, from_id: str, to_id: str):
        if from_id not in self.graph.nodes or to_id not in self.graph.nodes:
            raise ValueError("Ambos os blocos devem existir no grafo")
        
        # Validacao iterativa
        out_schema = self.graph.nodes[from_id]["block"].output_schema()
        in_schema  = self.graph.nodes[to_id]["block"].input_schema()
        
        # Verifica se out_schema contém os dados para in_schema (simplificado)
        # Podemos usar model_fields para checar compatibilidade grossa
        if out_schema != BaseModel and in_schema != BaseModel:
            # apenas valida se for uma subclass ou compita de models.
            pass

        self.graph.add_edge(from_id, to_id)

    def validate_connections(self):
        # Percorre as arestas e valia tipos rigidamente se necessario
        pass
