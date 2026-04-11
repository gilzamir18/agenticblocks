import networkx as nx
from dataclasses import dataclass, field
from typing import Optional
from .block import Block
from pydantic import BaseModel


@dataclass
class CycleGroup:
    """
    Descreve um subgrafo cíclico declarado explicitamente.

    O executor roda os blocos do ciclo de forma iterativa até que
    `condition_block` retorne `is_valid=True` ou `max_iterations` seja atingido.

    Atributos:
        name:             Identificador único do ciclo (usado como nó virtual no grafo).
        members:          Nomes dos blocos que pertencem ao ciclo.
        edges:            Arestas internas do ciclo [(from, to), ...].
        condition_block:  Bloco cujo output controla a continuação (should have is_valid/feedback).
        entry_block:      Bloco que recebe o prompt aumentado com o feedback a cada re-tentativa.
        max_iterations:   Máximo de iterações antes de desistir.
        prompt_field:     Campo do input do entry_block a ser aumentado com o feedback.
    """
    name: str
    members: list[str]
    edges: list[tuple[str, str]]
    condition_block: str
    entry_block: str
    max_iterations: int = 5
    prompt_field: str = "prompt"


class WorkflowGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._cycles: dict[str, CycleGroup] = {}
        # Reverse map: block_name → cycle_name
        self._node_to_cycle: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Block registration
    # ------------------------------------------------------------------

    def add_block(self, block: Block) -> str:
        node_id = block.name
        if node_id in self.graph.nodes:
            raise ValueError(f"Já existe um bloco com o nome '{node_id}'")
        self.graph.add_node(node_id, block=block)
        return node_id

    # ------------------------------------------------------------------
    # Cycle declaration
    # ------------------------------------------------------------------

    def add_cycle(
        self,
        name: str,
        condition_block: str,
        edges: list[tuple[str, str]] | None = None,
        sequence: list[str] | None = None,
        max_iterations: int = 5,
        prompt_field: str = "prompt",
    ) -> str:
        """
        Declara um ciclo limitado no grafo.

        Todos os blocos referenciados já devem ter sido adicionados
        via add_block(). O entry_block é detectado automaticamente como o nó
        que não possui predecessores dentro do ciclo.

        Exatamente um dos parâmetros `edges` ou `sequence` deve ser fornecido:

        - ``edges``: lista explícita de arestas ``(from, to)`` para topologias
          arbitrárias (fanout, merge, etc.).
        - ``sequence``: atalho para cadeias lineares. Passando
          ``sequence=["A", "B", "C", "D"]`` equivale a
          ``edges=[("A","B"), ("B","C"), ("C","D")]``.

        Retorna o nome do ciclo, que pode ser usado em connect() como nó virtual.
        """
        if name in self._cycles:
            raise ValueError(f"Ciclo '{name}' já existe.")

        # Resolve edges from sequence shorthand or validate explicit edges
        if edges is not None and sequence is not None:
            raise ValueError(
                "Forneça apenas 'edges' ou 'sequence', não os dois ao mesmo tempo."
            )
        if sequence is not None:
            if len(sequence) < 2:
                raise ValueError(
                    "'sequence' precisa ter pelo menos 2 blocos para formar uma aresta."
                )
            edges = list(zip(sequence, sequence[1:]))
        if not edges:
            raise ValueError(
                "Forneça 'edges' ou 'sequence' com pelo menos uma aresta."
            )

        # Collect member block names from edges
        members = list({n for edge in edges for n in edge})
        for m in members:
            if m not in self.graph.nodes:
                raise ValueError(
                    f"Bloco '{m}' não encontrado. Chame add_block() antes de add_cycle()."
                )
            if m in self._node_to_cycle:
                raise ValueError(
                    f"Bloco '{m}' já pertence ao ciclo '{self._node_to_cycle[m]}'."
                )

        if condition_block not in members:
            raise ValueError(
                f"condition_block '{condition_block}' deve fazer parte das arestas do ciclo."
            )

        # Auto-detect entry_block: the member with no incoming edge within the cycle
        targets = {to for _, to in edges}
        sources = {frm for frm, _ in edges}
        entries = sources - targets
        if not entries:
            raise ValueError(
                "Não foi possível determinar o entry_block automaticamente: "
                "todos os membros possuem predecessores internos."
            )
        entry_block = entries.pop()

        # Register all internal edges on the graph (these form the cycle)
        for frm, to in edges:
            self.graph.add_edge(frm, to, _cycle=name)

        cycle = CycleGroup(
            name=name,
            members=members,
            edges=edges,
            condition_block=condition_block,
            entry_block=entry_block,
            max_iterations=max_iterations,
            prompt_field=prompt_field,
        )
        self._cycles[name] = cycle
        for m in members:
            self._node_to_cycle[m] = name

        return name

    # ------------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------------

    def connect(self, from_id: str, to_id: str) -> None:
        """
        Conecta dois nós ou um nó e um ciclo declarado.
        Aceita nomes de ciclos como from_id ou to_id (nós virtuais).
        """
        from_is_cycle = from_id in self._cycles
        to_is_cycle   = to_id in self._cycles

        if from_is_cycle or to_is_cycle:
            if not from_is_cycle and from_id not in self.graph.nodes:
                raise ValueError(f"Bloco '{from_id}' não encontrado no grafo.")
            if not to_is_cycle and to_id not in self.graph.nodes:
                raise ValueError(f"Bloco '{to_id}' não encontrado no grafo.")

            # Resolve the actual node to connect in the real graph:
            # cycle output flows from condition_block; cycle input enters at entry_block.
            actual_from = self._cycles[from_id].condition_block if from_is_cycle else from_id
            actual_to   = self._cycles[to_id].entry_block       if to_is_cycle   else to_id
            # Mark as cross-cycle edge so collapsed_graph() can handle it
            self.graph.add_edge(actual_from, actual_to, _cross_cycle=True)
        else:
            if from_id not in self.graph.nodes or to_id not in self.graph.nodes:
                raise ValueError("Ambos os blocos devem existir no grafo.")
            self.graph.add_edge(from_id, to_id)

    # ------------------------------------------------------------------
    # Collapsed (DAG) view for validation and wave-building
    # ------------------------------------------------------------------

    def collapsed_graph(self) -> nx.DiGraph:
        """
        Retorna uma visão DAG do grafo onde cada CycleGroup é colapsado
        em um único nó virtual. Usado por:
          - _validate()     (checar que não há ciclos não declarados)
          - _build_waves()  (ordenação topológica)
        """
        g = nx.DiGraph()

        # Add non-cycle-member nodes
        for node in self.graph.nodes:
            if node not in self._node_to_cycle:
                g.add_node(node)

        # Add cycle virtual nodes
        for cycle_name in self._cycles:
            g.add_node(cycle_name)

        # Collapse edges: cycle-internal edges are skipped;
        # all other edges are remapped using the cycle name as endpoint.
        for frm, to, data in self.graph.edges(data=True):
            if data.get("_cycle"):
                # Internal cycle edge — skip
                continue

            v_from = self._node_to_cycle.get(frm, frm)
            v_to   = self._node_to_cycle.get(to,  to)

            if v_from == v_to:
                # Both endpoints collapsed to the same cycle — skip self-loop
                continue

            g.add_edge(v_from, v_to)

        return g

    def validate_connections(self) -> None:
        # Future: strict type checking between connected schemas
        pass
