import unittest
from pydantic import BaseModel
from agenticblocks.core.block import Block
from agenticblocks.core.graph import WorkflowGraph, CycleGroup

# Simple blocks for testing graph connectivity
class SimpleInput(BaseModel):
    prompt: str

class SimpleOutput(BaseModel):
    response: str

class SimpleBlock(Block[SimpleInput, SimpleOutput]):
    async def run(self, input: SimpleInput) -> SimpleOutput:
        return SimpleOutput(response=f"Simple: {input.prompt}")


class TestWorkflowGraph(unittest.TestCase):

    def test_add_block_success(self):
        """Test adding individual blocks to the WorkflowGraph."""
        graph = WorkflowGraph()
        block_a = SimpleBlock(name="A")
        block_b = SimpleBlock(name="B")

        graph.add_block(block_a)
        graph.add_block(block_b)

        self.assertIn("A", graph.graph.nodes)
        self.assertIn("B", graph.graph.nodes)
        self.assertEqual(graph.graph.nodes["A"]["block"], block_a)

    def test_add_duplicate_block_raises(self):
        """Test that adding a duplicate block name raises ValueError."""
        graph = WorkflowGraph()
        block_a = SimpleBlock(name="A")
        graph.add_block(block_a)

        with self.assertRaises(ValueError):
            graph.add_block(SimpleBlock(name="A"))

    def test_add_sequence(self):
        """Test add_sequence registers and connects blocks in order."""
        graph = WorkflowGraph()
        block_a = SimpleBlock(name="A")
        block_b = SimpleBlock(name="B")
        block_c = SimpleBlock(name="C")

        graph.add_sequence(block_a, block_b, block_c)

        self.assertIn("A", graph.graph.nodes)
        self.assertIn("B", graph.graph.nodes)
        self.assertIn("C", graph.graph.nodes)

        self.assertTrue(graph.graph.has_edge("A", "B"))
        self.assertTrue(graph.graph.has_edge("B", "C"))

    def test_connect_blocks(self):
        """Test direct connection between two individual blocks."""
        graph = WorkflowGraph()
        block_a = SimpleBlock(name="A")
        block_b = SimpleBlock(name="B")

        graph.add_block(block_a)
        graph.add_block(block_b)
        graph.connect("A", "B")

        self.assertTrue(graph.graph.has_edge("A", "B"))

    def test_connect_invalid_blocks_raises(self):
        """Test that connecting non-existent blocks raises ValueError."""
        graph = WorkflowGraph()
        block_a = SimpleBlock(name="A")
        graph.add_block(block_a)

        with self.assertRaises(ValueError):
            graph.connect("A", "B")

    def test_add_cycle_and_autodetect_entry(self):
        """Test cycle registration and automatic entry-block detection."""
        graph = WorkflowGraph()
        block_a = SimpleBlock(name="A")
        block_b = SimpleBlock(name="B")
        block_c = SimpleBlock(name="C")

        graph.add_block(block_a)
        graph.add_block(block_b)
        graph.add_block(block_c)

        # Declare cycle A -> B -> C where C is the condition block. A has no incoming internal edges.
        cycle_name = graph.add_cycle(
            name="Cycle1",
            sequence=["A", "B", "C"],
            condition_block="C",
            max_iterations=10
        )

        self.assertEqual(cycle_name, "Cycle1")
        self.assertIn("Cycle1", graph._cycles)
        
        cycle = graph._cycles["Cycle1"]
        self.assertEqual(cycle.condition_block, "C")
        self.assertEqual(cycle.entry_block, "A")  # Auto-detected entry
        self.assertEqual(cycle.max_iterations, 10)

        # Verify parent cycle mappings
        self.assertEqual(graph._node_to_cycle["A"], "Cycle1")
        self.assertEqual(graph._node_to_cycle["B"], "Cycle1")
        self.assertEqual(graph._node_to_cycle["C"], "Cycle1")

    def test_collapsed_graph_generation(self):
        """Test that the collapsed graph represents cycles as single virtual nodes."""
        graph = WorkflowGraph()
        block_in = SimpleBlock(name="Input")
        block_a = SimpleBlock(name="A")
        block_b = SimpleBlock(name="B")
        block_out = SimpleBlock(name="Output")

        graph.add_block(block_in)
        graph.add_block(block_a)
        graph.add_block(block_b)
        graph.add_block(block_out)

        graph.add_cycle(
            name="CycleAB",
            sequence=["A", "B"],
            condition_block="B"
        )

        # Connect Input -> CycleAB -> Output
        graph.connect("Input", "CycleAB")
        graph.connect("CycleAB", "Output")

        collapsed = graph.collapsed_graph()

        # The collapsed DAG nodes should only include top-level nodes and virtual cycle node
        self.assertEqual(set(collapsed.nodes), {"Input", "CycleAB", "Output"})
        self.assertTrue(collapsed.has_edge("Input", "CycleAB"))
        self.assertTrue(collapsed.has_edge("CycleAB", "Output"))
        self.assertFalse(collapsed.has_edge("A", "B"))  # Internal edge is collapsed
