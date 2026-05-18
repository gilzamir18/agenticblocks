import unittest
import asyncio
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

from agenticblocks.core.block import Block
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor
from agenticblocks.runtime.state import NodeStatus

# Schemas for execution test
class TaskInput(BaseModel):
    prompt: str

class TaskOutput(BaseModel):
    response: str

class TaskBlock(Block[TaskInput, TaskOutput]):
    async def run(self, input: TaskInput) -> TaskOutput:
        return TaskOutput(response=f"Task processed: {input.prompt}")

# Schemas for Cycle test
class ProducerInput(BaseModel):
    prompt: str

class ProducerOutput(BaseModel):
    response: str

class ProducerBlock(Block[ProducerInput, ProducerOutput]):
    async def run(self, input: ProducerInput) -> ProducerOutput:
        # If validator feedback is present, we output a corrected answer
        if "feedback" in input.prompt:
            return ProducerOutput(response="Correct Answer")
        return ProducerOutput(response="Incorrect Answer")

class ValidatorInput(BaseModel):
    response: str

class ValidatorOutput(BaseModel):
    is_valid: bool
    feedback: str

class ValidatorBlock(Block[ValidatorInput, ValidatorOutput]):
    async def run(self, input: ValidatorInput) -> ValidatorOutput:
        if input.response == "Correct Answer":
            return ValidatorOutput(is_valid=True, feedback="Perfect!")
        return ValidatorOutput(is_valid=False, feedback="Please output 'Correct Answer'.")


class TestWorkflowExecutor(unittest.IsolatedAsyncioTestCase):

    async def test_linear_executor_flow(self):
        """Test linear execution of two connected blocks in a graph."""
        graph = WorkflowGraph()
        block_a = TaskBlock(name="A")
        block_b = TaskBlock(name="B")

        graph.add_sequence(block_a, block_b)

        executor = WorkflowExecutor(graph, verbose=False)
        ctx = await executor.run(initial_input={"prompt": "Start"})

        # Verify execution context
        self.assertIn("A", ctx.results)
        self.assertIn("B", ctx.results)
        self.assertEqual(ctx.results["A"].status, NodeStatus.DONE)
        self.assertEqual(ctx.results["B"].status, NodeStatus.DONE)

        # Output from block A: "Task processed: Start"
        # Input to block B collects output from predecessor A: remapped to "prompt"
        # Output from block B: "Task processed: Task processed: Start"
        self.assertEqual(ctx.results["A"].output.response, "Task processed: Start")
        self.assertEqual(ctx.results["B"].output.response, "Task processed: Task processed: Start")

    async def test_cyclic_executor_loop_success(self):
        """Test that a declared cycle iterates until the validator block succeeds."""
        graph = WorkflowGraph()
        producer = ProducerBlock(name="Producer")
        validator = ValidatorBlock(name="Validator")

        graph.add_block(producer)
        graph.add_block(validator)

        graph.add_cycle(
            name="RefinementCycle",
            sequence=["Producer", "Validator"],
            condition_block="Validator",
            max_iterations=3,
            prompt_field="prompt"
        )

        executor = WorkflowExecutor(graph, verbose=False)
        # Iteration 1: Producer outputs "Incorrect Answer" -> Validator returns is_valid=False
        # Iteration 2: Input prompt augmented with Validator feedback ("feedback" string in prompt)
        # -> Producer now outputs "Correct Answer" -> Validator returns is_valid=True and loop terminates successfully.
        ctx = await executor.run(initial_input={"prompt": "Solve this task"})

        cycle_result = ctx.cycle_results.get("RefinementCycle")
        self.assertIsNotNone(cycle_result)
        self.assertEqual(cycle_result.iterations, 2)
        self.assertTrue(cycle_result.validated)
        self.assertEqual(cycle_result.output.response, "Correct Answer")
