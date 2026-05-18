import json
import unittest
from unittest.mock import AsyncMock, patch
from pydantic import BaseModel, Field
from typing import Optional, List, Any

from agenticblocks.blocks.llm.agent import LLMAgentBlock, AgentInput, AgentOutput
from agenticblocks.core.block import Block

# Define some dummy schemas for testing
class UserSchema(BaseModel):
    name: str = Field(description="The user's name")
    age: int = Field(description="The user's age")
    hobbies: List[str] = Field(default_factory=list, description="List of hobbies")

class DummyToolInput(BaseModel):
    query: str

class DummyToolOutput(BaseModel):
    result: str

class DummyToolBlock(Block[DummyToolInput, DummyToolOutput]):
    name: str = "DummyTool"
    description: str = "A dummy tool for unit tests"

    async def run(self, input: DummyToolInput) -> DummyToolOutput:
        return DummyToolOutput(result=f"Processed: {input.query}")


# Mock classes to mimic LiteLLM's response structure
class MockUsage:
    def __init__(self):
        self.prompt_tokens = 10
        self.completion_tokens = 15
        self.total_tokens = 25

class MockFunctionCall:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments

class MockToolCall:
    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.type = "function"
        self.function = MockFunctionCall(name, arguments)

class MockMessage:
    def __init__(self, content: Optional[str], tool_calls: Optional[List[MockToolCall]] = None):
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls or []

class MockChoice:
    def __init__(self, content: Optional[str], tool_calls: Optional[List[MockToolCall]] = None):
        self.message = MockMessage(content, tool_calls)

class MockResponse:
    def __init__(self, content: Optional[str], tool_calls: Optional[List[MockToolCall]] = None):
        self.choices = [MockChoice(content, tool_calls)]
        self.usage = MockUsage()


class TestLLMAgentBlock(unittest.IsolatedAsyncioTestCase):

    @patch("agenticblocks.blocks.llm.agent.litellm.acompletion")
    async def test_basic_unstructured_run(self, mock_acompletion):
        """Test LLMAgentBlock basic execution returning unstructured plain text."""
        # Setup mock response
        mock_acompletion.return_value = MockResponse(content="Hello world!")
        
        agent = LLMAgentBlock(name="BasicAgent", model="ollama/gemma4:latest", debug=False)
        output = await agent.run(AgentInput(prompt="Say hello"))
        
        self.assertEqual(output.response, "Hello world!")
        self.assertIsNone(output.structured_output)
        self.assertEqual(output.tool_calls_made, 0)
        
        # Verify litellm parameters
        mock_acompletion.assert_called_once()
        called_kwargs = mock_acompletion.call_args[1]
        self.assertEqual(called_kwargs["model"], "ollama/gemma4:latest")
        # Validate target user input message at index 1 (index 0 is system message)
        self.assertEqual(called_kwargs["messages"][1]["content"], "Say hello")

    @patch("agenticblocks.blocks.llm.agent.litellm.acompletion")
    async def test_structured_output_direct(self, mock_acompletion):
        """Test LLMAgentBlock returning valid structured output directly in JSON."""
        user_json = json.dumps({"name": "Alice", "age": 30, "hobbies": ["reading", "hiking"]})
        mock_acompletion.return_value = MockResponse(content=user_json)
        
        agent = LLMAgentBlock(
            name="StructuredAgent",
            model="ollama/gemma4:latest",
            response_schema=UserSchema
        )
        output = await agent.run(AgentInput(prompt="Generate user Alice"))
        
        self.assertIsNotNone(output.structured_output)
        self.assertIsInstance(output.structured_output, UserSchema)
        self.assertEqual(output.structured_output.name, "Alice")
        self.assertEqual(output.structured_output.age, 30)
        self.assertEqual(output.structured_output.hobbies, ["reading", "hiking"])
        
        # Verify litellm request included response_format
        called_kwargs = mock_acompletion.call_args[1]
        self.assertEqual(called_kwargs["response_format"], UserSchema)

    @patch("agenticblocks.blocks.llm.agent.litellm.acompletion")
    async def test_structured_output_wrapped_in_markdown(self, mock_acompletion):
        """Test LLMAgentBlock robust parsing when JSON is wrapped in markdown code blocks."""
        wrapped_content = "```json\n" + json.dumps({"name": "Bob", "age": 25, "hobbies": ["gaming"]}) + "\n```"
        mock_acompletion.return_value = MockResponse(content=wrapped_content)
        
        agent = LLMAgentBlock(
            name="MarkdownAgent",
            model="ollama/gemma4:latest",
            response_schema=UserSchema
        )
        output = await agent.run(AgentInput(prompt="Generate user Bob"))
        
        self.assertIsNotNone(output.structured_output)
        self.assertEqual(output.structured_output.name, "Bob")
        self.assertEqual(output.structured_output.age, 25)
        self.assertEqual(output.structured_output.hobbies, ["gaming"])

    @patch("agenticblocks.blocks.llm.agent.litellm.acompletion")
    async def test_structured_output_fallback_formatting(self, mock_acompletion):
        """Test LLMAgentBlock fallback formatting flow.
        
        When the first completion returns conversational plain text (violating the schema),
        the block should automatically trigger a fallback synthesis/formatting call to get valid JSON.
        """
        user_data = {"name": "Charlie", "age": 40, "hobbies": ["cooking"]}
        # First call returns conversational text (invalid JSON)
        # Second call (fallback) returns valid JSON
        mock_acompletion.side_effect = [
            MockResponse(content="Sure, here is Charlie! He is 40 years old and likes cooking."),
            MockResponse(content=json.dumps(user_data))
        ]
        
        agent = LLMAgentBlock(
            name="FallbackAgent",
            model="ollama/gemma4:latest",
            response_schema=UserSchema,
            debug=True
        )
        output = await agent.run(AgentInput(prompt="Generate user Charlie"))
        
        # Ensure we made both completion calls
        self.assertEqual(mock_acompletion.call_count, 2)
        self.assertIsNotNone(output.structured_output)
        self.assertEqual(output.structured_output.name, "Charlie")
        self.assertEqual(output.structured_output.age, 40)
        self.assertEqual(output.structured_output.hobbies, ["cooking"])

    @patch("agenticblocks.blocks.llm.agent.litellm.acompletion")
    async def test_tool_calling_loop_execution(self, mock_acompletion):
        """Test tool execution loop of LLMAgentBlock with a native connected block."""
        tool = DummyToolBlock()
        agent = LLMAgentBlock(
            name="ToolAgent",
            model="ollama/gemma4:latest",
            tools=[tool],
            max_tool_calls=1
        )
        
        # Iteration 1: LLM returns a tool call
        # Iteration 2: LLM receives tool response and produces final answer
        tool_arguments = json.dumps({"query": "Hello Block"})
        mock_tool_call = MockToolCall(call_id="call_abc123", name="DummyTool", arguments=tool_arguments)
        
        mock_acompletion.side_effect = [
            MockResponse(content=None, tool_calls=[mock_tool_call]),
            MockResponse(content="I executed DummyTool, output was: Processed: Hello Block")
        ]
        
        output = await agent.run(AgentInput(prompt="Run DummyTool with Hello Block"))
        
        self.assertEqual(output.tool_calls_made, 1)
        self.assertEqual(output.response, "I executed DummyTool, output was: Processed: Hello Block")
        self.assertEqual(mock_acompletion.call_count, 2)
