import json
import unittest
from unittest.mock import AsyncMock, patch
from pydantic import BaseModel, Field
from typing import Optional, List

from agenticblocks.blocks.llm.memgpt_agent import MemGPTAgentBlock
from agenticblocks.blocks.llm.agent import AgentInput

class UserSchema(BaseModel):
    name: str = Field(description="The user's name")
    age: int = Field(description="The user's age")

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

class TestMemGPTAgentBlock(unittest.IsolatedAsyncioTestCase):

    @patch("agenticblocks.blocks.llm.memgpt_agent.litellm.acompletion")
    async def test_memgpt_basic_unstructured_run(self, mock_acompletion):
        """Test MemGPT basic execution completing via send_message."""
        # MemGPT strictly uses send_message to talk to the user.
        # It terminates when request_heartbeat=False.
        tool_args = json.dumps({"message": "Hello from MemGPT!", "request_heartbeat": False})
        mock_tc = MockToolCall(call_id="call_123", name="send_message", arguments=tool_args)
        mock_acompletion.return_value = MockResponse(content=None, tool_calls=[mock_tc])
        
        agent = MemGPTAgentBlock(
            name="MemGPTBasic", 
            model="ollama/gemma4:latest", 
            use_shared_router=False,
            debug=False
        )
        
        output = await agent.run(AgentInput(prompt="Say hello"))
        
        self.assertEqual(output.response, "Hello from MemGPT!")
        self.assertIsNone(output.structured_output)
        self.assertEqual(output.tool_calls_made, 1)

    @patch("agenticblocks.blocks.llm.memgpt_agent.litellm.acompletion")
    async def test_memgpt_structured_output_direct(self, mock_acompletion):
        """Test MemGPT parsing structured JSON correctly from send_message."""
        user_json = json.dumps({"name": "Alice", "age": 30})
        tool_args = json.dumps({"message": user_json, "request_heartbeat": False})
        mock_tc = MockToolCall(call_id="call_456", name="send_message", arguments=tool_args)
        
        mock_acompletion.return_value = MockResponse(content=None, tool_calls=[mock_tc])
        
        agent = MemGPTAgentBlock(
            name="MemGPTStructured",
            model="ollama/gemma4:latest",
            use_shared_router=False,
            response_schema=UserSchema
        )
        
        output = await agent.run(AgentInput(prompt="Create user Alice"))
        
        self.assertIsNotNone(output.structured_output)
        self.assertIsInstance(output.structured_output, UserSchema)
        self.assertEqual(output.structured_output.name, "Alice")
        self.assertEqual(output.structured_output.age, 30)
        self.assertEqual(mock_acompletion.call_count, 1) # No fallback needed

    @patch("agenticblocks.blocks.llm.memgpt_agent.litellm.acompletion")
    async def test_memgpt_structured_output_fallback(self, mock_acompletion):
        """Test MemGPT fallback formatting when send_message contains unstructured conversational text."""
        # Turn 1: conversational text sent via send_message
        tool_args = json.dumps({"message": "Here is Bob, he is 25 years old.", "request_heartbeat": False})
        mock_tc = MockToolCall(call_id="call_789", name="send_message", arguments=tool_args)
        
        # Turn 2: Fallback schema formatting call (plain text response with JSON)
        fallback_json = json.dumps({"name": "Bob", "age": 25})
        
        mock_acompletion.side_effect = [
            MockResponse(content=None, tool_calls=[mock_tc]),
            MockResponse(content=fallback_json)
        ]
        
        agent = MemGPTAgentBlock(
            name="MemGPTFallback",
            model="ollama/gemma4:latest",
            use_shared_router=False,
            response_schema=UserSchema
        )
        
        output = await agent.run(AgentInput(prompt="Create user Bob"))
        
        self.assertIsNotNone(output.structured_output)
        self.assertEqual(output.structured_output.name, "Bob")
        self.assertEqual(output.structured_output.age, 25)
        # Should take 2 calls: one for the heartbeat iteration, one for the final schema formatting fallback
        self.assertEqual(mock_acompletion.call_count, 2)
