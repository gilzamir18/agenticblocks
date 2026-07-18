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

    @patch("agenticblocks.blocks.llm.memgpt_agent.litellm.acompletion")
    async def test_memgpt_on_iteration_sync(self, mock_acompletion):
        """Test MemGPT with a synchronous on_iteration callback."""
        tool_args = json.dumps({"message": "Hello from MemGPT!", "request_heartbeat": False})
        mock_tc = MockToolCall(call_id="call_123", name="send_message", arguments=tool_args)
        mock_acompletion.return_value = MockResponse(content=None, tool_calls=[mock_tc])

        iterations_called = []
        messages_called = []

        def sync_callback(iteration: int, messages: list):
            iterations_called.append(iteration)
            messages_called.append(list(messages))

        agent = MemGPTAgentBlock(
            name="MemGPTSyncCallback",
            model="ollama/gemma4:latest",
            use_shared_router=False,
            on_iteration=sync_callback,
            debug=False
        )

        output = await agent.run(AgentInput(prompt="Test sync callback"))

        self.assertEqual(output.response, "Hello from MemGPT!")
        self.assertEqual(iterations_called, [0, 1])
        self.assertEqual(len(messages_called), 2)
        self.assertEqual(messages_called[0][-1]["role"], "user")
        self.assertEqual(messages_called[0][-1]["content"], "Test sync callback")
        self.assertEqual(messages_called[1][-2]["role"], "assistant")
        self.assertEqual(messages_called[1][-1]["role"], "tool")

    @patch("agenticblocks.blocks.llm.memgpt_agent.litellm.acompletion")
    async def test_memgpt_on_iteration_async(self, mock_acompletion):
        """Test MemGPT with an asynchronous on_iteration callback."""
        tool_args = json.dumps({"message": "Async dynamic reply", "request_heartbeat": False})
        mock_tc = MockToolCall(call_id="call_123", name="send_message", arguments=tool_args)
        mock_acompletion.return_value = MockResponse(content=None, tool_calls=[mock_tc])

        iterations_called = []
        messages_called = []

        async def async_callback(iteration: int, messages: list):
            iterations_called.append(iteration)
            messages_called.append(list(messages))

        agent = MemGPTAgentBlock(
            name="MemGPTAsyncCallback",
            model="ollama/gemma4:latest",
            use_shared_router=False,
            on_iteration=async_callback,
            debug=False
        )

        output = await agent.run(AgentInput(prompt="Test async callback"))

        self.assertEqual(output.response, "Async dynamic reply")
        self.assertEqual(iterations_called, [0, 1])
        self.assertEqual(len(messages_called), 2)

    @patch("agenticblocks.blocks.llm.memgpt_agent.litellm.acompletion")
    async def test_memgpt_on_iteration_multiple_heartbeats(self, mock_acompletion):
        """Test MemGPT on_iteration callback with multiple heartbeats."""
        # Heartbeat 0: request_heartbeat=True
        tool_args_0 = json.dumps({"message": "Thinking...", "request_heartbeat": True})
        mock_tc_0 = MockToolCall(call_id="call_0", name="send_message", arguments=tool_args_0)
        
        # Heartbeat 1: request_heartbeat=False
        tool_args_1 = json.dumps({"message": "Done!", "request_heartbeat": False})
        mock_tc_1 = MockToolCall(call_id="call_1", name="send_message", arguments=tool_args_1)

        mock_acompletion.side_effect = [
            MockResponse(content=None, tool_calls=[mock_tc_0]),
            MockResponse(content=None, tool_calls=[mock_tc_1])
        ]

        iterations_called = []

        def callback(iteration: int, messages: list):
            iterations_called.append(iteration)

        agent = MemGPTAgentBlock(
            name="MemGPTMultiHeartbeat",
            model="ollama/gemma4:latest",
            use_shared_router=False,
            on_iteration=callback,
            debug=False
        )

        output = await agent.run(AgentInput(prompt="Two steps"))

        self.assertEqual(iterations_called, [0, 1, 2])
        self.assertEqual(output.tool_calls_made, 2)

    @patch("agenticblocks.blocks.llm.memgpt_agent.litellm.acompletion")
    async def test_empty_send_message_does_not_terminate_turn(self, mock_acompletion):
        """An empty send_message must be corrected instead of becoming an empty final response."""
        empty_args = json.dumps({"message": "", "request_heartbeat": False})
        empty_tc = MockToolCall(call_id="call_empty", name="send_message", arguments=empty_args)

        final_args = json.dumps({"message": "Corrected response.", "request_heartbeat": False})
        final_tc = MockToolCall(call_id="call_final", name="send_message", arguments=final_args)

        mock_acompletion.side_effect = [
            MockResponse(content=None, tool_calls=[empty_tc]),
            MockResponse(content=None, tool_calls=[final_tc]),
        ]

        agent = MemGPTAgentBlock(
            name="MemGPTEmptySendMessage",
            model="ollama/gemma4:latest",
            use_shared_router=False,
            debug=False,
        )

        output = await agent.run(AgentInput(prompt="Return a real answer"))

        self.assertEqual(output.response, "Corrected response.")
        self.assertEqual(output.tool_calls_made, 2)
        self.assertEqual(mock_acompletion.call_count, 2)
        self.assertTrue(
            any(
                item["role"] == "user"
                and "send_message with an empty message" in item["content"]
                for item in agent.internal_history
            )
        )


    @patch("agenticblocks.blocks.llm.memgpt_agent.litellm.acompletion")
    async def test_empty_response_retries_inside_heartbeat_loop(self, mock_acompletion):
        """An empty model response should consume a heartbeat and retry in the same run."""
        final_args = json.dumps({"message": "Recovered inside the heartbeat loop.", "request_heartbeat": False})
        final_tc = MockToolCall(call_id="call_final", name="send_message", arguments=final_args)

        mock_acompletion.side_effect = [
            MockResponse(content="", tool_calls=[]),
            MockResponse(content=None, tool_calls=[final_tc]),
        ]

        agent = MemGPTAgentBlock(
            name="MemGPTEmptyResponse",
            model="ollama/gemma4:latest",
            use_shared_router=False,
            debug=False,
        )

        output = await agent.run(AgentInput(prompt="Return a real answer"))

        self.assertEqual(output.response, "Recovered inside the heartbeat loop.")
        self.assertEqual(output.tool_calls_made, 1)
        self.assertEqual(mock_acompletion.call_count, 2)
        self.assertTrue(
            any(
                item["role"] == "user"
                and "You returned an empty response without calling a tool" in item["content"]
                for item in agent.internal_history
            )
        )

    @patch("agenticblocks.blocks.llm.memgpt_agent.litellm.acompletion")
    async def test_empty_response_does_not_retry_when_heartbeats_are_zero(self, mock_acompletion):
        """With a zero heartbeat budget, an empty model response still ends the run."""
        mock_acompletion.return_value = MockResponse(content="", tool_calls=[])

        agent = MemGPTAgentBlock(
            name="MemGPTZeroHeartbeats",
            model="ollama/gemma4:latest",
            use_shared_router=False,
            max_heartbeats=0,
            debug=False,
        )

        output = await agent.run(AgentInput(prompt="Return a real answer"))

        self.assertEqual(output.response, "")
        self.assertEqual(mock_acompletion.call_count, 1)


from agenticblocks.core.function_block import as_tool


@as_tool(name="run_skill", description="Run a skill.")
def _run_skill(skill_name: str, context: str, intent: str = "newfeat") -> str:
    return "ok"


class TestMemGPTTextToolCallRecovery(unittest.TestCase):
    """Recover tool calls a model emits as plain-text JSON (no native API).

    Small local models (e.g. gemma4) stop using the function-calling API after a
    few turns and emit the call as text JSON in several shapes. The MemGPT block
    must recover these via the shared parser and validate names against its tools.
    """

    def _agent(self):
        return MemGPTAgentBlock(
            name="rec", model="ollama/gemma4:latest",
            tools=[_run_skill], use_shared_router=False, debug=False,
        )

    def _agent_tools(self, agent):
        # Mirror run(): the live tool set includes send_message.
        @as_tool(name="send_message", description="Talk to the user.")
        def send_message(message: str, request_heartbeat: bool = False) -> str:
            return "ok"
        return list(agent.tools) + [send_message]

    def test_recovers_nested_openai_tool_calls_shape(self):
        """The exact shape gemma4 emits: {"tool_calls":[{"function":{name,arguments}}]}."""
        agent = self._agent()
        content = json.dumps({
            "tool_calls": [{
                "id": "call_x",
                "function": {
                    "name": "run_skill",
                    "arguments": {
                        "skill_name": "implement-feature",
                        "context": "Add subtract(a,b) to calc.py",
                        "intent": "newfeat",
                    },
                },
            }],
        })
        tc = agent._recover_tool_call_from_text(content, self._agent_tools(agent))
        self.assertIsNotNone(tc)
        self.assertEqual(tc.function.name, "run_skill")
        args = json.loads(tc.function.arguments)
        self.assertEqual(args["skill_name"], "implement-feature")
        self.assertEqual(args["intent"], "newfeat")

    def test_recovers_flat_shape(self):
        agent = self._agent()
        content = json.dumps({
            "function": "send_message",
            "arguments": {"message": "done"},
        })
        tc = agent._recover_tool_call_from_text(content, self._agent_tools(agent))
        self.assertIsNotNone(tc)
        self.assertEqual(tc.function.name, "send_message")

    def test_unknown_tool_name_is_rejected(self):
        """Names not registered on the block must never be invented."""
        agent = self._agent()
        content = json.dumps({
            "tool_calls": [{"function": {"name": "delete_everything", "arguments": {}}}],
        })
        tc = agent._recover_tool_call_from_text(content, self._agent_tools(agent))
        self.assertIsNone(tc)

    def test_plain_text_is_not_a_tool_call(self):
        agent = self._agent()
        self.assertIsNone(agent._recover_tool_call_from_text("Just a sentence.", self._agent_tools(agent)))
        self.assertIsNone(agent._recover_tool_call_from_text(None, self._agent_tools(agent)))

    def test_markdown_fenced_json_is_recovered(self):
        agent = self._agent()
        content = "```json\n" + json.dumps({
            "tool_calls": [{"function": {"name": "run_skill",
                                          "arguments": {"skill_name": "x", "context": "y"}}}],
        }) + "\n```"
        tc = agent._recover_tool_call_from_text(content, self._agent_tools(agent))
        self.assertIsNotNone(tc)
        self.assertEqual(tc.function.name, "run_skill")


if __name__ == "__main__":
    unittest.main()
