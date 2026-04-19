import json
import re
import uuid
from typing import Any
from agenticblocks.blocks.llm.agent import LLMAgentBlock

class HeuristicLLMAgentBlock(LLMAgentBlock):
    """
    A specialized LLMAgentBlock that parses plain-text responses for hallucinated 
    JSON tool calls. Useful when working with smaller or less-capable local models
    (like Granite4 or older Llama variants) that fail to use the native function 
    calling API properly.
    """
    description: str = "LLM Agent with heuristic fallback parsing for tool calls."

    def _parse_message(self, message: Any) -> Any:
        # If the model already returned a native tool call, do nothing.
        if message.tool_calls or not message.content:
            return message

        # Look for a JSON object in the text response
        match = re.search(r"\{.*\}", message.content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                # Check if it resembles the expected tool schema format
                if isinstance(data, dict) and "name" in data and ("parameters" in data or "arguments" in data):
                    tool_name = data["name"]
                    # Validate that it actually matches one of our available tools
                    if any(b.name == tool_name for b in self.tools):
                        args = data.get("parameters", data.get("arguments", {}))
                        args_str = json.dumps(args) if isinstance(args, dict) else str(args)

                        # Create mock classes to simulate litellm's internal object structure
                        class MockFunction:
                            def __init__(self, n, a):
                                self.name = n
                                self.arguments = a

                        class MockToolCall:
                            def __init__(self, tid, fn):
                                self.id = tid
                                self.type = "function"
                                self.function = fn

                        # Inject the mock tool call into the message
                        tc = MockToolCall(
                            f"call_{uuid.uuid4().hex[:10]}",
                            MockFunction(tool_name, args_str)
                        )
                        message.tool_calls = [tc]
                        # Clear the text content so it behaves exactly like a native tool call
                        message.content = ""  
            except json.JSONDecodeError:
                pass
                
        return message
