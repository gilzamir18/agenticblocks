import sys
sys.path.append("/home/gilzamir/projetos/agenticblocks/src")

from agenticblocks.blocks.llm.agent import LLMAgentBlock
import json

class MockMessage:
    def __init__(self, content):
        self.content = content
        self.tool_calls = None

agent = LLMAgentBlock()

# Test 1: Plain JSON
msg = MockMessage('{"tool_name": "read_file_smart", "arguments": {"path": "index.html"}}')
parsed = agent._parse_message(msg)
print("Test 1:", getattr(parsed, "tool_calls", None) is not None)
if parsed.tool_calls:
    print("  ", parsed.tool_calls[0].function.name, parsed.tool_calls[0].function.arguments)

# Test 2: JSON in markdown block
msg = MockMessage('Here is the tool call:\n```json\n{"tool_name": "read_file_smart", "arguments": {"path": "index.html"}}\n```')
parsed = agent._parse_message(msg)
print("Test 2:", getattr(parsed, "tool_calls", None) is not None)
if parsed.tool_calls:
    print("  ", parsed.tool_calls[0].function.name, parsed.tool_calls[0].function.arguments)

# Test 3: List of JSONs
msg = MockMessage('[\n{"tool_name": "read_file_smart", "arguments": {"path": "index.html"}}\n]')
parsed = agent._parse_message(msg)
print("Test 3:", getattr(parsed, "tool_calls", None) is not None)
if parsed.tool_calls:
    print("  ", parsed.tool_calls[0].function.name, parsed.tool_calls[0].function.arguments)
