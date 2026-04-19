# HeuristicLLMAgentBlock 🧠

The `HeuristicLLMAgentBlock` is a specialized subclass of `LLMAgentBlock` designed to handle models that have weak or inconsistent support for the official Function Calling API (e.g., smaller local models like `Granite4` or older Llama versions).

## The Problem: JSON Hallucinations

Standard agents expect models to return tool calls through a specific API channel (`tool_calls`). However, many open-source or quantized models often fail to use this channel and instead "hallucinate" the tool call as a raw JSON string inside their normal text response, like this:

```text
To find the answer, I will search the web:
{"name": "web_search", "parameters": {"query": "current president of USA"}}
```

By default, an agent would treat this as a final text response and stop, failing to execute the search.

## The Solution: Heuristic Parsing

`HeuristicLLMAgentBlock` uses a "greedy" regex-based parser to intercept these cases. Before deciding that an agent is finished, it checks the text content for:
1. A valid JSON object.
2. A `name` field that matches one of the agent's registered tools.
3. A `parameters` or `arguments` field containing the tool inputs.

If found, the block **transparently converts** the text into a formal tool call, executes it, and feeds the results back to the model as if it had used the official API.

## Usage

Simply replace `LLMAgentBlock` with `HeuristicLLMAgentBlock` when working with models that struggle with native tools:

```python
from agenticblocks.blocks.llm.heuristic_agent import HeuristicLLMAgentBlock

research_agent = HeuristicLLMAgentBlock(
    name="research_agent",
    model="ollama/granite4:latest",
    tools=[web_search],
    max_iterations=3
)
```

## Architecture

This was implemented using a clean hook in the base class:

```python
# In base LLMAgentBlock
def _parse_message(self, message: Any) -> Any:
    return message

# In HeuristicLLMAgentBlock
def _parse_message(self, message: Any) -> Any:
    # Custom regex logic to catch JSON in text and 
    # transform it into a tool_call object...
    return message
```

This design keeps the core library simple while providing an optional, robust solution for edge cases.
