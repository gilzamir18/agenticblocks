# Tutorial: Integrating MCP Servers in AgenticBlocks

In this tutorial, we'll explore how to expand the capabilities of your agents in `agenticblocks` using the **Model Context Protocol (MCP)**. MCP allows you to connect language models to external tools and data sources in a standardized way.

In this practical example, we'll build a **Research Agent** that uses two different MCP servers running locally via `uvx`:
1.  **`duckduckgo-mcp-server`**: To perform web searches.
2.  **`mcp-server-fetch`**: To extract content from the URLs found.

---

## Prerequisites

Make sure you have `agenticblocks` installed, as well as `uv` (a Python package manager, used here via the `uvx` command to run MCP servers dynamically). You'll also need a model running locally in Ollama (in this example, `mistral-nemo`).

## Step 1: Imports and Input Tool

First, we import the necessary components from `agenticblocks` and the `anyio` async library. We also create a custom tool to capture the research topic from the user.

```python
import anyio
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor
from agenticblocks.blocks.llm.agent import LLMAgentBlock, AgentInput
from agenticblocks import as_tool
from agenticblocks.tools.mcp_client import MCPClientBridge

@as_tool(name="get_user_input")
async def get_user_input(prompt: str) -> AgentInput:
    print("What do you want to research: ", end="")
    user_input = input()
    return AgentInput(prompt=f"Research the topic {user_input}")
```

## Step 2: Connecting to MCP Servers

The `MCPClientBridge` class is the bridge between your code and the MCP server. We start the servers as subprocesses by passing the command and its arguments.

By calling `await mcp.connect()`, the client automatically extracts all tools exposed by the server and returns a list of proxy blocks (`MCPProxyBlock`) ready to use.

```python
async def main():
    # 1. Connect to the MCP server that fetches page content (fetch)
    mcp_fetch = MCPClientBridge(command="uvx", args=["mcp-server-fetch"])
    mcp_tools_fetch = await mcp_fetch.connect()  # Extract fetch tools
    
    # 2. Connect to the DuckDuckGo search MCP server
    mcp_search = MCPClientBridge(command="uvx", args=["duckduckgo-mcp-server"])
    mcp_tools_search = await mcp_search.connect()  # Extract search tools
```

## Step 3: Configuring the Research Agent

Now we create our LLM agent. The most important part here is the `tools` property. We simply add together the tool lists returned by our two MCP servers (`mcp_tools_fetch + mcp_tools_search`). The agent now has the power to search DuckDuckGo and read web pages!

```python
    graph = WorkflowGraph()

    agent_block = LLMAgentBlock(
        name="research_agent",
        model="ollama/mistral-nemo:latest",
        description="Research agent",
        system_prompt="""You are a research assistant. When given a topic, use the search and fetch tools to extract information from URLs. Write a final report in journalistic prose style. Strict rule: deliver plain prose only, with absolutely no formatting, lists, or markdown markup.""",
        # Injecting MCP tools into the Agent:
        tools=mcp_tools_fetch + mcp_tools_search,
        max_iterations=10,
        on_max_iterations="return_last",
        litellm_kwargs={"temperature": 0.7, "tool_choice": "auto", "num_ctx": 32000}
    )
```

## Step 4: Building the Graph and Running

With the agent ready, we define the execution flow. The graph will start by asking for user input (`get_user_input`) and pass the result directly to the research agent (`agent_block`).

```python
    # Define the sequence: User Input -> Agent
    graph.add_sequence(get_user_input, agent_block)

    # Run the workflow
    executor = WorkflowExecutor(graph)
    ctx = await executor.run(initial_input={"prompt": ""})
    
    # Extract the agent's final response
    cr = ctx.get_output("research_agent")
    print("\nFinal Report:\n")
    print(cr.response)
```

## Step 5: Cleanup and Disconnection

Since the MCP servers are running as subprocesses, it's crucial to ensure they are properly shut down at the end of execution to avoid zombie processes on your system.

```python
    # Attempt to disconnect and shut down the Fetch server
    try:
        await mcp_fetch.disconnect()
    except RuntimeError:
        pass

    # Attempt to disconnect and shut down the Search server
    try:
        await mcp_search.disconnect()
    except Exception:
        pass

if __name__ == "__main__":
    # Use anyio.run to execute the main async loop
    anyio.run(main)
```

---

## Complete Code

For convenience, here is the complete code for you to copy and run:

```python
import anyio
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor
from agenticblocks.blocks.llm.agent import LLMAgentBlock, AgentInput
from agenticblocks import as_tool
from agenticblocks.tools.mcp_client import MCPClientBridge

@as_tool(name="get_user_input")
async def get_user_input(prompt: str) -> AgentInput:
    print("What do you want to research: ", end="")
    user_input = input()
    return AgentInput(prompt=f"Research the topic {user_input}")

async def main():
    # Connect to MCP servers
    mcp_fetch = MCPClientBridge(command="uvx", args=["mcp-server-fetch"])
    mcp_tools_fetch = await mcp_fetch.connect()
    
    mcp_search = MCPClientBridge(command="uvx", args=["duckduckgo-mcp-server"])
    mcp_tools_search = await mcp_search.connect()
    
    graph = WorkflowGraph()

    agent_block = LLMAgentBlock(
        name="research_agent",
        model="ollama/mistral-nemo:latest",
        description="Research agent",
        system_prompt="""You are a research assistant. When given a topic, use the search and fetch tools to extract information from URLs. Write a final report in journalistic prose style. Strict rule: deliver plain prose only, with absolutely no formatting, lists, or markdown markup.""",
        tools=mcp_tools_fetch + mcp_tools_search,
        max_iterations=10,
        on_max_iterations="return_last",
        litellm_kwargs={"temperature": 0.7, "tool_choice": "auto", "num_ctx": 32000}
    )

    graph.add_sequence(get_user_input, agent_block)

    executor = WorkflowExecutor(graph)
    ctx = await executor.run(initial_input={"prompt": ""})
    cr = ctx.get_output("research_agent")
    print(cr.response)

    # Cleanup
    try:
        await mcp_fetch.disconnect()
    except RuntimeError:
        pass

    try:
        await mcp_search.disconnect()
    except Exception:
        pass

if __name__ == "__main__":
    anyio.run(main)
```

### Summary

In this tutorial you learned how to:
1. Instantiate `MCPClientBridge` by passing CLI commands to start MCP servers.
2. Use `await bridge.connect()` to expose external tools as native AgenticBlocks blocks.
3. Inject multiple MCP tool lists directly into `LLMAgentBlock`.
4. Manage the connection lifecycle using `disconnect()` safely.