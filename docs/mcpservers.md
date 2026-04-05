# Connecting to External MCP Servers

One of the most powerful features of **AgenticBlocks.IO** is its seamless interoperability with the **Model Context Protocol (MCP)**. Because our `MCPClientBridge` is designed to communicate via Standard Input/Output (Stdio), you are not restricted to writing Python servers. 

You can effortlessly connect your `LLMAgentBlock` to external MCP servers written in NodeJS, Go, Rust, or any other language, leveraging the global ecosystem of pre-built tools.

## How it works

The `MCPClientBridge` class only requires a `command` and its `args` to spin up a background subprocess. It handles the JSON-RPC handshake, extracts the schemas of all available tools, translates them to Pydantic models automatically, and makes them ready for your LLM.

### Example 1: Connecting to a NodeJS Server (Anthropic's Brave Search)

Anthropic and the community provide DOZENS of production-ready public servers via NPM (Node Package Manager). If you have Node.js and `npx` installed, you can consume these tools instantly.

```python
import asyncio
from agenticblocks.tools.mcp_client import MCPClientBridge
from agenticblocks.blocks.llm.agent import LLMAgentBlock

async def main():
    # 1. Provide the shell command (npx) and the package name to download/run
    mcp_bridge = MCPClientBridge(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-brave-search"]
    )

    # 2. Extract tools from the external server
    mcp_tools = await mcp_bridge.connect()
    
    # 3. Inject them directly into your Agent!
    agent = LLMAgentBlock(
        name="web_searcher",
        model="gemini/gemini-1.5-flash",
        system_prompt="You are a helpful assistant. Use the tools to search the web.",
        tools=mcp_tools
    )
    
    print(f"Loaded tools: {[t.name for t in mcp_tools]}")
    # ... Add to your WorkflowGraph and execute!
    
    await mcp_bridge.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
```

*(Note: Certain servers like Brave Search require environment variables for their own API keys, which you can specify in your host OS or `os.environ` before running).*

### Example 2: Connecting to an Executable or Binary (Go/Rust)

If you have downloaded a compiled `.exe` or a Linux binary of an MCP server built by a third-party developer, you simply point the bridge to the absolute path of the executable file.

```python
mcp_bridge = MCPClientBridge(
    command="C:\\Absolute\\Path\\To\\mcp_database_server.exe",
    args=["--readonly"] # Add any CLI arguments the binary might require
)

mcp_tools = await mcp_bridge.connect()
```

### Example 3: Remote Servers via HTTP / SSE

*Currently, AgenticBlocks.IO primarily implements the **Stdio** bridge transport. HTTP SSE support for purely remote web-hosted MCP servers will be mapped through a future `MCPSSEBridge` adapter.*

## Why does this matter?

By using external MCP servers, your Python codebase remains extremely lightweight. You can use a robust enterprise database connector written in Go or a headless browser automation tool written in Node, without ever modifying your agent logic or maintaining the external code. 

AgenticBlocks.IO handles the burden of wrapping these disjoint architectures into structured, typing-safe (Pydantic) Python loops!
