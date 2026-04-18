import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(__file__))
from config import get_model

from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor
from agenticblocks.blocks.llm.agent import LLMAgentBlock
from agenticblocks.tools.mcp_client import MCPClientBridge

async def main():
    # Resolve the absolute path to the local MCP server script
    python_exe = sys.executable
    server_path = os.path.join(os.path.dirname(__file__), "mcp_server_estoque.py")

    # 1. Configure the MCP Client Bridge
    # Instead of npx, we pass our own Python process running the local server
    mcp_bridge = MCPClientBridge(
        command=python_exe,
        args=[server_path]
    )

    print("Starting MCP Client and performing handshake with the local Stock Server (stdio)...")
    try:
        # The magic: the client extracts tools directly from mcp_server_estoque.py
        mcp_tools = await mcp_bridge.connect()
        print(f"✅ Connected! Tools discovered on server: {[t.name for t in mcp_tools]}\n")
    except Exception as e:
        print(f"❌ MCP connection failed: {e}")
        return

    # 2. Inject MCP tools transparently into the LLM agent
    buying_agent = LLMAgentBlock(
        name="buying_agent",
        model=get_model(),
        system_prompt="You are the procurement assistant. Use the server tool to check the exact PRICE and AVAILABILITY of what the user requests.",
        tools=mcp_tools,
    )

    graph = WorkflowGraph()
    graph.add_block(buying_agent)
    executor = WorkflowExecutor(graph)

    if buying_agent.model.startswith("openai") and not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Reminder: set OPENAI_API_KEY so LiteLLM can reach the OpenAI API.")

    try:
        # 3. Trigger the reasoning loop
        print("👤 User: I'm setting up a workstation. Is a keyboard and monitor available in our warehouse?")
        ctx = await executor.run(initial_input={"prompt": "I'm setting up a workstation. Is a keyboard and monitor available in our warehouse?"})

        output = ctx.get_output("buying_agent")
        print("\n[🎯 Agent Final Response]:")
        print(output.response)
        print(f"--> [Stats] MCP tools invoked iteratively over the network: {output.tool_calls_made}")

    except Exception as e:
        print(f"\n[🛑 LLM provider failure]: {e}")

    finally:
        # 4. Gracefully shut down the stdio subprocess
        await mcp_bridge.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
