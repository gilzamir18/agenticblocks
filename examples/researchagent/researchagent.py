import anyio
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import get_model, get_litellm_kwargs

from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor
from agenticblocks.blocks.llm.agent import LLMAgentBlock, AgentInput
from agenticblocks import as_tool
from agenticblocks.tools.mcp_client import MCPClientBridge

@as_tool(name="get_user_input")
async def get_user_input(prompt: str) -> AgentInput:
    print("What do you want to research: ", end="")
    user_input = input()
    return AgentInput(prompt=f"Research the topic: {user_input}")

async def main():
    # Connect to the MCP fetch server (starts as a subprocess)
    mcp_fetch = MCPClientBridge(command="uvx", args=["mcp-server-fetch"])
    mcp_tools_fetch = await mcp_fetch.connect()  # returns a list of MCPProxyBlocks

    # Connect to the MCP search server (starts as a subprocess)
    mcp_search = MCPClientBridge(command="uvx", args=["duckduckgo-mcp-server"])
    mcp_tools_search = await mcp_search.connect()  # returns a list of MCPProxyBlocks

    graph = WorkflowGraph()

    agent_block = LLMAgentBlock(
        name="research_agent",
        model=get_model(),
        description="Research agent",
        system_prompt="You are a research assistant. Given a topic, use the search and fetch tools to gather information from URLs. Write a final journalistic-style report. Strict rule: deliver only plain prose, with absolutely no formatting, lists, or markdown markup.",
        tools=mcp_tools_fetch + mcp_tools_search,  # <-- tools coming from MCP servers
        max_iterations=3,
        debug=True,
        on_max_iterations="return_last",
        litellm_kwargs=get_litellm_kwargs()
    )

    graph.add_sequence(get_user_input, agent_block)

    executor = WorkflowExecutor(graph)
    ctx = await executor.run(initial_input={"prompt": ""})
    cr = ctx.get_output("research_agent")


    print(cr.response)

    try:
        await mcp_fetch.disconnect()
    except RuntimeError:
        pass

    try:
        await mcp_search.disconnect()
    except:
        pass

if __name__ == "__main__":
    anyio.run(main)