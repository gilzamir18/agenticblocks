import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import get_model

from pydantic import BaseModel
from agenticblocks.core.block import Block
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor
from agenticblocks.blocks.llm.agent import LLMAgentBlock

# 1. Researcher agent (sub-agent called via A2A)
class ResearcherInput(BaseModel):
    query: str

class ResearcherOutput(BaseModel):
    findings: str

class ResearcherBlock(Block[ResearcherInput, ResearcherOutput]):
    name: str = "researcher_agent"
    description: str = "Searches for information in a file or database for a given query."

    async def run(self, input: ResearcherInput) -> ResearcherOutput:
        print(f"\n[A2A Call] 🕵️ Researcher invoked searching for: '{input.query}'")
        await asyncio.sleep(0.5)  # Simulates heavy processing or an MCP call
        return ResearcherOutput(findings=f"Found data on '{input.query}': Absolute Success.")


# 2. Graph and Director agent setup
async def main():
    llm_model = get_model()

    graph = WorkflowGraph()

    # Instantiate the A2A subordinate block
    researcher = ResearcherBlock()

    # Instantiate the main Director agent (delegator)
    director_agent = LLMAgentBlock(
        name="director_agent",
        model=llm_model,
        system_prompt="You are the Director. Always use the researcher_agent tool to fetch exact data before responding.",
        tools=[researcher],  # A2A Bridge + automatic Pydantic schema
        max_iterations=5,    # Prevents infinite loops with local LLMs
    )

    graph.add_block(director_agent)
    executor = WorkflowExecutor(graph)

    if llm_model.startswith("gemini") and not os.getenv("GEMINI_API_KEY"):
        print("⚠️  Reminder: set GEMINI_API_KEY so LiteLLM can reach the Gemini API.")

    try:
        ctx = await executor.run(initial_input={"prompt": "Please investigate today's operation."})
        output = ctx.get_output("director_agent")

        print("\n[Director LLM — Final Response]:")
        print(output.response)
        print(f"- Transparent A2A tool calls made: {output.tool_calls_made}")
    except Exception as e:
        print(f"\n[Early termination (API error)]: {e}")

if __name__ == "__main__":
    asyncio.run(main())
