import anyio
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import get_model, get_litellm_kwargs

from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor
from agenticblocks.blocks.llm.agent import LLMAgentBlock, AgentInput
from agenticblocks.blocks.flow.prompt_builder import PromptBuilderBlock
from agenticblocks import as_tool
from ddgs import DDGS

@as_tool(name="get_user_input")
async def get_user_input(prompt: str) -> AgentInput:
    print("What do you want to research: ", end="")
    user_input = input()
    return AgentInput(prompt=f"Research the topic: {user_input}")

@as_tool(name="web_search")
async def web_search(query: str):
    """Realiza uma busca no DuckDuckGo e retorna os resultados."""
    with DDGS() as ddgs:
        res = [r for r in ddgs.text(query, max_results=5)]
        to_text = '\n'.join([r['body'] for r in res])
        return to_text

async def main():
    graph = WorkflowGraph()

    prompt_specialist = LLMAgentBlock(
        name="prompt_specialist",
        model=get_model(),
        description="Prompt specialist",
        system_prompt="You are a prompt specialist. You will receive a topic and you need to create the best possible prompt for a research agent. Return the best possible prompt for a research agent.",
        litellm_kwargs={"temperature": 0.3, "num_ctx": 4096, "max_tokens":100},
    )

    research_agent = LLMAgentBlock(
        name="research_agent",
        model=get_model(),
        description="Research agent",
        system_prompt="You are a research assistant. Given a topic, use the web_search tool to gather information about the topic. Write a final journalistic-style report. Strict rule: deliver only plain prose, with absolutely no formatting, lists, or markdown markup.",
        tools=[web_search],
        max_iterations=3,
        debug=True,
        on_max_iterations="return_last",
        litellm_kwargs=get_litellm_kwargs() | {"max_tokens":1500}
    )

    # Combines the original research topic ({prompt}) with the report ({response}).
    # The executor merges outputs from both predecessors (get_user_input and
    # research_agent) before passing them to this block, so both fields are
    # available to the template.
    answer_prompt_builder = PromptBuilderBlock(
        name="answer_prompt_builder",
        template="Research topic: {prompt}\n\nResearch report:\n{response}",
    )

    answer_specialist = LLMAgentBlock(
        name="answer_specialist",
        model=get_model(),
        description="Answer specialist",
        system_prompt="You are an answer specialist. You will receive a research topic and a detailed report. Produce a clear, direct answer to the topic in fluent prose. Answer in portuguese brazil.",
        litellm_kwargs={"temperature": 0.1, "num_ctx": 8192, "max_tokens":100},
    )

    graph.add_sequence(
        get_user_input,
        research_agent,
        answer_prompt_builder,
        answer_specialist,
    )
    # Extra edge: brings {prompt} (the research topic) from get_user_input
    # directly to answer_prompt_builder, forming a diamond in the DAG.
    graph.connect("get_user_input", "answer_prompt_builder")

    executor = WorkflowExecutor(graph)
    ctx = await executor.run(initial_input={"prompt": ""})
    cr = ctx.get_output("answer_specialist")

    print(cr.response)


if __name__ == "__main__":
    anyio.run(main)