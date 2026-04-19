import anyio
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import get_model, get_litellm_kwargs

from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor
from agenticblocks.blocks.llm.agent import LLMAgentBlock, AgentInput
from agenticblocks.blocks.llm.heuristic_agent import HeuristicLLMAgentBlock
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
        print(to_text)
        return to_text

async def main():
    graph = WorkflowGraph()

    research_agent = HeuristicLLMAgentBlock(
        name="research_agent",
        model=get_model(),
        description="Research agent",
        system_prompt="""You are a research assistant. Given a topic, use the web_search tool to gather 
                      information about the topic. Write a final structured resport identifying 
                      the main ideas, important facts and relevant information. Highlight 
                      the dates (day, month, year), names, places and any specific keywords
                        that might be relevant to the topic.
                      """,
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
        model="ollama/mistral-nemo:latest",
        description="Answer specialist",
        system_prompt="""You are an answer specialist. You will receive a research
                        topic and a detailed report. Produce a clear, direct answer 
                        to the topic in fluent prose based only on report. Answer in portuguese brazil.""",
        litellm_kwargs={"temperature": 0.1, "num_ctx": 8192, "max_tokens":200},
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
    
 # === VERIFICANDO O OUTPUT DO BUILDER ===
    builder_out = ctx.get_output("answer_prompt_builder")
    print("\n" + "="*50)
    print("🔍 PROMPT MONTADO PELO BUILDER:")
    print("="*50)
    # Como ele retorna um AgentInput, o dado está em .prompt
    print(builder_out.prompt if builder_out else "Nenhum output!")
    print("="*50 + "\n")
    
    # === VERIFICANDO O OUTPUT DO RESEARCH AGENT ===
    research_out = ctx.get_output("research_agent")
    print("🔍 RESULTADO DA PESQUISA BRUTA:")
    print("="*50)
    print(research_out.response if research_out else "Nenhum output!")
    print("="*50 + "\n")
    cr = ctx.get_output("answer_specialist")
    print(cr.response)

if __name__ == "__main__":
    anyio.run(main)