import anyio
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor
from agenticblocks.blocks.llm.agent import LLMAgentBlock, AgentInput
from agenticblocks.blocks.llm.heuristic_agent import HeuristicLLMAgentBlock
from agenticblocks import as_tool
from agenticblocks.tools.mcp_client import MCPClientBridge

@as_tool(name="get_user_input")
async def get_user_input(prompt: str) -> AgentInput:
    print("Sobre o que você quer pesquisar: ", end="")
    user_input = input()
    return AgentInput(prompt=f"Pesquise sobre o tópico {user_input}")

async def main():
    # Conecta ao servidor MCP fetch (inicia como subprocesso)
    mcp_fetch = MCPClientBridge(command="uvx", args=["mcp-server-fetch"])
    mcp_tools_fetch = await mcp_fetch.connect()  # retorna lista de MCPProxyBlock
    
    # Conecta ao servidor MCP fetch (inicia como subprocesso)
    mcp_search = MCPClientBridge(command="uvx", args=["duckduckgo-mcp-server"])
    mcp_tools_search = await mcp_search.connect()  # retorna lista de MCPProxyBlock
    
    graph = WorkflowGraph()

    # Adiciona o bloco de agente ao grafo
    agent_block = LLMAgentBlock(
        name="research_agent",
        use_shared_router=True,
        model="ollama/mistral-nemo:latest",
        description="Agente de pesquisa",
        system_prompt="""Você é um assistente de pesquisa especializado em ajudar pesquisadores
        sobre os mais diversos tópicos. Você vai receber uma consulta sobre um tópico dado
        pelo usuário e vai produzir um relatório sobre este tópico. Para isso, faça uma busca 
        em fontes de dados usando as ferramentas disponíveis. Chame uma vez cada ferramenta usando 
        palavras-chaves correspondentes ao tópico informado pelo usuário. 
        Use apenas dados retornados pelas ferramentas.
        Exemplo de tópico do usuário e palavras-chaves correspondentes que você deve usar nas
        ferramentas:
        
        Tópico: Brasil, Palabras-Chave: Brasil, brasil, países emergentes, país emergente
        Tópico: economia, Palavras-Chave: PIB, peço, petróleo, mercado, ...

        Estruture a resposta final destacando as informações relevantes de acordo com o tópico 
        pesquisado.
        """,
        tools=mcp_tools_search+mcp_tools_fetch,
        debug=True,
        max_iterations=2,
        on_max_iterations="return_last", #it returns the last message.
        litellm_kwargs={"temperature": 1.0, "tool_choice": "auto"}
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
