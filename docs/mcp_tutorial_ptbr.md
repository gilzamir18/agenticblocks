# Tutorial: Integrando Servidores MCP no AgenticBlocks

Neste tutorial, vamos explorar como expandir as capacidades dos seus agentes no `agenticblocks` utilizando o **Model Context Protocol (MCP)**. O MCP permite conectar modelos de linguagem a ferramentas e fontes de dados externas de forma padronizada.

Neste exemplo prático, construiremos um **Agente de Pesquisa** que utiliza dois servidores MCP diferentes rodando localmente via `uvx`:
1.  **`duckduckgo-mcp-server`**: Para realizar buscas na web.
2.  **`mcp-server-fetch`**: Para extrair o conteúdo das URLs encontradas.

---

## Pré-requisitos

Certifique-se de ter o `agenticblocks` instalado, assim como o `uv` (gerenciador de pacotes Python, usado aqui através do comando `uvx` para rodar os servidores MCP dinamicamente). Você também precisará de um modelo rodando localmente no Ollama (neste exemplo, `mistral-nemo`).

## Passo 1: Importações e Ferramenta de Entrada

Primeiro, importamos os componentes necessários do `agenticblocks` e da biblioteca assíncrona `anyio`. Também criamos uma ferramenta customizada para capturar o tema da pesquisa do usuário.

```python
import anyio
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor
from agenticblocks.blocks.llm.agent import LLMAgentBlock, AgentInput
from agenticblocks import as_tool
from agenticblocks.tools.mcp_client import MCPClientBridge

@as_tool(name="get_user_input")
async def get_user_input(prompt: str) -> AgentInput:
    print("Sobre o que você quer pesquisar: ", end="")
    user_input = input()
    return AgentInput(prompt=f"Pesquise sobre o tópico {user_input}")
```

## Passo 2: Conectando aos Servidores MCP

A classe `MCPClientBridge` é a ponte entre o seu código e o servidor MCP. Nós iniciamos os servidores como subprocessos passando o comando e os argumentos. 

Ao chamar `await mcp.connect()`, o cliente extrai automaticamente todas as ferramentas disponibilizadas pelo servidor e retorna uma lista de blocos proxy (`MCPProxyBlock`) prontos para uso.

```python
async def main():
    # 1. Conecta ao servidor MCP que extrai conteúdo de páginas (fetch)
    mcp_fetch = MCPClientBridge(command="uvx", args=["mcp-server-fetch"])
    mcp_tools_fetch = await mcp_fetch.connect()  # Extrai ferramentas de fetch
    
    # 2. Conecta ao servidor MCP de buscas do DuckDuckGo
    mcp_search = MCPClientBridge(command="uvx", args=["duckduckgo-mcp-server"])
    mcp_tools_search = await mcp_search.connect()  # Extrai ferramentas de busca
```

## Passo 3: Configurando o Agente de Pesquisa

Agora, criamos o nosso agente LLM. A parte mais importante aqui é a propriedade `tools`. Nós simplesmente somamos as listas de ferramentas retornadas pelos nossos dois servidores MCP (`mcp_tools_fetch + mcp_tools_search`). O Agente agora tem o poder de buscar no DuckDuckGo e ler páginas da web!

```python
    graph = WorkflowGraph()

    agent_block = LLMAgentBlock(
        name="research_agent",
        model="ollama/mistral-nemo:latest",
        description="Agente de pesquisa",
        system_prompt="""Você é um assistente de pesquisa. Ao receber um tópico, use as ferramentas de busca e fetch para extrair informações de URLs. Escreva um relatório final em estilo jornalístico. Regra estrita: entregue apenas texto em prosa, sem absolutamente nenhuma formatação, listas ou marcações markdown.""",
        # Injetando as ferramentas MCP no Agente:
        tools=mcp_tools_fetch + mcp_tools_search,   
        max_iterations=10,
        on_max_iterations="return_last",
        litellm_kwargs={"temperature": 0.7, "tool_choice": "auto", "num_ctx": 32000}
    )
```

## Passo 4: Construindo o Grafo e Executando

Com o agente pronto, definimos o fluxo de execução. O grafo começará pedindo a entrada do usuário (`get_user_input`) e passará o resultado diretamente para o agente de pesquisa (`agent_block`).

```python
    # Define a sequência: Entrada do Usuário -> Agente
    graph.add_sequence(get_user_input, agent_block)

    # Executa o workflow
    executor = WorkflowExecutor(graph)
    ctx = await executor.run(initial_input={"prompt": ""})
    
    # Extrai a resposta final do agente
    cr = ctx.get_output("research_agent")
    print("\nRelatório Final:\n")
    print(cr.response)
```

## Passo 5: Limpeza e Desconexão

Como os servidores MCP estão rodando como subprocessos, é crucial garantir que eles sejam encerrados corretamente ao final da execução, evitando processos zumbis no seu sistema.

```python
    # Tenta desconectar e encerrar o servidor Fetch
    try:
        await mcp_fetch.disconnect()
    except RuntimeError:
        pass

    # Tenta desconectar e encerrar o servidor de Busca
    try:
        await mcp_search.disconnect()
    except Exception:
        pass

if __name__ == "__main__":
    # Usa anyio.run para executar o loop assíncrono principal
    anyio.run(main)
```

---

## Código Completo

Para facilitar, aqui está o código completo para você copiar e executar:

```python
import anyio
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor
from agenticblocks.blocks.llm.agent import LLMAgentBlock, AgentInput
from agenticblocks import as_tool
from agenticblocks.tools.mcp_client import MCPClientBridge

@as_tool(name="get_user_input")
async def get_user_input(prompt: str) -> AgentInput:
    print("Sobre o que você quer pesquisar: ", end="")
    user_input = input()
    return AgentInput(prompt=f"Pesquise sobre o tópico {user_input}")

async def main():
    # Conecta aos servidores MCP
    mcp_fetch = MCPClientBridge(command="uvx", args=["mcp-server-fetch"])
    mcp_tools_fetch = await mcp_fetch.connect()  
    
    mcp_search = MCPClientBridge(command="uvx", args=["duckduckgo-mcp-server"])
    mcp_tools_search = await mcp_search.connect()  
    
    graph = WorkflowGraph()

    agent_block = LLMAgentBlock(
        name="research_agent",
        model="ollama/mistral-nemo:latest",
        description="Agente de pesquisa",
        system_prompt="""Você é um assistente de pesquisa. Ao receber um tópico, use as ferramentas de busca e fetch para extrair informações de URLs. Escreva um relatório final em estilo jornalístico. Regra estrita: entregue apenas texto em prosa, sem absolutamente nenhuma formatação, listas ou marcações markdown.""",
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

    # Limpeza
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

### Resumo

Neste tutorial você aprendeu como:
1. Instanciar `MCPClientBridge` passando comandos CLI para iniciar servidores MCP.
2. Utilizar `await bridge.connect()` para expor ferramentas externas como blocos nativos do AgenticBlocks.
3. Injetar múltiplas listas de ferramentas MCP diretamente no `LLMAgentBlock`.
4. Gerenciar o ciclo de vida da conexão usando `disconnect()` de forma segura.
