# agenticblocks 🧱

*A composable building block library for AI agent workflows. / Uma biblioteca componível para construir fluxos de agentes de IA.*

[🇺🇸 English](#english) | [🇧🇷 Português](#português)

---

## <a name="english"></a>🇺🇸 English

### Philosophy
A library to build agent workflows like **Lego blocks**. Each step in your agentic pipeline is a self-contained block, with strictly typed inputs and outputs via **Pydantic** and natively concurrent execution using **AsyncIO** and **NetworkX** graphs.

- **Strong typing**: Pydantic validates connections and prevents unmatched dependencies between LLM tool calls.
- **Standardized connections**: Blocks only know their own inputs and outputs. Thus, entire workflows can act as single blocks later.
- **Smart Parallelism (Waves)**: The asyncio engine fires simultaneous tasks (waves) whenever dependencies are resolved, maximizing API speed.
- **Native Cycles**: Declare bounded feedback loops directly in the graph (`add_cycle()`). The executor handles iteration, feedback propagation, and exit conditions automatically.
- **Functions as Tools**: Any plain Python function (sync or async) becomes a block with `@as_tool` — no class boilerplate required.
- **Focus on local open-source models**: small models run well with this library, as we provide ready-made blocks that handle their limitations, such as HeuristicLLMAgentBlock, which heuristically extracts tool calls in JSON format from plain text and executes them transparently. See more in [docs/heuristicagent.md](docs/heuristicagent.md).

### Getting Started

Install the module locally for development:
```bash
pip install -e .
```

#### 1. Define Input and Output Models
```python
from pydantic import BaseModel

class HelloInput(BaseModel):
    name: str

class HelloOutput(BaseModel):
    greeting: str
```

#### 2. Create the Logic Block
```python
from agenticblocks.core.block import Block

class HelloWorldBlock(Block[HelloInput, HelloOutput]):
    name: str = "say_hello"
    
    async def run(self, input: HelloInput) -> HelloOutput:
        msg = f"Hello, {input.name}! Welcome to agenticblocks."
        return HelloOutput(greeting=msg)
```

#### 3. Connect and Execute
```python
import asyncio
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor

async def main():
    graph = WorkflowGraph()
    graph.add_block(HelloWorldBlock(name="say_hello"))

    executor = WorkflowExecutor(graph)
    ctx = await executor.run(initial_input={"name": "Alice"})
    
    print(ctx.get_output("say_hello").greeting)

asyncio.run(main())
```

#### 4. Functions as Tools

Any Python function can be registered as a block with `@as_tool`. Both sync and async functions are supported — sync functions run in a thread pool automatically.

```python
from agenticblocks import as_tool
from agenticblocks.blocks.llm.agent import LLMAgentBlock

@as_tool
async def fetch_weather(city: str) -> str:
    """Returns the current weather for a city."""
    return f"Sunny in {city}."

agent = LLMAgentBlock(
    name="assistant",
    model="gpt-4o-mini",
    tools=[fetch_weather],   # same interface as any Block
)
```

#### 5. LLM Agent Autonomy & A2A

`LLMAgentBlock` is a ready-to-use orchestrator that dynamically translates your other Blocks into Tools (Agent-to-Agent) seamlessly.

- **Bounded tool loop**: `max_tool_calls` prevents runaway loops.
- **A2A bridging**: sub-agents are called as tools transparently — the parent LLM receives only the text response, not raw JSON metadata.
- **Connection Pooling**: Pass any `litellm_kwargs` (HTTP clients, timeouts, etc.) to optimize API performance.
- **Iteration Hooks**: Use the `on_iteration` parameter to inject callbacks that execute at the beginning of each loop iteration. These callbacks receive the current iteration number and the list of messages, allowing for real-time monitoring, logging, or dynamic prompt adjustments during the agent's execution.

#### 6. Advanced Flow Control & Heuristics

- **PromptBuilderBlock**: Merges outputs from multiple predecessors into a single formatted `AgentInput` prompt using Python format-strings. Useful for "diamond" graph patterns (e.g., feeding both the original topic and a search report to a final summarizer).
- **HeuristicLLMAgentBlock**: A specialized agent for models with weak native tool support (like smaller local models). It heuristically parses hallucinated JSON tool calls out of plain text responses and executes them transparently.

#### 7. Native Feedback Cycles

Declare validator loops directly in the graph without any wrapper block:

```python
from agenticblocks import as_tool
from agenticblocks.core.graph import WorkflowGraph

@as_tool
def validate_output(content: str) -> dict:
    ok = len(content.split()) >= 100
    return {"is_valid": ok, "feedback": "Too short." if not ok else ""}

graph = WorkflowGraph()
graph.add_block(writer)
graph.add_block(validate_output)

graph.add_cycle(
    name="refine",
    edges=[("writer", "validate_output")],
    condition_block="validate_output",
    max_iterations=3,
)
# Downstream nodes connect to the cycle output as a normal node
graph.connect("refine", "publisher")
```

The executor runs the cycle, propagates feedback to the producer on each rejection, and stores the result in `ctx` under the cycle name.

### Examples & Model Recommendations

It is recommended to install [Ollama](https://ollama.com/) with the model `granite4:1b` (`ollama run granite4:1b`) to test the examples locally. Alternatively, you can modify the examples to use a commercial API, such as Gemini (`gemini/gemini-2.0-flash`) or OpenAI.

| Example | Description |
|---|---|
| `01_hello_world.py` | Minimal block + graph + executor setup |
| `02_llm_pipeline.py` | Parallel wave execution with multiple blocks |
| `03_mcp_a2a_agent.py` | MCP bridge + Agent-to-Agent (A2A) tool delegation |
| `04_mcp_python_native.py` | Native Python MCP server |
| `05_basic_blocks.py` | Overhead benchmarking |
| `06_functionastool.py` | `@as_tool` decorator for plain functions |
| `07_validator_loop.py` | Native graph cycle with producer + validator feedback loop |

> **Note:** Quantized or small models like `granite` may produce lower-quality reasoning and struggle with native tool calling. For reliable local tool usage, use `llama3.1` or `mistral-nemo`. If using `granite4`, prefer the `HeuristicLLMAgentBlock` to capture hallucinated JSON tool calls. Large commercial models (OpenAI, Gemini, Anthropic) yield excellent results but require an API key.

---

## <a name="português"></a>🇧🇷 Português

### Filosofia
Uma biblioteca para construir fluxos de agentes no estilo **Lego**. Cada passo do seu pipeline agêntico é um bloco auto-contido, com entradas e saídas rigorosamente tipadas via **Pydantic** e execução simultânea usando **AsyncIO** e grafos do **NetworkX**.

- **Forte tipagem**: Pydantic valida os encaixes e previne dependências não satisfeitas.
- **Encaixes padronizados**: Blocos só conhecem as próprias entradas e saídas. Workflows inteiros funcionam como blocos únicos.
- **Paralelismo Inteligente (Ondas)**: O motor dispara tarefas simultâneas sempre que as dependências de um bloco são resolvidas.
- **Ciclos Nativos**: Declare loops de feedback diretamente no grafo com `add_cycle()`. O executor gerencia iteração, propagação de feedback e condição de saída automaticamente.
- **Funções como Ferramentas**: Qualquer função Python (síncrona ou async) vira um bloco com `@as_tool` — sem boilerplate de classe.
- **Foco em modelos locais open-source**: modelos pequenos rodam bem com esta biblioteca, pois provemos blocos prontos que lidam com suas limitações, como HeuristicLLMAgentBlock, que extrai heuristicamente chamadas de ferramenta em formato JSON do texto plano e as executa de forma transparente. Veja mais em [docs/heuristicagent.md](docs/heuristicagent.md).



### Primeiros Passos

Instale o módulo de forma local editável:
```bash
pip install -e .
```

#### 1–3. Blocos, Grafo e Execução

A estrutura básica é idêntica ao tutorial acima (seção em inglês): defina modelos Pydantic, crie um `Block`, adicione ao `WorkflowGraph` e execute com `WorkflowExecutor`.

#### 4. Funções como Ferramentas

Qualquer função pode ser registrada como bloco com `@as_tool`. Funções síncronas rodam em thread pool automaticamente.

```python
from agenticblocks import as_tool

@as_tool
def buscar_clima(cidade: str) -> str:
    """Retorna o clima atual de uma cidade."""
    return f"Ensolarado em {cidade}."
```

#### 5. Autonomia com Agentes LLM & A2A

O `LLMAgentBlock` abstrai e converte sub-blocos em ferramentas nativas (A2A). Destaques:

- **`max_tool_calls`**: Limita o loop de ferramentas para evitar execuções infinitas.
- **A2A transparente**: Agentes subordinados são chamados como ferramentas; o agente pai recebe apenas o texto da resposta, sem metadados JSON brutos.
- **Connection Pooling**: Aceite sessões HTTP e parâmetros estendidos via `litellm_kwargs`.

#### 6. Controle de Fluxo Avançado & Heurísticas

- **PromptBuilderBlock**: Mescla saídas de múltiplos predecessores em um único prompt `AgentInput` formatado. Ideal para padrões de grafo em "diamante".
- **HeuristicLLMAgentBlock**: Agente especializado para modelos com suporte fraco a chamadas de ferramentas (como modelos locais pequenos). Ele extrai heuristicamente chamadas de ferramenta em formato JSON do texto plano e as executa de forma transparente.
- **Callbacks de Iteração (`on_iteration`)**: Use o parâmetro `on_iteration` para injetar funções de callback que serão executadas no início de cada iteração do loop do agente. Isso permite monitoramento em tempo real, logging ou ajustes dinâmicos no prompt durante a execução do agente.

#### 7. Ciclos de Feedback Nativos

Declare um loop validador diretamente no grafo — sem bloco orquestrador especial:

```python
from agenticblocks import as_tool
from agenticblocks.core.graph import WorkflowGraph

@as_tool
def validar(content: str) -> dict:
    ok = len(content.split()) >= 100
    return {"is_valid": ok, "feedback": "Muito curto." if not ok else ""}

graph = WorkflowGraph()
graph.add_block(escritor)
graph.add_block(validar)

graph.add_cycle(
    name="refinar",
    edges=[("escritor", "validar")],
    condition_block="validar",
    max_iterations=3,
)
graph.connect("refinar", "publicador")
```

O executor itera automaticamente, injeta o feedback no prompt do produtor a cada rejeição e disponibiliza o resultado final em `ctx.get_output("refinar")`.

### Exemplos & Modelos

Recomenda-se instalar o [Ollama](https://ollama.com/) com o modelo `granite4:1b` para testar localmente. Alternativamente, use uma API comercial como Gemini ou OpenAI.

| Exemplo | Descrição |
|---|---|
| `01_hello_world.py` | Setup mínimo: bloco + grafo + executor |
| `02_llm_pipeline.py` | Execução paralela em waves |
| `03_mcp_a2a_agent.py` | Bridge MCP + delegação A2A entre agentes |
| `04_mcp_python_native.py` | Servidor MCP nativo em Python |
| `05_basic_blocks.py` | Benchmark de overhead |
| `06_functionastool.py` | Decorator `@as_tool` para funções simples |
| `07_validator_loop.py` | Ciclo nativo no grafo: produtor + validador com feedback |

> **Atenção:** Modelos quantizados ou menores podem produzir resultados abaixo do esperado e ter dificuldade com chamadas nativas de ferramentas. Para uso local confiável de ferramentas, prefira `llama3.1` ou `mistral-nemo`. Caso use `granite4`, utilize o `HeuristicLLMAgentBlock`. Modelos comerciais grandes geram excelentes resultados, mas exigem configuração de API KEY.
