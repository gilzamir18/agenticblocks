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

### 4. LLM Agent Autonomy & A2A
The library features `LLMAgentBlock`, a ready-to-use orchestrator that dynamically translates your other Blocks into Tools (Agent-to-Agent) smoothly.
- **Unbounded Reasoning Loop:** Operates completely natively with `max_iterations=None` by default to avoid breaking long autonomous tasks abruptly. Limit it explicitly per agent if needed.
- **Connection Pooling & Advanced API Parameters:** Pass HTTP client instances (e.g., `httpx.AsyncClient()`) or any specific API argument via `litellm_kwargs` to improve efficiency and skip initial TLS Handshake delays.

Check the `examples/` directory for full demos.

---

## <a name="português"></a>🇧🇷 Português

### Filosofia
Uma biblioteca para construir fluxos de agentes no estilo **Lego**. Cada passo do seu pipeline agêntico é um bloco auto-contido, com entradas e saídas rigorosamente tipadas via **Pydantic** e execução simultânea usando **AsyncIO** e grafos do **NetworkX**.

- **Forte tipagem**: Pydantic valida os encaixes e previne dependências não satisfeitas.
- **Encaixes padronizados**: Blocos só conhecem as próprias entradas e saídas. Workflows inteiros funcionam como blocos únicos.
- **Paralelismo Inteligente (Ondas)**: O motor dispara tarefas simultâneas (waves) sempre que as dependências de um bloco são resolvidas, otimizando a velocidade de conexões a APIs.

### Primeiros Passos

Instale o módulo de forma local editável:
```bash
pip install -e .
```

### 4. Autonomia com Agentes LLM & A2A
O módulo traz o `LLMAgentBlock`, um orquestrador pronto que abstrai e converte seus sub-blocos transparentemente em tools nativas.
- **Raciocínio Ilimitado:** Sem amarras (`max_iterations=None` como padrão) para não abortar tarefas autônomas demoradas, permitindo definição exata pontualmente.
- **Connection Pooling:** Aceita passagem de Sessões HTTP e dezenas de parâmetros estendidos via argumento `litellm_kwargs` para zerar o atraso inicial nas requisições do seu loop de execução.

Explore os laboratórios interativos completos dentro da pasta `examples/`:
- `01_hello_world.py`: Simulação básica e limpa do tutorial inicial.
- `03_mcp_a2a_agent.py`: Exemplo do framework criando pontes Automáticas pra LLMs e lidando com delegação de chamadas entre dois agentes LLM em Loop (A2A).
- `05_basic_blocks.py`: Teste rígido nativo para aferição de Overheads.
