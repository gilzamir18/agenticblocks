# AgentBlocks 🧱

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
from agentblocks.core.block import Block

class HelloWorldBlock(Block[HelloInput, HelloOutput]):
    name: str = "say_hello"
    
    async def run(self, input: HelloInput) -> HelloOutput:
        msg = f"Hello, {input.name}! Welcome to AgentBlocks."
        return HelloOutput(greeting=msg)
```

#### 3. Connect and Execute
```python
import asyncio
from agentblocks.core.graph import WorkflowGraph
from agentblocks.runtime.executor import WorkflowExecutor

async def main():
    graph = WorkflowGraph()
    graph.add_block(HelloWorldBlock(name="say_hello"))

    executor = WorkflowExecutor(graph)
    ctx = await executor.run(initial_input={"name": "Alice"})
    
    print(ctx.get_output("say_hello").greeting)

asyncio.run(main())
```

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

A estrutura segue o modelo mostrado na sessão em inglês (Inglês: Input/Output Models, Logic Block, e Graph Execution). Consulte os scripts interativos e completos dentro da pasta `examples/`:
- `01_hello_world.py`: Simulação básica e limpa do tutorial inicial.
- `02_llm_pipeline.py`: Um pipeline completo demonstrando a paralelização de parsing de dados complexos com um mock de LLM Call.
