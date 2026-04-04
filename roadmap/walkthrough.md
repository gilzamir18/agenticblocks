# Walkthrough: AgentBlocks

Sua arquitetura assíncrona baseada na filosofia de blocos encaixáveis (tipo Lego) já está devidamente refatorada e validada no código.

## O que foi construído

> [!NOTE]
> Todos os arquivos de simulação anteriores (`base.py`, `runtime.py`) foram substituídos por sua correspondente estrutura final do projeto.

### Core e Camada Analítica
- [core/block.py](file:///c:/Users/gilza/projetos/agentblocks/core/block.py): Introduz a classe primordial `Block[Input, Output]`. Ela extrai automagicamente as entradas (schemas) do método assíncrono `run()`, permitindo validação implícita nativa pela engine Pydantic.
- [core/graph.py](file:///c:/Users/gilza/projetos/agentblocks/core/graph.py): Engloba todo o modelo `NetworkX` responsável por construir restrições diretas e validadas sobre como um schema interage com o do próximo Nó (Arestas e Vertices).

### Runtime e Motor de Workflow
- [runtime/executor.py](file:///c:/Users/gilza/projetos/agentblocks/runtime/executor.py): O motor "mágico" capaz de simular a Execução Baseado em Ondas. Usamos o princípio das ordens topológicas iterativas para executar em lotes com uso inteligente de recursos usando `asyncio.gather`!
- [runtime/retry.py](file:///c:/Users/gilza/projetos/agentblocks/runtime/retry.py): Decorator simplificado de wrapper via *wraps* para atrasar ou forçar uma repetição controlada se a API/LLM falhar.
- [runtime/state.py](file:///c:/Users/gilza/projetos/agentblocks/runtime/state.py): As Context Vars para rastrear de forma stateful os tempos e estados das respostas sem interferência concorrente de locks explícitos em excesso entre tasks.

## Validation Results

Criamos o arquivo principal [main.py](file:///c:/Users/gilza/projetos/agentblocks/main.py) implementando quatro blocos que representam o caso de uso. O executor disparou o grafo, e você pode observar que **parse** e **enrich** rodam perfeitamente em modo simultâneo (paralelamente):

```text
Executando workflow...
  → iniciando: fetch
  ✓ fetch (109.0ms)
  → iniciando: enrich
  → iniciando: parse       
  ✓ parse (109.0ms)
  ✓ enrich (203.0ms)
  → iniciando: summarize
  ✓ summarize (500.0ms)   
Resumo Final: Summary of 'Mock content mapped' with {source: mock, verified: true}
```
