# Implementação de Memória (MemGPT Style)

A biblioteca `AgenticBlocks` fornece abstrações e implementações prontas para o gerenciamento avançado de memória de agentes, inspirado na arquitetura do **MemGPT**.

A memória de um agente avançado é dividida em dois tipos principais:
1. **Archival Memory (Memória de Arquivamento):** Um armazenamento de longo prazo baseado em semântica (busca vetorial). Ideal para carregar documentos extensos e base de conhecimento.
2. **Recall Memory (Memória de Recordação):** Um armazenamento temporal para salvar o histórico de conversas e eventos. Ideal para buscas exatas (palavras-chave) ou resgate de histórico passado.

## 1. Archival Memory (ChromaDB)

Por padrão, fornecemos o `ChromaArchivalMemory`, que utiliza o banco de dados vetorial [ChromaDB](https://docs.trychroma.com/) para converter textos em *embeddings* e buscar os documentos mais próximos semanticamente.

### Como Instalar
Certifique-se de que o pacote extra está instalado:
```bash
pip install chromadb
```

### Como Usar

```python
from agenticblocks.blocks.memory import ChromaArchivalMemory
from agenticblocks.core.function_block import as_tool

# Instancia a memória (pode ser persistida passando persist_directory)
archival = ChromaArchivalMemory(collection_name="conhecimento_geral")

# Inserindo dados (geralmente feito offline ou pelo próprio agente)
archival.insert(
    content="O projeto TasteFast é uma lanchonete rápida com pedidos 100% digitais.",
    metadata={"source": "briefing.txt"}
)

# Criando a ferramenta para o LLM usar
@as_tool(name="search_archival", description="Busca conhecimentos gerais por significado semântico.")
def search_archival(query: str, page: int = 1) -> str:
    resultados = archival.search(query, page=page, page_size=3)
    if not resultados:
        return "Nenhuma informação encontrada."
    
    # Formata os resultados de forma amigável para o LLM
    text = f"--- Página {page} ---\n"
    for r in resultados:
        text += f"- {r['content']} (Meta: {r['metadata']})\n"
    return text
```

## 2. Recall Memory (SQLite)

O histórico de interação contínua do agente e usuário é guardado na `Recall Memory`. Utilizamos o `SQLiteRecallMemory` que usa o `sqlite3` nativo do Python, dispensando qualquer dependência externa e garantindo gravação rápida e segura do histórico de interação.

### Como Usar

```python
from agenticblocks.blocks.memory import SQLiteRecallMemory
from agenticblocks.core.function_block import as_tool

# Instancia o banco de dados (pode ser ':memory:' para efêmero ou um arquivo como 'recall.db')
recall = SQLiteRecallMemory(db_path="historico_agente.db")

# Simulando inserção automática no log de conversa
recall.append_message(role="user", content="Oi, meu nome é Gil.")
recall.append_message(role="assistant", content="Olá Gil! Como posso ajudar?")

# Criando a ferramenta para o LLM usar
@as_tool(name="search_recall", description="Busca mensagens passadas em conversas com o usuário usando palavras-chave.")
def search_recall(keyword: str) -> str:
    resultados = recall.search_keyword(keyword, limit=5)
    if not resultados:
        return "Nenhuma lembrança encontrada."
    
    text = "--- Histórico Recente ---\n"
    for r in resultados:
        text += f"[{r['timestamp']}] {r['role'].upper()}: {r['content']}\n"
    return text
```

## 3. Integrando no Agente (MemGPTAgentBlock com Heartbeats)

Para que o agente possa gerenciar ativamente a sua memória seguindo o protocolo rigoroso do MemGPT, onde **todas as respostas** devem ser chamadas de ferramentas e existe uma contagem de *heartbeats*, utilizamos o `MemGPTAgentBlock`.

```python
from agenticblocks.blocks.llm.memgpt_agent import MemGPTAgentBlock

agente = MemGPTAgentBlock(
    name="agente_memgpt",
    tools=[search_archival, search_recall], # A ferramenta `send_message` é embutida automaticamente
    max_heartbeats=5,
    system_prompt='''
Você é um assistente persistente com memória de longo e curto prazo.
- Use a ferramenta `search_recall` para lembrar de coisas que o usuário disse no passado.
- Use a ferramenta `search_archival` para buscar conhecimentos da base da empresa.
- IMPORTANTE: você deve sempre usar a ferramenta `send_message` para falar com o usuário.
'''
)

# Durante a execução, o agente decidirá de forma autônoma 
# quando chamar as ferramentas baseadas no que o usuário perguntar,
# controlando seus próprios heartbeats.
```

## Arquitetura Modular (Interfaces)

Caso queira implementar sua própria lógica de persistência (por exemplo, `Pinecone` para Archival ou `PostgreSQL` para Recall), você pode herdar as classes base `BaseArchivalMemory` e `BaseRecallMemory` de `agenticblocks.blocks.memory.base`. Elas definem os contratos obrigatórios de `insert()` e `search()`.
