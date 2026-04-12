# Tutorial: Criando um Agente Conversacional com Loop no AgenticBlocks

Este tutorial demonstra como construir um chatbot operando em um loop contínuo de terminal. Utilizando a estrutura de grafos do `agenticblocks`, vamos conectar um modelo de linguagem (LLM) a ferramentas de entrada e saída, criando um ciclo de execução que só é interrompido por um comando específico do usuário.

## Pré-requisitos

Certifique-se de importar os módulos essenciais para lidar com operações assíncronas, gerenciamento do grafo e definição de ferramentas:

```python
import asyncio
import os
import sys
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor
from agenticblocks.blocks.llm.agent import LLMAgentBlock, AgentInput, AgentOutput
from agenticblocks import as_tool
```

---

## Passo 1: Observabilidade e Manipulação de Prompt

Para entender melhor como o agente está "pensando", é útil estender a classe base do agente para imprimir o prompt exato que está sendo enviado ao modelo em cada iteração.

Além disso, precisamos de uma função para construir e atualizar dinamicamente o contexto do prompt a cada ciclo do chat.

```python
# Modifica o prompt a cada iteração, focando na última mensagem do usuário
def build_chat_prompt(orig, iteration, producer, feedback):
    lines = feedback.splitlines()
    last_user = next((l for l in reversed(lines) if l.startswith("User:")), "")
    return (
        f"{feedback}\n\n"
        f"[Focus on answering: {last_user}]"
    )

# Agente customizado que imprime o prompt no terminal antes de executar
class ObservableLLMAgent(LLMAgentBlock):
    async def run(self, input: AgentInput) -> AgentOutput:
        print("-" * 94)
        print(f"[{self.name}] Prompt recebido: {input.prompt}")
        print("-" * 94)
        return await super().run(input)
```

---

## Passo 2: Definindo as Ferramentas (Tools)

O `agenticblocks` permite transformar funções Python comuns em ferramentas que o grafo ou o agente podem utilizar, utilizando o decorador `@as_tool`. Vamos definir três ferramentas principais: controle de estado, entrada de dados e saída de dados.

```python
chat_history = []

# 1. Ferramenta para verificar se o chat deve ser encerrado
@as_tool(name="check_done")
def check_done(last_message: str) -> dict:
    # Se o usuário digitar /bye, o loop é considerado concluído (is_valid=True)
    if last_message and "/bye" in last_message:
        return {"is_valid": True, "feedback": ""}
    
    # Caso contrário, continua o loop retornando o histórico atualizado
    hist = "\n".join(chat_history)
    return {"is_valid": False, "feedback": f"history: {hist}"}

# 2. Ferramenta para capturar a entrada do usuário via terminal
@as_tool(name="get_user_input")
def get_user_input() -> dict:
    print("Você: ", end="")
    user_input = input()
    chat_history.append(f"User: {user_input}")
    return {"prompt": user_input}

# 3. Ferramenta obrigatória para o agente imprimir e salvar sua resposta
@as_tool(name="print_researcher_response", description="Imprime a resposta do pesquisador e adiciona ao histórico de chat")
def print_researcher_response(response: str):
    print(f"Researcher: {response}")
    chat_history.append(f"Researcher: {response}")
```

---

## Passo 3: Construindo o Grafo e Configurando o Agente

Agora, dentro da nossa função principal, inicializamos o grafo e configuramos o nó central de inteligência: o `ObservableLLMAgent`. 

Nesta configuração, instruímos explicitamente o modelo (neste caso, `ollama/mistral-nemo:latest`) a assumir a persona de um assistente de pesquisa e forçamos o uso da ferramenta de impressão usando os parâmetros de `litellm_kwargs`.

```python
async def main():
    # Inicializa o grafo de fluxo de trabalho
    graph = WorkflowGraph()

    # Instancia o bloco do agente
    agent_block = ObservableLLMAgent(
        name="research_agent",
        model="ollama/mistral-nemo:latest",
        description="Agente de pesquisa para responder perguntas do usuário",
        system_prompt="""Você é um assistente de pesquisa especializado em ajudar pesquisadores
        sobre os mais diversos tópicos. Use o teu conhecimento intrínseco para responder às 
        perguntas do usuário. Você vai ser o primeiro a falar, dê as boas vindas e pergunte:
        Sobre o que minha divindade quer saber?""",
        tools=[print_researcher_response],
        max_iterations=1,
        # Força o LLM a sempre utilizar a ferramenta para responder
        litellm_kwargs={
            "temperature": 0.7, 
            "tool_choice": {"type": "function", "function": {"name": "print_researcher_response"}}
        }
    )

    # Adiciona todos os blocos construídos ao grafo
    graph.add_block(agent_block)
    graph.add_block(get_user_input)
    graph.add_block(check_done)
```

---

## Passo 4: Criando o Ciclo de Execução

A mágica de uma conversa contínua acontece conectando os blocos em um ciclo. A sequência definida dita a ordem de execução, e a condição de parada controla quando o ciclo deve ser rompido.

```python
    # Define o ciclo iterativo de conversa
    graph.add_cycle(
        name="chat_loop",
        sequence=["research_agent", "get_user_input", "check_done"],
        condition_block="check_done", # O bloco que dita se o loop termina
        max_iterations=1000,          # Limite de segurança
        augment_fn=build_chat_prompt  # Função que formata o contexto a cada turno
    )
```
*A sequência de execução será:*
1. O agente (`research_agent`) processa o prompt e responde usando a ferramenta de impressão.
2. O sistema aguarda a entrada do usuário (`get_user_input`).
3. O sistema checa se a condição de parada foi atingida (`check_done`). Se não for, o `augment_fn` atualiza o prompt e o ciclo reinicia.

---

## Passo 5: Executando o Fluxo de Trabalho

Finalmente, criamos o executor e disparamos o fluxo de trabalho com um gatilho inicial.

```python
    # Cria o executor com logs ativados
    executor = WorkflowExecutor(graph, verbose=True)

    # Dispara a execução fornecendo o estado inicial
    ctx = await executor.run(
        initial_input={"prompt": "Você está em uma conversa contínua. Responda à mensagem mais recente do histórico."}
    )
    
    # Extrai e exibe o resultado final do ciclo de chat
    cr = ctx.cycle_results.get("chat_loop")
    if cr:
        print("Chat conversation ", cr.output)

# Inicializador padrão do asyncio
if __name__ == "__main__":
    asyncio.run(main())
```

### Resumo do Comportamento Esperado:
Ao executar o script, o terminal exibirá logs indicando o início do fluxo. O agente analisará o *system prompt* e fornecerá a saudação inicial através da função `print_researcher_response`. O fluxo então pausará esperando sua interação no terminal via `input()`. O loop se manterá ativo, alimentando a inteligência com o histórico acumulado até que o comando `/bye` seja detectado, momento em que o grafo encerrará a execução com sucesso.